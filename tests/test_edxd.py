from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from bluesky import Msg, RunEngine
from bluesky import preprocessors as bpp
from bluesky_tiled_plugins import TiledWriter
from bluesky_tiled_plugins.clients.catalog_of_bluesky_runs import CatalogOfBlueskyRuns
from ophyd_async.core import (
    Device,
    get_mock_execute,
    get_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor

from hextools.detectors.germ import GeRMDetector
from hextools.edxd import (
    configure_test_pulses,
    edxd_2theta_tilt,
    edxd_calib_scan,
    edxd_count,
    edxd_custom_pos_list_grid,
    edxd_grid_scan,
    edxd_scan,
)
from hextools.motors import CollimatorTable, EDXDTable, SampleTower
from hextools.photon_delivery_system import Shutter, ShutterStatus

Docs = dict[str, list[dict[str, Any]]]

HDF5 = "application/x-hdf5"
TIFF = "multipart/related;type=image/tiff"


def _motors_in(device: Device) -> Generator[AsyncEpicsMotor]:
    if isinstance(device, AsyncEpicsMotor):
        yield device
        return
    for _, child in device.children():
        yield from _motors_in(child)


def _fast(*devices: Device) -> None:
    """Speed up the velocity-respecting mock motors so moves take milliseconds."""
    for device in devices:
        for motor in _motors_in(device):
            set_mock_value(motor.max_velocity, 10000)
            set_mock_value(motor.velocity, 1000)
            set_mock_value(motor.acceleration_time, 0.001)


def _run(RE: RunEngine, plan: Generator[Msg, Any, Any]) -> tuple[Docs, list[float]]:
    """Run ``plan``, returning its documents and every sleep it requested.

    Sleeps of a second or more (shutter settling, delays between repeats) are
    skipped; shorter ones, like the calibration scan's polling, still run.
    """
    docs: Docs = {}
    sleeps: list[float] = []

    def _skip_long_sleeps(msg: Msg) -> Msg:
        if msg.command != "sleep":
            return msg
        sleeps.append(msg.args[0])
        return msg._replace(args=(0,)) if msg.args[0] >= 1 else msg

    RE(
        bpp.msg_mutator(plan, _skip_long_sleeps),
        lambda name, doc: docs.setdefault(name, []).append(doc),
    )
    return docs, sleeps


def _events_by_run(docs: Docs) -> list[list[dict[str, Any]]]:
    descriptors = {d["uid"]: d["run_start"] for d in docs["descriptor"]}
    return [
        [e for e in docs["event"] if descriptors[e["descriptor"]] == start["uid"]]
        for start in docs["start"]
    ]


@pytest.fixture
def stage() -> tuple[AsyncEpicsMotor, AsyncEpicsMotor]:
    with init_devices(mock=True):
        x = AsyncEpicsMotor("X:")
        y = AsyncEpicsMotor("Y:")
    _fast(x, y)
    return x, y


@pytest.fixture
def tiled(
    RE: RunEngine, tiled_client: tuple[Path, CatalogOfBlueskyRuns]
) -> CatalogOfBlueskyRuns:
    _, client = tiled_client
    RE.subscribe(TiledWriter(client))
    return client


async def _assert_closed(*shutters: Shutter) -> None:
    for shutter in shutters:
        assert await shutter.status.get_value() == ShutterStatus.CLOSED


def _shutters_opened_before_first_count(
    messages: list[Msg], *shutters: Shutter
) -> bool:
    first_count = next(i for i, m in enumerate(messages) if m.command == "trigger")
    opened = {
        m.obj
        for m in messages[:first_count]
        if m.command == "set" and m.obj in shutters and m.args[0] is True
    }
    return opened == set(shutters)


# --- configure_test_pulses -----------------------------------------------------------


@pytest.mark.parametrize("use_test_pulses", [True, False])
@pytest.mark.parametrize("initially_enabled", [True, False])
async def test_configure_test_pulses(
    RE: RunEngine,
    germ: GeRMDetector,
    use_test_pulses: bool,
    initially_enabled: bool,
):
    set_mock_value(germ.driver.test_pulse_enable, initially_enabled)

    RE(
        configure_test_pulses(
            use_test_pulses,
            test_pulse_amplitude=50,
            test_pulse_freq=200,
            test_pulse_count=10,
            germ=germ,
        )
    )

    driver = germ.driver
    enables = [c.args[0] for c in get_mock_put(driver.test_pulse_enable).call_args_list]
    # The IOC acts on a change, so an already-matching state is toggled off first.
    if initially_enabled == use_test_pulses:
        assert enables == [not use_test_pulses, use_test_pulses]
    else:
        assert enables == [use_test_pulses]
    assert await driver.enable_all_tsen.get_value() == use_test_pulses
    settings = (
        await driver.test_pulse_amplitude.get_value(),
        await driver.test_pulse_frequency.get_value(),
        await driver.test_pulse_count.get_value(),
    )
    assert settings == ((50, 200, 10) if use_test_pulses else (0, 0, 0))


# --- edxd_count -----------------------------------------------------------------------


@pytest.mark.parametrize("description", [None, "silicon standard"])
async def test_edxd_count(
    RE: RunEngine,
    germ: GeRMDetector,
    tiled: CatalogOfBlueskyRuns,
    description: str | None,
):
    docs, _ = _run(RE, edxd_count(0.02, num=3, description=description, germ=germ))

    (start,) = docs["start"]
    assert start["plan_name"] == "edxd_count"
    assert start.get("description") == description
    assert docs["stop"][0]["exit_status"] == "success"
    assert await germ.driver.acquire_time.get_value() == 0.02

    # Three spectra frames, each matching its event's stats total.
    data = tiled[start["uid"]]["primary"]["germ"].read()
    frames = np.asarray(data).reshape(3, -1)
    totals = [e["data"]["germ-stats1-total"] for e in docs["event"]]
    assert totals == [float(f.sum()) for f in frames]


# --- edxd_scan ------------------------------------------------------------------------


@pytest.mark.parametrize(("save_as_tiff", "mimetype"), [(False, HDF5), (True, TIFF)])
async def test_edxd_scan(
    RE: RunEngine,
    germ: GeRMDetector,
    stage: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
    two_shutters: tuple[Shutter, Shutter],
    tiled: CatalogOfBlueskyRuns,
    save_as_tiff: bool,
    mimetype: str,
):
    x, _ = stage
    fe_shutter, photon_shutter = two_shutters
    messages: list[Msg] = []
    RE.msg_hook = messages.append  # type: ignore

    docs, _ = _run(
        RE,
        edxd_scan(
            x, 0, 2, 3,
            0.02,
            description="strain map",
            save_as_tiff=save_as_tiff,
            germ=germ,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
        ),
    )  # fmt: skip

    (start,) = docs["start"]
    assert start["plan_name"] == "edxd_scan"
    assert start["description"] == "strain map"
    assert [e["data"]["x"] for e in docs["event"]] == pytest.approx([0, 1, 2])
    assert {r["mimetype"] for r in docs["stream_resource"]} == {mimetype}
    data = tiled[start["uid"]]["primary"]["germ"].read()
    assert np.asarray(data).shape[0] == 3
    assert await germ.driver.acquire_time.get_value() == 0.02

    assert _shutters_opened_before_first_count(messages, fe_shutter, photon_shutter)
    await _assert_closed(fe_shutter, photon_shutter)
    # reset_position returns the motor to the start of the scan
    assert await x.user_readback.get_value() == pytest.approx(0)


async def test_edxd_scan_repeats_without_shutters_or_reset(
    RE: RunEngine,
    germ: GeRMDetector,
    stage: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
    two_shutters: tuple[Shutter, Shutter],
):
    x, _ = stage
    fe_shutter, photon_shutter = two_shutters
    messages: list[Msg] = []
    RE.msg_hook = messages.append  # type: ignore

    docs, sleeps = _run(
        RE,
        edxd_scan(
            x, 0, 2, 3,
            0.02,
            num_iterations=2,
            time_between_iterations=5.0,
            use_shutter=False,
            reset_position=False,
            germ=germ,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
        ),
    )  # fmt: skip

    assert [len(events) for events in _events_by_run(docs)] == [3, 3]
    # the delay, less the time the first iteration took, falls between the runs
    commands = [m.command for m in messages]
    first_close = commands.index("close_run")
    second_open = commands.index("open_run", first_close)
    assert "sleep" in commands[first_close:second_open]
    assert sleeps[0] == pytest.approx(5.0, abs=1.0)
    assert not get_mock_execute(fe_shutter.open_cmd).called
    assert not get_mock_execute(photon_shutter.open_cmd).called
    assert await x.user_readback.get_value() == pytest.approx(2)


# --- edxd_grid_scan -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("snake", "expected"),
    [
        (False, [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]),
        (True, [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]),
    ],
)
async def test_edxd_grid_scan(
    RE: RunEngine,
    germ: GeRMDetector,
    stage: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
    two_shutters: tuple[Shutter, Shutter],
    snake: bool,
    expected: list[tuple[float, float]],
):
    x, y = stage
    fe_shutter, photon_shutter = two_shutters
    # The grid scan always writes HDF5, whatever the detector was set to.
    germ.save_as_tiff()

    docs, _ = _run(
        RE,
        edxd_grid_scan(
            x, 0, 1, 2,
            y, 0, 2, 3,
            0.02,
            snake=snake,
            description="grid",
            germ=germ,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
        ),
    )  # fmt: skip

    (start,) = docs["start"]
    assert start["plan_name"] == "edxd_grid_scan"
    assert start["description"] == "grid"
    positions = [(e["data"]["x"], e["data"]["y"]) for e in docs["event"]]
    assert positions == [pytest.approx(p) for p in expected]
    assert {r["mimetype"] for r in docs["stream_resource"]} == {HDF5}
    await _assert_closed(fe_shutter, photon_shutter)
    assert await x.user_readback.get_value() == pytest.approx(0)
    assert await y.user_readback.get_value() == pytest.approx(0)


async def test_edxd_grid_scan_without_reset_stays_at_last_point(
    RE: RunEngine,
    germ: GeRMDetector,
    stage: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
    two_shutters: tuple[Shutter, Shutter],
):
    x, y = stage
    fe_shutter, photon_shutter = two_shutters

    _run(
        RE,
        edxd_grid_scan(
            x, 0, 1, 2,
            y, 0, 2, 3,
            0.02,
            reset_position=False,
            use_shutter=False,
            germ=germ,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
        ),
    )  # fmt: skip

    assert await x.user_readback.get_value() == pytest.approx(1)
    assert await y.user_readback.get_value() == pytest.approx(2)
    assert not get_mock_execute(photon_shutter.open_cmd).called


# --- edxd_custom_pos_list_grid --------------------------------------------------------


@pytest.mark.parametrize(("save_as_tiff", "mimetype"), [(False, HDF5), (True, TIFF)])
async def test_edxd_custom_pos_list_grid(
    RE: RunEngine,
    germ: GeRMDetector,
    stage: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
    two_shutters: tuple[Shutter, Shutter],
    save_as_tiff: bool,
    mimetype: str,
):
    x, y = stage
    fe_shutter, photon_shutter = two_shutters
    for motor, position in ((x, 5.0), (y, -1.0)):
        set_mock_value(motor.user_setpoint, position)
        set_mock_value(motor.user_readback, position)

    docs, _ = _run(
        RE,
        edxd_custom_pos_list_grid(
            x, [0.0, 1.0],
            y, [0.0], [1.0], [3],
            0.02,
            save_as_tiff=save_as_tiff,
            description="pos list",
            germ=germ,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
        ),
    )  # fmt: skip

    # One run per outer position, each a 3-point inner scan from the shared range.
    first, second = docs["start"]
    assert [s["outer_position"] for s in docs["start"]] == [0.0, 1.0]
    assert all(s["plan_name"] == "edxd_custom_pos_list_grid" for s in docs["start"])
    assert all(s["description"] == "pos list" for s in docs["start"])
    assert "initial_scan_uid" not in first
    assert second["initial_scan_uid"] == first["uid"]
    for events in _events_by_run(docs):
        assert [e["data"]["y"] for e in events] == pytest.approx([0, 0.5, 1])
    assert {r["mimetype"] for r in docs["stream_resource"]} == {mimetype}

    await _assert_closed(fe_shutter, photon_shutter)
    # both motors go back to where they were before the scan
    assert await x.user_readback.get_value() == pytest.approx(5.0)
    assert await y.user_readback.get_value() == pytest.approx(-1.0)


async def test_edxd_custom_pos_list_grid_per_position_ranges(
    RE: RunEngine,
    germ: GeRMDetector,
    stage: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
    two_shutters: tuple[Shutter, Shutter],
):
    x, y = stage
    fe_shutter, photon_shutter = two_shutters

    docs, _ = _run(
        RE,
        edxd_custom_pos_list_grid(
            x, [0.0, 1.0],
            y, [0.0, 10.0], [1.0, 12.0], [2, 3],
            0.02,
            reset_positions=False,
            use_shutter=False,
            germ=germ,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
        ),
    )  # fmt: skip

    inner = [[e["data"]["y"] for e in events] for events in _events_by_run(docs)]
    assert inner == [pytest.approx([0, 1]), pytest.approx([10, 11, 12])]
    assert await x.user_readback.get_value() == pytest.approx(1.0)
    assert await y.user_readback.get_value() == pytest.approx(12.0)


def test_edxd_custom_pos_list_grid_rejects_mismatched_ranges(
    RE: RunEngine,
    germ: GeRMDetector,
    stage: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
    two_shutters: tuple[Shutter, Shutter],
):
    x, y = stage
    fe_shutter, photon_shutter = two_shutters

    with pytest.raises(ValueError, match="same length as the number of outer"):
        RE(
            edxd_custom_pos_list_grid(
                x, [0.0, 1.0],
                y, [0.0, 1.0, 2.0], [1.0], [3],
                0.02,
                germ=germ,
                fe_shutter=fe_shutter,
                photon_shutter=photon_shutter,
            )
        )  # fmt: skip
    assert not get_mock_put(x.user_setpoint).called


# --- edxd_2theta_tilt -----------------------------------------------------------------


@pytest.fixture
def tables() -> tuple[EDXDTable, CollimatorTable]:
    with init_devices(mock=True):
        edxd_table = EDXDTable("EDXD:")
        collimator_table = CollimatorTable("COLL:")
    _fast(edxd_table, collimator_table)
    return edxd_table, collimator_table


@pytest.mark.parametrize(
    ("theta", "expected_y", "expected_pitch", "expected_y_coarse"),
    [
        # reference values for a detector 600 mm downstream (edxd_table.z)
        (5.0, -52.49319811555441, 0.9264658388913709, -25.721634367274543),
        (0.0, 0.0, -4.073534161108629, 11.715195973507164),
    ],
)
async def test_edxd_2theta_tilt(
    RE: RunEngine,
    tables: tuple[EDXDTable, CollimatorTable],
    theta: float,
    expected_y: float,
    expected_pitch: float,
    expected_y_coarse: float,
):
    edxd_table, collimator_table = tables
    set_mock_value(edxd_table.z.user_readback, 600.0)

    RE(
        edxd_2theta_tilt(
            theta, edxd_table=edxd_table, collimator_table=collimator_table
        )
    )

    assert await edxd_table.x.user_readback.get_value() == pytest.approx(0.0)
    assert await edxd_table.rx.user_readback.get_value() == pytest.approx(theta)
    assert await edxd_table.y.user_readback.get_value() == pytest.approx(expected_y)
    assert await collimator_table.slits_pitch.user_readback.get_value() == (
        pytest.approx(expected_pitch)
    )
    assert await collimator_table.y_coarse.user_readback.get_value() == (
        pytest.approx(expected_y_coarse)
    )
    assert await collimator_table.x.user_readback.get_value() == pytest.approx(-54)
    # the detector distance is only read
    assert not get_mock_put(edxd_table.z.user_setpoint).called


async def test_edxd_2theta_tilt_move_out_only_retracts_the_table(
    RE: RunEngine, tables: tuple[EDXDTable, CollimatorTable]
):
    edxd_table, collimator_table = tables

    RE(
        edxd_2theta_tilt(
            10.0,
            move_out=True,
            edxd_table=edxd_table,
            collimator_table=collimator_table,
        )
    )

    assert await edxd_table.x.user_readback.get_value() == pytest.approx(-270.0)
    others = [edxd_table.y, edxd_table.rx, *_motors_in(collimator_table)]
    assert not any(get_mock_put(m.user_setpoint).called for m in others)


# --- edxd_calib_scan ------------------------------------------------------------------


@pytest.fixture
async def sample_tower() -> SampleTower:
    with init_devices(mock=True):
        sample_tower = SampleTower("TWR:")
    _fast(sample_tower)
    z1 = sample_tower.z1
    set_mock_value(z1.max_velocity, 100.0)
    # A distinct starting velocity, so a missed restore shows. Put rather than
    # mocked, since the reset restores the last setpoint written.
    await z1.velocity.set(7.0)
    return sample_tower


def _put_values(motor_signal) -> list[float]:
    return [c.args[0] for c in get_mock_put(motor_signal).call_args_list]


async def test_edxd_calib_scan_sweeps_during_one_count(
    RE: RunEngine,
    germ: GeRMDetector,
    sample_tower: SampleTower,
    two_shutters: tuple[Shutter, Shutter],
    tiled: CatalogOfBlueskyRuns,
):
    z1 = sample_tower.z1
    fe_shutter, photon_shutter = two_shutters
    set_mock_value(z1.user_setpoint, 0.5)
    set_mock_value(z1.user_readback, 0.5)

    docs, _ = _run(
        RE,
        edxd_calib_scan(
            0.0,
            1.0,
            iterations=1,
            count_time=0.5,
            description="CeO2",
            sample_tower=sample_tower,
            germ=germ,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
        ),
    )

    (start,) = docs["start"]
    assert start["plan_name"] == "edxd_calib_scan"
    assert start["description"] == "CeO2"
    # A single count, read once the sweep is done.
    (event,) = docs["event"]
    assert event["data"]["germ-stats1-total"] > 0
    assert np.asarray(tiled[start["uid"]]["primary"]["germ"].read()).shape[0] == 1
    assert await germ.driver.acquire_time.get_value() == 0.5

    # Already in range, so it sweeps from where it is: toward stop, then start.
    setpoints = _put_values(z1.user_setpoint)
    assert setpoints[:2] == pytest.approx([1.0, 0.0])
    # two spans per iteration over the count time
    assert 2 * 1.0 * 1 / 0.5 in _put_values(z1.velocity)
    assert get_mock_put(z1.motor_stop).called

    # z1 speed and position restored, photon shutter closed
    assert await z1.velocity.get_value() == 7.0
    assert await z1.user_readback.get_value() == pytest.approx(0.5)
    assert await photon_shutter.status.get_value() == ShutterStatus.CLOSED


async def test_edxd_calib_scan_moves_in_range_first_at_full_speed(
    RE: RunEngine,
    germ: GeRMDetector,
    sample_tower: SampleTower,
    two_shutters: tuple[Shutter, Shutter],
):
    z1 = sample_tower.z1
    fe_shutter, photon_shutter = two_shutters
    set_mock_value(z1.user_setpoint, 3.0)
    set_mock_value(z1.user_readback, 3.0)

    _run(
        RE,
        edxd_calib_scan(
            0.0,
            1.0,
            iterations=1,
            count_time=0.5,
            sample_tower=sample_tower,
            germ=germ,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
        ),
    )

    # out of range: to the nearer end (stop) at max speed, then sweep at 4/s
    # (the first put is the fixture's starting velocity)
    assert _put_values(z1.velocity)[1:3] == [100.0, 4.0]
    assert _put_values(z1.user_setpoint)[0] == pytest.approx(1.0)
    assert await z1.velocity.get_value() == 7.0
    assert await z1.user_readback.get_value() == pytest.approx(3.0)


def test_edxd_calib_scan_rejects_a_sweep_faster_than_the_motor(
    RE: RunEngine,
    germ: GeRMDetector,
    sample_tower: SampleTower,
    two_shutters: tuple[Shutter, Shutter],
):
    fe_shutter, photon_shutter = two_shutters

    # 2 spans of 1 mm, twice, in 0.01 s needs 400 mm/s; z1 tops out at 100
    with pytest.raises(ValueError, match="exceeds max velocity"):
        RE(
            edxd_calib_scan(
                0.0,
                1.0,
                iterations=2,
                count_time=0.01,
                sample_tower=sample_tower,
                germ=germ,
                fe_shutter=fe_shutter,
                photon_shutter=photon_shutter,
            )
        )
    assert not get_mock_put(sample_tower.z1.user_setpoint).called
    assert not get_mock_put(germ.driver.acquire).called
