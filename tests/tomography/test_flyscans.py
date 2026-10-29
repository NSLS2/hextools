from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from bluesky import Msg, RunEngine
from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from bluesky.protocols import Collectable
from bluesky.utils import (
    FailedStatus,
    PlanHalt,
    RequestAbort,
    RunEngineInterrupted,
)
from bluesky_tiled_plugins import TiledWriter
from ophyd_async.core import (
    Device,
    DeviceVector,
    StaticPathProvider,
    UUIDFilenameProvider,
    callback_on_mock_put,
    get_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.adcore import NDPluginFileIO, NDProcessFilterType, NDProcessIO
from ophyd_async.epics.adkinetix import KinetixDetector, KinetixTriggerMode
from ophyd_async.epics.core import epics_signal_r, epics_signal_rw
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor
from ophyd_async.fastcs.panda import (
    DatasetTable,
    HDFPanda,
    PandaHdf5DatasetType,
    PandaPcompDirection,
)
from tiled.client.container import Container

from hextools.detectors import FRAME_PERIOD_MARGIN
from hextools.motors import RotationMotor, SampleTower
from hextools.photon_delivery_system import Shutter, ShutterStatus
from hextools.specs import TOMO_DARK_FLAT_V1, TOMO_FLYSCAN_V1
from hextools.tomography import flyscans
from hextools.tomography.flyscans import (
    MAX_MISSING_FRAMES,
    tomo_1d_step_scan,
    tomo_2d_step_scan,
    tomo_dark_flat,
    tomo_flyscan,
    tomo_flyscan_loop,
    tomo_nd_step_scan,
)

Docs = dict[str, list[dict[str, Any]]]


def _doc_cache() -> tuple[Docs, Callable[[str, dict[str, Any]], None]]:
    docs: Docs = {}

    def cache(name: str, doc: dict[str, Any]):
        docs.setdefault(name, []).append(doc)

    return docs, cache


def _no_sleeps(plan: Generator[Msg, Any, Any]):
    """Drop the shutter settling and preview sleeps, which are only for hardware."""
    return (
        yield from bpp.msg_mutator(
            plan, lambda msg: msg._replace(args=(0,)) if msg.command == "sleep" else msg
        )
    )


def _mock_motor(motor: AsyncEpicsMotor) -> AsyncEpicsMotor:
    set_mock_value(motor.velocity, 10000)
    set_mock_value(motor.max_velocity, 10000)
    set_mock_value(motor.acceleration_time, 0.0001)
    callback_on_mock_put(
        motor.user_setpoint, lambda v: set_mock_value(motor.user_readback, v)
    )
    return motor


@pytest.fixture
def static_path_provider(tmp_path: Path) -> StaticPathProvider:
    # tiled_client only serves files under tmp_path / "data"
    return StaticPathProvider(UUIDFilenameProvider(), tmp_path / "data")


@pytest.fixture
def sample_tower() -> SampleTower:
    with init_devices(mock=True):
        tower = SampleTower("TWR:")
    _mock_motor(tower.x1)
    return tower


@pytest.fixture
def step_motors() -> tuple[AsyncEpicsMotor, AsyncEpicsMotor]:
    with init_devices(mock=True):
        outer = AsyncEpicsMotor("OUTER:")
        inner = AsyncEpicsMotor("INNER:")
    return _mock_motor(outer), _mock_motor(inner)


@pytest.fixture
def open_fe_shutter(two_shutters: tuple[Shutter, Shutter]) -> tuple[Shutter, Shutter]:
    fe_shutter, photon_shutter = two_shutters
    set_mock_value(fe_shutter.status, ShutterStatus.OPEN)
    return fe_shutter, photon_shutter


# ==================================================================================
# tomo_dark_flat
# ==================================================================================


def _shutter_and_x_at_each_trigger(
    det: KinetixDetector, photon_shutter: Shutter, sample_tower: SampleTower
) -> list[tuple[ShutterStatus, float]]:
    """Record the photon shutter state and sample x each time the camera acquires."""
    seen: list[tuple[ShutterStatus, float]] = []
    file_plugin = det.get_plugin("hdf", NDPluginFileIO)

    async def _acquire(value):
        if not value:
            return
        seen.append(
            (
                await photon_shutter.status.get_value(),
                await sample_tower.x1.user_setpoint.get_value(),
            )
        )
        if await det.driver.trigger_mode.get_value() != KinetixTriggerMode.INTERNAL:
            return
        frames = max(1, await det.driver.num_images.get_value())
        captured = await file_plugin.num_captured.get_value()
        set_mock_value(file_plugin.num_captured, captured + frames)

    callback_on_mock_put(det.driver.acquire, _acquire)
    return seen


async def test_tomo_dark_flat(
    RE: RunEngine,
    kinetix_det_factory: Callable[[int], KinetixDetector],
    open_fe_shutter: tuple[Shutter, Shutter],
    sample_tower: SampleTower,
):
    fe_shutter, photon_shutter = open_fe_shutter
    ktx = kinetix_det_factory(1)
    exposure_time, num_dark, num_flat, x_offset = 0.05, 4, 7, 2.5
    set_mock_value(sample_tower.x1.user_readback, 1.0)
    await sample_tower.x1.user_setpoint.set(1.0)
    await ktx.driver.acquire_time.set(0.3)
    seen = _shutter_and_x_at_each_trigger(ktx, photon_shutter, sample_tower)
    docs, cache = _doc_cache()

    RE(
        _no_sleeps(
            tomo_dark_flat(
                [ktx],
                exposure_time,
                num_dark_images=num_dark,
                num_flat_images=num_flat,
                x_offset=x_offset,
                sample_tower=sample_tower,
                fe_shutter=fe_shutter,
                photon_shutter=photon_shutter,
            )
        ),
        cache,  # type: ignore
    )

    assert len(docs["start"]) == 1
    start = docs["start"][0]
    assert start["plan_name"] == "tomo_dark_flat"
    assert start["detectors"] == ["kinetix1"]
    assert start["num_dark_images"] == num_dark
    assert start["num_flat_images"] == num_flat
    assert start["x_offset"] == x_offset
    assert start["tiled_specs"] == [
        {"name": TOMO_DARK_FLAT_V1.name, "version": TOMO_DARK_FLAT_V1.version}
    ]
    assert [d["name"] for d in docs["descriptor"]] == ["dark", "flat"]
    assert len(docs["event"]) == 2
    assert docs["stop"][0]["exit_status"] == "success"

    # Darks with the photon shutter closed at the sample position; flats with it
    # open and the sample moved out. The trailing entry is the preview restart.
    assert seen[:2] == [
        (ShutterStatus.CLOSED, pytest.approx(1.0)),
        (ShutterStatus.OPEN, pytest.approx(1.0 + x_offset)),
    ]
    assert get_mock_put(ktx.driver.num_images).call_args_list[0].args[0] == num_dark

    # Positions restored, shutter closed, camera back in preview.
    assert await sample_tower.x1.user_setpoint.get_value() == pytest.approx(1.0)
    assert await photon_shutter.status.get_value() == ShutterStatus.CLOSED
    assert await ktx.driver.trigger_mode.get_value() == KinetixTriggerMode.INTERNAL


async def test_tomo_dark_flat_restores_acquire_time(
    RE: RunEngine,
    kinetix_det_factory: Callable[[int], KinetixDetector],
    open_fe_shutter: tuple[Shutter, Shutter],
    sample_tower: SampleTower,
):
    fe_shutter, photon_shutter = open_fe_shutter
    ktx = kinetix_det_factory(1)
    await ktx.driver.acquire_time.set(0.3)

    RE(
        _no_sleeps(
            tomo_dark_flat(
                [ktx],
                0.05,
                num_dark_images=2,
                num_flat_images=2,
                sample_tower=sample_tower,
                fe_shutter=fe_shutter,
                photon_shutter=photon_shutter,
            )
        )
    )

    assert await ktx.driver.acquire_time.get_value() == pytest.approx(0.3)


async def test_tomo_dark_flat_without_shutter_never_opens_it(
    RE: RunEngine,
    kinetix_det_factory: Callable[[int], KinetixDetector],
    two_shutters: tuple[Shutter, Shutter],
    sample_tower: SampleTower,
):
    # The front-end shutter is only checked when the shutter is in use.
    fe_shutter, photon_shutter = two_shutters
    ktx = kinetix_det_factory(1)
    seen = _shutter_and_x_at_each_trigger(ktx, photon_shutter, sample_tower)
    docs, cache = _doc_cache()

    RE(
        _no_sleeps(
            tomo_dark_flat(
                [ktx],
                0.05,
                num_dark_images=2,
                num_flat_images=3,
                x_offset=1.0,
                use_shutter=False,
                sample_tower=sample_tower,
                fe_shutter=fe_shutter,
                photon_shutter=photon_shutter,
            )
        ),
        cache,  # type: ignore
    )

    assert docs["stop"][0]["exit_status"] == "success"
    assert seen
    assert all(status == ShutterStatus.CLOSED for status, _ in seen)


def test_tomo_dark_flat_fails_if_fe_shutter_closed(
    RE: RunEngine,
    kinetix_det_factory: Callable[[int], KinetixDetector],
    two_shutters: tuple[Shutter, Shutter],
    sample_tower: SampleTower,
):
    fe_shutter, photon_shutter = two_shutters
    ktx = kinetix_det_factory(1)
    docs, cache = _doc_cache()

    with pytest.raises(ValueError, match=f"Shutter {fe_shutter.name}"):
        RE(
            tomo_dark_flat(
                [ktx],
                0.05,
                sample_tower=sample_tower,
                fe_shutter=fe_shutter,
                photon_shutter=photon_shutter,
            ),
            cache,  # type: ignore
        )
    assert "start" not in docs
    assert not get_mock_put(ktx.driver.acquire).called


# ==================================================================================
# tomo_flyscan
# ==================================================================================


class _Bits(Device):
    def __init__(self, prefix: str, name: str = ""):
        self.a = epics_signal_rw(int, f"{prefix}A")
        self.b = epics_signal_rw(int, f"{prefix}B")
        super().__init__(name=name)


class _Calc(Device):
    def __init__(self, prefix: str, name: str = ""):
        self.out = epics_signal_r(int, f"{prefix}OUT")
        self.out_dataset = epics_signal_rw(str, f"{prefix}OUT:DATASET")
        self.out_units = epics_signal_rw(str, f"{prefix}OUT:UNITS")
        self.out_scale = epics_signal_rw(float, f"{prefix}OUT:SCALE")
        self.out_offset = epics_signal_rw(float, f"{prefix}OUT:OFFSET")
        super().__init__(name=name)


class _FlyPanda(HDFPanda):
    """HDFPanda with the blocks the real one discovers from the IOC."""

    def __init__(self, prefix: str, path_provider, name: str = ""):
        self.bits = _Bits(f"{prefix}BITS:")
        self.calc = DeviceVector({1: _Calc(f"{prefix}CALC1:")})
        super().__init__(prefix, path_provider, name=name)


class FlyRig:
    """Mock PandA, rotation stage and cameras wired together for a flyscan.

    Enabling BITS.A starts PCOMP; the rotation stage's fly move then writes
    ``frames_per_move`` frames on every camera and the PandA, and ends PCOMP.
    With ``write_files``, each move also writes the HDF5 files the real devices
    would, recording what it wrote in ``written``.
    """

    # neither the max velocity nor any fly speed, so a missed restore shows
    INITIAL_VELOCITY = 7.0

    def __init__(
        self,
        static_path_provider: StaticPathProvider,
        cameras: list[KinetixDetector],
        encoder_counts: int = 1000,
    ):
        # init_devices only connects devices bound to local names
        with init_devices(mock=True):
            panda = _FlyPanda("PANDA:", static_path_provider, name="panda1")
            rot_motor = RotationMotor("ROT:", name="rot")
        self.panda, self.rot_motor = panda, rot_motor
        self.cameras = cameras
        self.frames_per_move = 0
        self.photon_shutter: Shutter | None = None
        self.shutter_during_move: ShutterStatus | None = None
        self.write_files = False
        self.written: list[dict[str, np.ndarray]] = []

        set_mock_value(panda.data.directory_exists, True)
        set_mock_value(
            panda.data.datasets,
            DatasetTable(
                name=["rotation_angle"], dtype=[PandaHdf5DatasetType.FLOAT_64]
            ),
        )
        set_mock_value(panda.calc[1].out, encoder_counts)
        callback_on_mock_put(
            panda.pcap.arm, lambda v: set_mock_value(panda.pcap.active, v)
        )
        callback_on_mock_put(
            panda.data.capture, lambda _: set_mock_value(panda.data.num_captured, 0)
        )
        callback_on_mock_put(
            panda.bits.a, lambda v: set_mock_value(panda.pcomp[1].active, bool(v))
        )

        rot = _mock_motor(self.rot_motor)
        set_mock_value(rot.velocity, self.INITIAL_VELOCITY)
        set_mock_value(rot.encoder_resolution, 0.01)
        callback_on_mock_put(rot.user_setpoint, self._move)

    async def _move(self, value: float):
        start = await self.rot_motor.user_readback.get_value()
        set_mock_value(self.rot_motor.user_readback, value)
        if not await self.panda.pcomp[1].active.get_value():
            return  # the run-up move, before PCOMP is enabled
        if self.photon_shutter is not None:
            self.shutter_during_move = await self.photon_shutter.status.get_value()
        if self.write_files:
            await self._write_files(start, value)
        for cam in self.cameras:
            hdf = cam.get_plugin("hdf", NDPluginFileIO)
            captured = await hdf.num_captured.get_value()
            set_mock_value(hdf.num_captured, captured + self.frames_per_move)
        captured = await self.panda.data.num_captured.get_value()
        set_mock_value(self.panda.data.num_captured, captured + self.frames_per_move)
        set_mock_value(self.panda.pcomp[1].active, False)

    async def _write_files(self, start: float, stop: float):
        n = self.frames_per_move
        written: dict[str, np.ndarray] = {}
        for cam in self.cameras:
            hdf = cam.get_plugin("hdf", NDPluginFileIO)
            y = await cam.driver.array_size_y.get_value()
            x = await cam.driver.array_size_x.get_value()
            # distinct frames per move, so a mixed-up run shows in the data
            offset = len(self.written) * n * y * x
            images = (np.arange(n * y * x) + offset).reshape(n, y, x).astype(np.uint16)
            path = Path(await hdf.file_path.get_value())
            with h5py.File(path / f"{await hdf.file_name.get_value()}.h5", "w") as f:
                f.create_dataset("/entry/data/data", data=images)
            written[cam.name] = images

        angles = np.linspace(start, stop, n)
        data = self.panda.data
        path = Path(await data.hdf_directory.get_value())
        with h5py.File(path / await data.hdf_file_name.get_value(), "w") as f:
            f.create_dataset("/rotation_angle", data=angles)
        written["rotation_angle"] = angles
        self.written.append(written)

    def flyscan(self, detectors: list[Any], shutters, **kwargs):
        fe_shutter, photon_shutter = shutters
        self.photon_shutter = photon_shutter
        self.frames_per_move = kwargs["num_projections"]
        return _no_sleeps(
            tomo_flyscan(
                detectors,
                panda=self.panda,  # type: ignore
                rot_motor=self.rot_motor,
                fe_shutter=fe_shutter,
                photon_shutter=photon_shutter,
                **kwargs,
            )
        )


@pytest.fixture
def fly_rig(static_path_provider: StaticPathProvider) -> Callable[..., FlyRig]:
    def _factory(cameras: list[KinetixDetector]) -> FlyRig:
        return FlyRig(static_path_provider, cameras)

    return _factory


@pytest.mark.parametrize("num_cameras", [1, 2])
async def test_tomo_flyscan(
    RE: RunEngine,
    kinetix_det_factory: Callable[[int], KinetixDetector],
    fly_rig: Callable[..., FlyRig],
    open_fe_shutter: tuple[Shutter, Shutter],
    num_cameras: int,
):
    cameras = [kinetix_det_factory(i + 1) for i in range(num_cameras)]
    rig = fly_rig(cameras)
    num_projections, exposure_time = 11, 0.01
    docs, cache = _doc_cache()

    RE(
        rig.flyscan(
            cameras,
            open_fe_shutter,
            num_projections=num_projections,
            exposure_time=exposure_time,
            start=0,
            stop=180,
            sample_name="rock",
        ),
        cache,  # type: ignore
    )

    assert len(docs["start"]) == 1
    start = docs["start"][0]
    assert start["plan_name"] == "tomo_flyscan"
    assert start["sample_name"] == "rock"
    assert start["detectors"] == [cam.name for cam in cameras]
    assert start["num_points"] == num_projections
    assert (start["start_position"], start["stop_position"]) == (0, 180)
    assert start["tiled_specs"] == [
        {"name": TOMO_FLYSCAN_V1.name, "version": TOMO_FLYSCAN_V1.version}
    ]
    assert docs["stop"][0]["exit_status"] == "success"
    assert [d["name"] for d in docs["descriptor"]] == ["primary"]

    # Every camera and the PandA's angle land in primary, covering every projection.
    data_keys = {d["data_key"] for d in docs["stream_resource"]}
    assert {cam.name for cam in cameras} <= data_keys
    assert "rotation_angle" in data_keys
    for resource in docs["stream_resource"]:
        datums = [
            d for d in docs["stream_datum"] if d["stream_resource"] == resource["uid"]
        ]
        assert sum(d["indices"]["stop"] - d["indices"]["start"] for d in datums) == (
            num_projections
        )

    # Cameras armed for external edges, with plugin queues big enough for the run.
    for cam in cameras:
        modes = [
            c.args[0] for c in get_mock_put(cam.driver.trigger_mode).call_args_list
        ]
        assert any(mode != KinetixTriggerMode.INTERNAL for mode in modes)
        hdf = cam.get_plugin("hdf", NDPluginFileIO)
        assert await hdf.queue_size.get_value() == num_projections
        # cleanup puts the camera back into preview
        assert await cam.driver.trigger_mode.get_value() == KinetixTriggerMode.INTERNAL

    # Time based: PCOMP fires once and PULSE paces the projections.
    pcomp, pulse = rig.panda.pcomp[1], rig.panda.pulse[1]
    assert await pcomp.dir.get_value() == PandaPcompDirection.POSITIVE
    assert await pcomp.pulses.get_value() == 1
    assert await pulse.pulses.get_value() == num_projections
    assert await pulse.width.get_value() == pytest.approx(exposure_time)
    move_time = num_projections * (exposure_time + FRAME_PERIOD_MARGIN)
    assert await pulse.step.get_value() == pytest.approx(
        move_time / (num_projections - 1)
    )

    _, photon_shutter = open_fe_shutter
    assert rig.shutter_during_move == ShutterStatus.OPEN
    assert await photon_shutter.status.get_value() == ShutterStatus.CLOSED
    assert get_mock_put(rig.rot_motor.motor_stop).call_args.args[0] == 1
    assert await rig.rot_motor.velocity.get_value() == FlyRig.INITIAL_VELOCITY


# Nghia's scan 21 (2026-10-27) averaged with Sum: the IOC had been left on it.
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, NDProcessFilterType.AVERAGE),
        ({"filter_type": "Sum"}, NDProcessFilterType.SUM),
        (
            {"filter_type": NDProcessFilterType.RECURSIVE_AVG},
            NDProcessFilterType.RECURSIVE_AVG,
        ),
    ],
)
async def test_tomo_flyscan_sets_the_proc_filter_type(
    RE: RunEngine,
    kinetix_det_factory: Callable[..., KinetixDetector],
    fly_rig: Callable[..., FlyRig],
    open_fe_shutter: tuple[Shutter, Shutter],
    kwargs: dict[str, Any],
    expected: NDProcessFilterType,
):
    cam = kinetix_det_factory(1, with_proc=True)
    proc = cam.get_plugin("proc", NDProcessIO)
    set_mock_value(proc.filter_type, NDProcessFilterType.DIFFERENCE)
    rig = fly_rig([cam])
    docs, cache = _doc_cache()

    RE(
        rig.flyscan(
            [cam], open_fe_shutter, num_projections=5, exposure_time=0.01, **kwargs
        ),
        cache,  # type: ignore
    )

    assert docs["stop"][0]["exit_status"] == "success"
    assert await proc.filter_type.get_value() == expected
    assert docs["start"][0]["filter_type"] == expected.value


def test_tomo_flyscan_rejects_an_unknown_filter_type(
    RE: RunEngine,
    kinetix_det_factory: Callable[..., KinetixDetector],
    fly_rig: Callable[..., FlyRig],
    open_fe_shutter: tuple[Shutter, Shutter],
):
    cam = kinetix_det_factory(1, with_proc=True)
    rig = fly_rig([cam])

    with pytest.raises(ValueError, match="Median"):
        RE(
            rig.flyscan(
                [cam],
                open_fe_shutter,
                num_projections=5,
                exposure_time=0.01,
                filter_type="Median",
            )
        )


async def test_tomo_flyscan_arms_only_after_run_up(
    RE: RunEngine,
    kinetix_det_factory: Callable[[int], KinetixDetector],
    fly_rig: Callable[..., FlyRig],
    open_fe_shutter: tuple[Shutter, Shutter],
):
    ktx = kinetix_det_factory(1)
    rig = fly_rig([ktx])
    messages: list[Msg] = []
    RE.msg_hook = messages.append  # type: ignore

    RE(rig.flyscan([ktx], open_fe_shutter, num_projections=5, exposure_time=0.01))

    prepared = [msg.obj for msg in messages if msg.command == "prepare"]
    assert prepared[0] is rig.rot_motor
    assert set(prepared[1:]) >= {ktx, rig.panda}
    wait_for_run_up = next(
        i
        for i, msg in enumerate(messages)
        if msg.command == "wait" and msg.kwargs.get("group") == "prepare"
    )
    first_arm = next(
        i
        for i, msg in enumerate(messages)
        if msg.command == "prepare" and msg.kwargs.get("group") == "arm"
    )
    assert wait_for_run_up < first_arm


async def test_tomo_flyscan_reverse_direction(
    RE: RunEngine,
    kinetix_det_factory: Callable[[int], KinetixDetector],
    fly_rig: Callable[..., FlyRig],
    open_fe_shutter: tuple[Shutter, Shutter],
):
    ktx = kinetix_det_factory(1)
    rig = fly_rig([ktx])

    RE(
        rig.flyscan(
            [ktx],
            open_fe_shutter,
            num_projections=5,
            exposure_time=0.01,
            start=180,
            stop=0,
        )
    )

    assert await rig.panda.pcomp[1].dir.get_value() == PandaPcompDirection.NEGATIVE


async def test_tomo_flyscan_position_based(
    RE: RunEngine,
    kinetix_det_factory: Callable[[int], KinetixDetector],
    fly_rig: Callable[..., FlyRig],
    open_fe_shutter: tuple[Shutter, Shutter],
):
    ktx = kinetix_det_factory(1)
    rig = fly_rig([ktx])
    num_projections = 11

    RE(
        rig.flyscan(
            [ktx],
            open_fe_shutter,
            num_projections=num_projections,
            exposure_time=0.01,
            start=0,
            stop=10,
            time_based=False,
        )
    )

    # 10 deg at 0.01 deg/count over 10 steps: PCOMP paces 100 counts apart.
    pcomp = rig.panda.pcomp[1]
    assert await pcomp.pulses.get_value() == num_projections
    assert await pcomp.step.get_value() == 100
    assert await rig.panda.pulse[1].pulses.get_value() == 1


async def test_tomo_flyscan_acquire_period_sets_deadtime(
    RE: RunEngine,
    kinetix_det_factory: Callable[[int], KinetixDetector],
    fly_rig: Callable[..., FlyRig],
    open_fe_shutter: tuple[Shutter, Shutter],
):
    ktx = kinetix_det_factory(1)
    rig = fly_rig([ktx])
    num_projections, exposure_time, acquire_period = 5, 0.01, 0.05
    docs, cache = _doc_cache()

    RE(
        rig.flyscan(
            [ktx],
            open_fe_shutter,
            num_projections=num_projections,
            exposure_time=exposure_time,
            acquire_period=acquire_period,
        ),
        cache,  # type: ignore
    )

    assert docs["start"][0]["acquire_period"] == acquire_period
    move_time = num_projections * acquire_period
    assert await rig.panda.pulse[1].step.get_value() == pytest.approx(
        move_time / (num_projections - 1)
    )


async def test_tomo_flyscan_without_shutter(
    RE: RunEngine,
    kinetix_det_factory: Callable[[int], KinetixDetector],
    fly_rig: Callable[..., FlyRig],
    two_shutters: tuple[Shutter, Shutter],
):
    # A closed front end is fine when the shutter is not used.
    ktx = kinetix_det_factory(1)
    rig = fly_rig([ktx])
    docs, cache = _doc_cache()

    RE(
        rig.flyscan(
            [ktx],
            two_shutters,
            num_projections=5,
            exposure_time=0.01,
            use_shutter=False,
        ),
        cache,  # type: ignore
    )

    assert docs["stop"][0]["exit_status"] == "success"
    assert rig.shutter_during_move == ShutterStatus.CLOSED


def test_tomo_flyscan_fails_if_fe_shutter_closed(
    RE: RunEngine,
    kinetix_det_factory: Callable[[int], KinetixDetector],
    fly_rig: Callable[..., FlyRig],
    two_shutters: tuple[Shutter, Shutter],
):
    ktx = kinetix_det_factory(1)
    rig = fly_rig([ktx])
    docs, cache = _doc_cache()

    with pytest.raises(ValueError, match="is closed but the desired state is open"):
        RE(
            rig.flyscan([ktx], two_shutters, num_projections=5, exposure_time=0.01),
            cache,  # type: ignore
        )
    assert "start" not in docs


# --- Dropped frames ---------------------------------------------------------------


def _failing_fly_stub(rig: FlyRig, frames_written: dict[str, int]):
    """A fly_stub whose devices wrote some frames before a status failed."""

    def _fly_stub(flyers, *, collect_flush_period=None, stream_name="", watch=()):
        dets = [f for f in flyers if isinstance(f, Collectable)]
        yield from bps.declare_stream(*dets, name=stream_name)  # type: ignore
        for cam in rig.cameras:
            hdf = cam.get_plugin("hdf", NDPluginFileIO)
            set_mock_value(hdf.num_captured, frames_written[cam.name])
        set_mock_value(rig.panda.data.num_captured, max(frames_written.values()))
        raise FailedStatus("a device timed out")

    return _fly_stub


@pytest.mark.parametrize("missed", [1, MAX_MISSING_FRAMES])
async def test_tomo_flyscan_tolerates_a_few_missed_frames(
    RE: RunEngine,
    kinetix_det_factory: Callable[[int], KinetixDetector],
    fly_rig: Callable[..., FlyRig],
    open_fe_shutter: tuple[Shutter, Shutter],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    missed: int,
):
    num_projections = 50
    cameras = [kinetix_det_factory(1), kinetix_det_factory(2)]
    rig = fly_rig(cameras)
    written = {"kinetix1": num_projections - missed, "kinetix2": num_projections}
    monkeypatch.setattr(flyscans.bps, "fly_stub", _failing_fly_stub(rig, written))
    docs, cache = _doc_cache()

    RE(
        rig.flyscan(
            cameras,
            open_fe_shutter,
            num_projections=num_projections,
            exposure_time=0.01,
        ),
        cache,  # type: ignore
    )

    assert docs["stop"][0]["exit_status"] == "success"
    # Events stay aligned, so only the frames every camera wrote are recorded.
    for cam in cameras:
        datums = [
            d
            for d in docs["stream_datum"]
            for r in docs["stream_resource"]
            if r["data_key"] == cam.name and d["stream_resource"] == r["uid"]
        ]
        assert sum(d["indices"]["stop"] - d["indices"]["start"] for d in datums) == (
            num_projections - missed
        )
    assert f"Detector kinetix1 missed {missed} frame(s)" in caplog.text
    assert "Detector kinetix2 missed 0 frame(s)" in caplog.text


@pytest.mark.parametrize(
    ("missed", "error", "match"),
    [
        # Nothing dropped: the failure is elsewhere, so it propagates as is.
        (0, FailedStatus, "timed out"),
        (MAX_MISSING_FRAMES + 1, RuntimeError, "missed up to 11 frame"),
        # Losing half the scan points to a bigger problem than dropped frames.
        (25, FailedStatus, "timed out"),
        (50, FailedStatus, "timed out"),
    ],
)
async def test_tomo_flyscan_fails_on_too_many_missed_frames(
    RE: RunEngine,
    kinetix_det_factory: Callable[[int], KinetixDetector],
    fly_rig: Callable[..., FlyRig],
    open_fe_shutter: tuple[Shutter, Shutter],
    monkeypatch: pytest.MonkeyPatch,
    missed: int,
    error: type[Exception],
    match: str,
):
    num_projections = 50
    ktx = kinetix_det_factory(1)
    rig = fly_rig([ktx])
    written = {"kinetix1": num_projections - missed}
    monkeypatch.setattr(flyscans.bps, "fly_stub", _failing_fly_stub(rig, written))
    docs, cache = _doc_cache()

    with pytest.raises(error, match=match):
        RE(
            rig.flyscan(
                [ktx],
                open_fe_shutter,
                num_projections=num_projections,
                exposure_time=0.01,
            ),
            cache,  # type: ignore
        )

    assert docs["stop"][0]["exit_status"] == "fail"
    # Cleanup still runs.
    _, photon_shutter = open_fe_shutter
    assert await photon_shutter.status.get_value() == ShutterStatus.CLOSED
    assert get_mock_put(rig.rot_motor.motor_stop).called
    assert await rig.rot_motor.velocity.get_value() == FlyRig.INITIAL_VELOCITY
    assert await ktx.driver.trigger_mode.get_value() == KinetixTriggerMode.INTERNAL


# ==================================================================================
# Orchestration: tomo_flyscan_loop and the step scans
# ==================================================================================


class Recorder:
    """Stands in for tomo_flyscan and tomo_dark_flat, recording each call."""

    def __init__(self, motors: tuple[AsyncEpicsMotor, ...] = ()):
        self.motors = motors
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.failures: list[BaseException] = []
        self.pause_after_call: int | None = None

    def flyscan(self, **kwargs):
        positions = []
        for motor in self.motors:
            positions.append((yield from bps.rd(motor)))
        kwargs["positions"] = tuple(positions)
        self.calls.append(("flyscan", kwargs))
        if self.failures:
            raise self.failures.pop(0)
        if len(self.calls) - 1 == self.pause_after_call:
            # what a deferred pause from the queue server or Ctrl-C sends
            yield Msg("pause", None, defer=True)
        yield from bps.null()
        # tomo_flyscan returns its run's start uid
        return f"uid-{len(self.calls) - 1}"

    def dark_flat(self, *args, **kwargs):
        if args:
            kwargs["detectors"] = args[0]
        self.calls.append(("dark_flat", kwargs))
        yield from bps.null()

    @property
    def sequence(self) -> str:
        return "".join("F" if kind == "flyscan" else "D" for kind, _ in self.calls)

    @property
    def flyscans(self) -> list[dict[str, Any]]:
        return [kwargs for kind, kwargs in self.calls if kind == "flyscan"]

    @property
    def dark_flats(self) -> list[dict[str, Any]]:
        return [kwargs for kind, kwargs in self.calls if kind == "dark_flat"]


@pytest.fixture
def recorder(
    monkeypatch: pytest.MonkeyPatch,
    step_motors: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
) -> Recorder:
    rec = Recorder(step_motors)
    monkeypatch.setattr(flyscans, "tomo_flyscan", rec.flyscan)
    monkeypatch.setattr(flyscans, "tomo_dark_flat", rec.dark_flat)
    return rec


DEVICES = {
    "panda": "PANDA",
    "rot_motor": "ROT",
    "fe_shutter": "FE",
    "photon_shutter": "PH",
}


# --- tomo_flyscan_loop --------------------------------------------------------------


def test_tomo_flyscan_loop_forwards_flyscan_arguments(
    RE: RunEngine, recorder: Recorder
):
    RE(
        tomo_flyscan_loop(
            ["det"],  # type: ignore
            num_projections=100,
            exposure_time=0.2,
            acquire_period=0.3,
            images_to_average=2,
            filter_type=NDProcessFilterType.SUM,
            start=10,
            stop=190,
            num_iterations=3,
            use_shutter=False,
            sample_name="rock",
            time_based=False,
            **DEVICES,  # type: ignore
        )
    )

    assert recorder.sequence == "FFF"
    expected = {
        "detectors": ["det"],
        "num_projections": 100,
        "exposure_time": 0.2,
        "acquire_period": 0.3,
        "images_to_average": 2,
        "filter_type": NDProcessFilterType.SUM,
        "start": 10,
        "stop": 190,
        "use_shutter": False,
        "sample_name": "rock",
        "time_based": False,
        **DEVICES,
    }
    for call in recorder.flyscans:
        assert {k: call[k] for k in expected} == expected


@pytest.mark.parametrize(
    ("num_iterations", "dark_flat_offset", "dark_flat_frequency", "sequence"),
    [
        (3, 0.0, 0, "FFF"),
        # A zero offset means no darks/flats, whatever the frequency.
        (3, 0.0, 1, "FFF"),
        (3, 1.5, 0, "DFFFD"),
        (1, 1.5, 1, "DFD"),
        # The final dark/flat covers the last tomogram, so none is doubled up.
        (3, 1.5, 1, "DFDFDFD"),
        (5, -1.5, 2, "DFFDFFDFD"),
        (4, 1.5, 2, "DFFDFFD"),
        (3, 1.5, 5, "DFFFD"),
    ],
)
def test_tomo_flyscan_loop_dark_flat_schedule(
    RE: RunEngine,
    recorder: Recorder,
    num_iterations: int,
    dark_flat_offset: float,
    dark_flat_frequency: int,
    sequence: str,
):
    RE(
        tomo_flyscan_loop(
            ["det"],  # type: ignore
            num_projections=10,
            exposure_time=0.1,
            num_iterations=num_iterations,
            dark_flat_offset=dark_flat_offset,
            dark_flat_frequency=dark_flat_frequency,
            **DEVICES,  # type: ignore
        )
    )
    assert recorder.sequence == sequence


@pytest.mark.parametrize(
    ("dark_flat_exposure_time", "expected"), [(None, 0.1), (0.5, 0.5)]
)
def test_tomo_flyscan_loop_dark_flat_arguments(
    RE: RunEngine,
    recorder: Recorder,
    sample_tower: SampleTower,
    dark_flat_exposure_time: float | None,
    expected: float,
):
    RE(
        tomo_flyscan_loop(
            ["det"],  # type: ignore
            num_projections=10,
            exposure_time=0.1,
            dark_flat_offset=2.0,
            num_dark_images=7,
            num_flat_images=9,
            dark_flat_exposure_time=dark_flat_exposure_time,
            use_shutter=False,
            sample_tower=sample_tower,
            **DEVICES,  # type: ignore
        )
    )

    assert len(recorder.dark_flats) == 2
    for call in recorder.dark_flats:
        assert call == {
            "detectors": ["det"],
            "exposure_time": expected,
            "num_dark_images": 7,
            "num_flat_images": 9,
            "x_offset": 2.0,
            "sample_tower": sample_tower,
            "use_shutter": False,
            "fe_shutter": "FE",
            "photon_shutter": "PH",
        }


def test_tomo_flyscan_loop_wraps_flyscan_failures(RE: RunEngine, recorder: Recorder):
    recorder.failures = [FailedStatus("a device timed out")]

    with pytest.raises(RuntimeError, match="point 0 failed after 1 attempts") as info:
        RE(
            tomo_flyscan_loop(
                ["det"],  # type: ignore
                num_projections=10,
                exposure_time=0.1,
                num_iterations=3,
                **DEVICES,  # type: ignore
            )
        )

    assert isinstance(info.value.__cause__, FailedStatus)
    assert len(recorder.flyscans) == 1


@pytest.mark.parametrize(("num_iterations", "delay"), [(1, 2.0), (4, 2.0), (4, 0.0)])
def test_tomo_flyscan_loop_delay_between_tomograms(
    RE: RunEngine, recorder: Recorder, num_iterations: int, delay: float
):
    messages: list[Msg] = []
    RE.msg_hook = messages.append  # type: ignore

    RE(
        bpp.msg_mutator(
            tomo_flyscan_loop(
                ["det"],  # type: ignore
                num_projections=10,
                exposure_time=0.1,
                num_iterations=num_iterations,
                delay_between_tomograms=delay,
                **DEVICES,  # type: ignore
            ),
            # count the sleeps without waiting them out
            lambda msg: (
                msg._replace(args=(0,), kwargs={"requested": msg.args[0]})
                if msg.command == "sleep"
                else msg
            ),
        )
    )

    sleeps = [msg.kwargs["requested"] for msg in messages if msg.command == "sleep"]
    # Only between tomograms, never before the first.
    assert sleeps == ([delay] * (num_iterations - 1) if delay else [])
    assert len(recorder.flyscans) == num_iterations


# --- tomo_nd_step_scan --------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ("motor", 0, 1),
        ("motor", 0, 1, 3, "other"),
    ],
)
def test_tomo_nd_step_scan_rejects_malformed_axes(args: tuple):
    plan = tomo_nd_step_scan(["det"], *args, num_projections=10, exposure_time=0.1)  # type: ignore
    with pytest.raises(ValueError, match="motor, start, stop, num"):
        next(plan)


def test_tomo_nd_step_scan_without_axes_repeats_in_place(
    RE: RunEngine,
    recorder: Recorder,
    step_motors: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
):
    outer, inner = step_motors

    RE(
        tomo_nd_step_scan(
            ["det"],  # type: ignore
            num_projections=10,
            exposure_time=0.1,
            num_repeats=3,
            **DEVICES,  # type: ignore
        )
    )

    assert recorder.sequence == "FFF"
    assert [c["md"]["grid_index"] for c in recorder.flyscans] == [0, 1, 2]
    assert all(c["md"]["grid_positions"] == {} for c in recorder.flyscans)
    assert not get_mock_put(outer.user_setpoint).called
    assert not get_mock_put(inner.user_setpoint).called


def test_tomo_nd_step_scan_repeats_the_whole_grid(
    RE: RunEngine,
    recorder: Recorder,
    step_motors: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
):
    outer, _ = step_motors

    RE(
        tomo_nd_step_scan(
            ["det"],  # type: ignore
            outer, 0, 1, 2,
            num_projections=10,
            exposure_time=0.1,
            num_repeats=2,
            **DEVICES,  # type: ignore
        )
    )  # fmt: skip

    positions = [c["positions"][0] for c in recorder.flyscans]
    assert positions == pytest.approx([0, 1, 0, 1])


def test_tomo_nd_step_scan_rejects_snaking_outer_axis(
    RE: RunEngine,
    step_motors: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
):
    outer, inner = step_motors
    plan = tomo_nd_step_scan(
        ["det"],  # type: ignore
        outer, 0, 1, 2,
        inner, 0, 1, 2,
        num_projections=10,
        exposure_time=0.1,
        snake_axes=[outer],
    )  # fmt: skip
    with pytest.raises(ValueError, match="outermost axis cannot be snaked"):
        next(plan)


@pytest.mark.parametrize(
    ("snake_axes", "expected"),
    [
        (
            False,
            [(0, 10), (0, 15), (0, 20), (1, 10), (1, 15), (1, 20)],
        ),
        (
            True,
            [(0, 10), (0, 15), (0, 20), (1, 20), (1, 15), (1, 10)],
        ),
        (
            "inner",
            [(0, 10), (0, 15), (0, 20), (1, 20), (1, 15), (1, 10)],
        ),
    ],
)
async def test_tomo_nd_step_scan_grid(
    RE: RunEngine,
    recorder: Recorder,
    step_motors: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
    snake_axes: bool | str,
    expected: list[tuple[float, float]],
):
    outer, inner = step_motors
    await outer.user_setpoint.set(-3.0)
    await inner.user_setpoint.set(7.0)
    snake: Any = [inner] if snake_axes == "inner" else snake_axes

    RE(
        tomo_nd_step_scan(
            ["det"],  # type: ignore
            outer, 0, 1, 2,
            inner, 10, 20, 3,
            num_projections=10,
            exposure_time=0.1,
            snake_axes=snake,
            **DEVICES,  # type: ignore
        )
    )  # fmt: skip

    positions = [
        (call["positions"][0], call["positions"][1]) for call in recorder.flyscans
    ]
    assert positions == [pytest.approx(p) for p in expected]
    # Each flyscan's run records where on the grid it was taken, and the run before.
    assert [c["md"] for c in recorder.flyscans] == [
        {
            "grid_index": i,
            "num_grid_points": len(expected),
            "grid_positions": {
                outer.name: pytest.approx(o),
                inner.name: pytest.approx(n),
            },
            **({"previous_grid_point_uid": f"uid-{i - 1}"} if i else {}),
        }
        for i, (o, n) in enumerate(expected)
    ]
    # Step motors go back to where they started.
    assert await outer.user_setpoint.get_value() == pytest.approx(-3.0)
    assert await inner.user_setpoint.get_value() == pytest.approx(7.0)


async def test_tomo_nd_step_scan_only_moves_changed_axes(
    RE: RunEngine,
    recorder: Recorder,
    step_motors: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
):
    outer, inner = step_motors

    RE(
        tomo_nd_step_scan(
            ["det"],  # type: ignore
            outer, 0, 1, 2,
            inner, 10, 20, 3,
            num_projections=10,
            exposure_time=0.1,
            **DEVICES,  # type: ignore
        )
    )  # fmt: skip

    outer_moves = [c.args[0] for c in get_mock_put(outer.user_setpoint).call_args_list]
    inner_moves = [c.args[0] for c in get_mock_put(inner.user_setpoint).call_args_list]
    # two grid rows plus the return to the start position
    assert outer_moves == pytest.approx([0, 1, 0])
    assert inner_moves == pytest.approx([10, 15, 20, 10, 15, 20, 0])


@pytest.mark.parametrize(
    ("alternate", "expected"),
    [
        (False, [(0, 180)] * 4),
        (True, [(0, 180), (180, 0), (0, 180), (180, 0)]),
    ],
)
def test_tomo_nd_step_scan_alternates_flyscan_direction(
    RE: RunEngine,
    recorder: Recorder,
    step_motors: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
    alternate: bool,
    expected: list[tuple[float, float]],
):
    outer, _ = step_motors

    RE(
        tomo_nd_step_scan(
            ["det"],  # type: ignore
            outer, 0, 3, 4,
            num_projections=10,
            exposure_time=0.1,
            start=0,
            stop=180,
            alternate_flyscan_dir=alternate,
            **DEVICES,  # type: ignore
        )
    )  # fmt: skip

    assert [(c["start"], c["stop"]) for c in recorder.flyscans] == expected


def test_tomo_nd_step_scan_forwards_flyscan_arguments(
    RE: RunEngine,
    recorder: Recorder,
    step_motors: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
):
    outer, _ = step_motors

    RE(
        tomo_nd_step_scan(
            ["det"],  # type: ignore
            outer, 0, 1, 2,
            num_projections=100,
            exposure_time=0.2,
            acquire_period=0.3,
            images_to_average=2,
            filter_type=NDProcessFilterType.SUM,
            use_shutter=False,
            sample_name="rock",
            time_based=False,
            **DEVICES,  # type: ignore
        )
    )  # fmt: skip

    expected = {
        "detectors": ["det"],
        "num_projections": 100,
        "exposure_time": 0.2,
        "acquire_period": 0.3,
        "images_to_average": 2,
        "filter_type": NDProcessFilterType.SUM,
        "use_shutter": False,
        "sample_name": "rock",
        "time_based": False,
        **DEVICES,
    }
    assert len(recorder.flyscans) == 2
    for call in recorder.flyscans:
        assert {k: call[k] for k in expected} == expected


@pytest.mark.parametrize(
    ("num_points", "dark_flat_offset", "dark_flat_frequency", "sequence"),
    [
        (3, 0.0, 1, "FFF"),
        (3, 1.0, 0, "DFFFD"),
        (3, 1.0, 1, "DFDFDFD"),
        (5, 1.0, 2, "DFFDFFDFD"),
    ],
)
def test_tomo_nd_step_scan_dark_flat_schedule(
    RE: RunEngine,
    recorder: Recorder,
    step_motors: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
    num_points: int,
    dark_flat_offset: float,
    dark_flat_frequency: int,
    sequence: str,
):
    outer, _ = step_motors

    RE(
        tomo_nd_step_scan(
            ["det"],  # type: ignore
            outer, 0, 1, num_points,
            num_projections=10,
            exposure_time=0.1,
            dark_flat_offset=dark_flat_offset,
            dark_flat_frequency=dark_flat_frequency,
            dark_flat_exposure_time=0.4,
            num_dark_images=3,
            num_flat_images=4,
            **DEVICES,  # type: ignore
        )
    )  # fmt: skip

    assert recorder.sequence == sequence
    for call in recorder.dark_flats:
        assert call["exposure_time"] == 0.4
        assert (call["num_dark_images"], call["num_flat_images"]) == (3, 4)
        assert call["x_offset"] == dark_flat_offset


def test_tomo_nd_step_scan_retries_failed_flyscans(
    RE: RunEngine,
    recorder: Recorder,
    step_motors: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
):
    outer, _ = step_motors
    recorder.failures = [RuntimeError("one"), RuntimeError("two")]

    RE(
        tomo_nd_step_scan(
            ["det"],  # type: ignore
            outer, 0, 1, 2,
            num_projections=10,
            exposure_time=0.1,
            num_retries=2,
            **DEVICES,  # type: ignore
        )
    )  # fmt: skip

    # the first point succeeds on its third attempt, the second on its first
    positions = [c["positions"][0] for c in recorder.flyscans]
    assert positions == pytest.approx([0, 0, 0, 1])
    # Failed attempts are skipped: the second point links to the first's success.
    previous = [c["md"].get("previous_grid_point_uid") for c in recorder.flyscans]
    assert previous == [None, None, None, "uid-2"]


@pytest.mark.parametrize("num_retries", [0, 2])
def test_tomo_nd_step_scan_gives_up_after_retries(
    RE: RunEngine,
    recorder: Recorder,
    step_motors: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
    num_retries: int,
):
    outer, _ = step_motors
    recorder.failures = [RuntimeError("boom")] * (num_retries + 1)

    with pytest.raises(
        RuntimeError,
        match=f"point 0 failed after {num_retries + 1} attempts: boom",
    ):
        RE(
            tomo_nd_step_scan(
                ["det"],  # type: ignore
                outer, 0, 1, 2,
                num_projections=10,
                exposure_time=0.1,
                num_retries=num_retries,
                **DEVICES,  # type: ignore
            )
        )  # fmt: skip

    assert len(recorder.flyscans) == num_retries + 1


@pytest.mark.parametrize("request_cls", [RequestAbort, PlanHalt])
def test_tomo_nd_step_scan_never_retries_an_abort_or_halt(
    RE: RunEngine,
    recorder: Recorder,
    step_motors: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
    request_cls: type[BaseException],
):
    outer, _ = step_motors
    recorder.failures = [request_cls()]

    result = RE(
        tomo_nd_step_scan(
            ["det"],  # type: ignore
            outer, 0, 1, 2,
            num_projections=10,
            exposure_time=0.1,
            num_retries=5,
            **DEVICES,  # type: ignore
        )
    )  # fmt: skip

    assert result.exit_status == "abort"  # type: ignore
    assert len(recorder.flyscans) == 1


def test_tomo_nd_step_scan_checkpoints_before_each_step(
    RE: RunEngine,
    recorder: Recorder,
    step_motors: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
):
    outer, _ = step_motors
    messages: list[Msg] = []
    RE.msg_hook = messages.append  # type: ignore

    RE(
        tomo_nd_step_scan(
            ["det"],  # type: ignore
            outer, 0, 2, 3,
            num_projections=10,
            exposure_time=0.1,
            dark_flat_offset=1.0,
            dark_flat_frequency=2,
            **DEVICES,  # type: ignore
        )
    )  # fmt: skip

    # 3 points and 3 dark/flats (before, after point 2, at the end), plus the one
    # bps.move_per_step adds before each move
    assert recorder.sequence == "DFFDFD"
    assert sum(msg.command == "checkpoint" for msg in messages) == 6 + 3
    # a checkpoint falls between every pair of grid moves
    commands = [msg.command for msg in messages]
    moves = [i for i, msg in enumerate(messages) if msg.command == "set"]
    for before, after in zip([-1, *moves], moves, strict=False):
        assert "checkpoint" in commands[before + 1 : after]


def test_tomo_nd_step_scan_deferred_pause_stops_between_points(
    RE: RunEngine,
    recorder: Recorder,
    step_motors: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
):
    outer, _ = step_motors
    recorder.pause_after_call = 1

    with pytest.raises(RunEngineInterrupted):
        RE(
            tomo_nd_step_scan(
                ["det"],  # type: ignore
                outer, 0, 3, 4,
                num_projections=10,
                exposure_time=0.1,
                **DEVICES,  # type: ignore
            )
        )  # fmt: skip

    # The point in progress finishes, and the scan holds before the next one.
    assert RE.state == "paused"
    assert [c["positions"][0] for c in recorder.flyscans] == pytest.approx([0, 1])

    RE.resume()

    # Resuming picks up at the next point; nothing is repeated or skipped.
    positions = [c["positions"][0] for c in recorder.flyscans]
    assert positions == pytest.approx([0, 1, 2, 3])
    previous = [c["md"].get("previous_grid_point_uid") for c in recorder.flyscans]
    assert previous == [None, "uid-0", "uid-1", "uid-2"]


# --- tomo_1d_step_scan / tomo_2d_step_scan ----------------------------------------


async def test_tomo_1d_step_scan(
    RE: RunEngine,
    recorder: Recorder,
    step_motors: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
    sample_tower: SampleTower,
):
    outer, _ = step_motors

    RE(
        tomo_1d_step_scan(
            ["det"],  # type: ignore
            outer,
            step_start=-1,
            step_stop=1,
            num_tomograms=5,
            num_projections=100,
            exposure_time=0.2,
            start=0,
            stop=360,
            alternate_flyscan_dir=True,
            dark_flat_offset=3.0,
            dark_flat_frequency=2,
            num_dark_images=6,
            num_flat_images=8,
            sample_name="rock",
            sample_tower=sample_tower,
            **DEVICES,  # type: ignore
        )
    )

    positions = [c["positions"][0] for c in recorder.flyscans]
    assert positions == pytest.approx(list(np.linspace(-1, 1, 5)))
    assert [(c["start"], c["stop"]) for c in recorder.flyscans] == [
        (0, 360),
        (360, 0),
        (0, 360),
        (360, 0),
        (0, 360),
    ]
    assert recorder.sequence == "DFFDFFDFD"
    assert all(c["sample_name"] == "rock" for c in recorder.flyscans)
    for call in recorder.dark_flats:
        assert call["sample_tower"] is sample_tower
        assert (call["num_dark_images"], call["num_flat_images"]) == (6, 8)


def test_tomo_1d_step_scan_retries_by_default(
    RE: RunEngine,
    recorder: Recorder,
    step_motors: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
):
    outer, _ = step_motors
    recorder.failures = [RuntimeError("boom")] * 3

    RE(
        tomo_1d_step_scan(
            ["det"],  # type: ignore
            outer,
            step_start=0,
            step_stop=1,
            num_tomograms=1,
            num_projections=10,
            exposure_time=0.1,
            **DEVICES,  # type: ignore
        )
    )

    assert len(recorder.flyscans) == 4


@pytest.mark.parametrize(
    ("snake_axes", "expected_inner"),
    [(False, [5, 6, 5, 6, 5, 6]), (True, [5, 6, 6, 5, 5, 6])],
)
def test_tomo_2d_step_scan(
    RE: RunEngine,
    recorder: Recorder,
    step_motors: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
    snake_axes: bool,
    expected_inner: list[float],
):
    outer, inner = step_motors

    RE(
        tomo_2d_step_scan(
            ["det"],  # type: ignore
            outer, 0, 2, 3,
            inner, 5, 6, 2,
            num_projections=100,
            exposure_time=0.2,
            sample_name="rock",
            snake_axes=snake_axes,
            **DEVICES,  # type: ignore
        )
    )  # fmt: skip

    positions = [(c["positions"][0], c["positions"][1]) for c in recorder.flyscans]
    assert positions == [
        pytest.approx((o, i))
        for o, i in zip([0, 0, 1, 1, 2, 2], expected_inner, strict=True)
    ]
    assert all(c["sample_name"] == "rock" for c in recorder.flyscans)
    assert recorder.sequence == "F" * 6


# ==================================================================================
# Integration
# ==================================================================================


async def test_tomo_2d_step_scan_saves_to_tiled(
    RE: RunEngine,
    kinetix_det_factory: Callable[[int], KinetixDetector],
    fly_rig: Callable[..., FlyRig],
    open_fe_shutter: tuple[Shutter, Shutter],
    step_motors: tuple[AsyncEpicsMotor, AsyncEpicsMotor],
    tiled_client: tuple[Path, Container],
):
    _, client = tiled_client
    outer, inner = step_motors
    fe_shutter, photon_shutter = open_fe_shutter
    ktx = kinetix_det_factory(1)
    set_mock_value(ktx.driver.array_size_x, 4)
    set_mock_value(ktx.driver.array_size_y, 3)
    rig = fly_rig([ktx])
    num_projections = 5
    rig.frames_per_move = num_projections
    rig.photon_shutter = photon_shutter
    rig.write_files = True
    RE.subscribe(TiledWriter(client))

    result = RE(
        _no_sleeps(
            tomo_2d_step_scan(
                [ktx],
                outer, 0, 1, 2,
                inner, 0, 1, 2,
                num_projections=num_projections,
                exposure_time=0.01,
                sample_name="rock",
                panda=rig.panda,  # type: ignore
                rot_motor=rig.rot_motor,
                fe_shutter=fe_shutter,
                photon_shutter=photon_shutter,
            )
        )
    )  # fmt: skip

    uids = result.run_start_uids  # type: ignore
    assert len(uids) == 4
    grid_positions = []
    for i, uid in enumerate(uids):
        run = client[uid]
        start = run.metadata["start"]
        assert start["plan_name"] == "tomo_flyscan"
        assert start["sample_name"] == "rock"
        assert start["num_points"] == num_projections
        assert (start["grid_index"], start["num_grid_points"]) == (i, 4)
        grid_positions.append(start["grid_positions"])
        assert run.metadata["stop"]["exit_status"] == "success"
        primary = run["primary"]
        assert {"kinetix1", "rotation_angle"} <= set(primary)
        assert primary["kinetix1"].shape == (num_projections, 1, 3, 4)
        assert primary["rotation_angle"].shape == (num_projections,)
        # The data read back through tiled is what this point's devices wrote.
        written = rig.written[i]
        np.testing.assert_array_equal(
            primary["kinetix1"].read(), written["kinetix1"][:, np.newaxis]
        )
        np.testing.assert_allclose(
            primary["rotation_angle"].read(), written["rotation_angle"]
        )
    assert grid_positions == [
        {outer.name: o, inner.name: n} for o, n in [(0, 0), (0, 1), (1, 0), (1, 1)]
    ]

    # Walking back from the last point visits every point.
    chain = [uids[-1]]
    while (
        previous := client[chain[-1]].metadata["start"].get("previous_grid_point_uid")
    ):
        chain.append(previous)
    assert chain[::-1] == list(uids)
