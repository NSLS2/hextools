"""XRD plans against a mocked Perkin Elmer (a continuous-acquisition detector)."""

import asyncio
from typing import Any

import pytest
from bluesky import RunEngine
from ophyd_async.core import (
    StaticPathProvider,
    callback_on_mock_execute,
    callback_on_mock_put,
    get_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.adcore import (
    ADImageMode,
    ADWriterFactory,
    ContAcqDetector,
    NDCircularBuffIO,
    NDFileHDF5IO,
    NDProcessIO,
)
from ophyd_async.sim import SimMotor

from hextools.photon_delivery_system import Shutter, ShutterStatus
from hextools.xrd import ensure_cont_acq_exposure, xrd_count, xrd_dark, xrd_scan


@pytest.fixture
def perkin_elmer(static_path_provider: StaticPathProvider) -> ContAcqDetector:
    """A Perkin Elmer acquiring continuously at 0.1 s, as the profile builds it."""
    with init_devices(mock=True):
        pe = ContAcqDetector(
            "PE:",
            ADWriterFactory.hdf(static_path_provider, hinted=False),
            name="perkin_elmer",
            proc_suffix="Proc1:",
        )
    set_mock_value(pe.driver.image_mode, ADImageMode.CONTINUOUS)
    set_mock_value(pe.driver.acquire, True)
    set_mock_value(pe.driver.acquire_time, 0.1)
    set_mock_value(pe.driver.array_size_x, 2048)
    set_mock_value(pe.driver.array_size_y, 2048)
    hdf = pe.get_plugin("hdf", NDFileHDF5IO)
    set_mock_value(hdf.file_path_exists, True)
    cb = pe.get_plugin("cb", NDCircularBuffIO)
    proc = pe.get_plugin("proc", NDProcessIO)

    async def _buffer_frames(value):
        # The CB plugin passes post_count frames; Proc1 averages num_filter of them.
        if not value:
            return
        frames = await cb.post_count.get_value()
        frames //= max(1, await proc.num_filter.get_value())
        set_mock_value(hdf.num_captured, await hdf.num_captured.get_value() + frames)

        async def _done():
            await asyncio.sleep(0.01)
            set_mock_value(cb.trigger_, False)

        asyncio.get_running_loop().create_task(_done())

    callback_on_mock_put(cb.trigger_, _buffer_frames)
    # each run opens a new file, which starts the count again
    callback_on_mock_put(hdf.capture, lambda _: set_mock_value(hdf.num_captured, 0))
    return pe


@pytest.fixture
def open_fe(two_shutters: tuple[Shutter, Shutter]) -> tuple[Shutter, Shutter]:
    fe_shutter, photon_shutter = two_shutters
    set_mock_value(fe_shutter.status, ShutterStatus.OPEN)
    set_mock_value(photon_shutter.status, ShutterStatus.CLOSED)
    return fe_shutter, photon_shutter


def _collect(RE: RunEngine, plan) -> dict[str, list[Any]]:
    docs: dict[str, list[Any]] = {}
    RE(plan, lambda name, doc: docs.setdefault(name, []).append(doc))
    return docs


def test_ensure_cont_acq_exposure_leaves_a_matching_detector_alone(
    RE: RunEngine, perkin_elmer: ContAcqDetector
):
    RE(ensure_cont_acq_exposure(perkin_elmer, 0.1))
    get_mock_put(perkin_elmer.driver.acquire).assert_not_called()
    get_mock_put(perkin_elmer.driver.acquire_time).assert_not_called()


async def test_ensure_cont_acq_exposure_restarts_at_a_new_exposure(
    RE: RunEngine, perkin_elmer: ContAcqDetector
):
    RE(ensure_cont_acq_exposure(perkin_elmer, 0.5))
    acquire_puts = [
        c.args[0] for c in get_mock_put(perkin_elmer.driver.acquire).call_args_list
    ]
    assert acquire_puts == [False, True]
    assert await perkin_elmer.driver.acquire_time.get_value() == 0.5
    assert await perkin_elmer.driver.image_mode.get_value() == ADImageMode.CONTINUOUS


async def test_xrd_count_takes_averaged_images_in_each_run(
    RE: RunEngine, perkin_elmer: ContAcqDetector, open_fe: tuple[Shutter, Shutter]
):
    fe_shutter, photon_shutter = open_fe
    docs = _collect(
        RE,
        xrd_count(
            0.2,
            num_images=3,
            images_to_average=5,
            num_iterations=2,
            sample_name="LaB6",
            detector=perkin_elmer,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
        ),
    )
    assert [d["plan_name"] for d in docs["start"]] == ["xrd_count", "xrd_count"]
    assert all(d["sample_name"] == "LaB6" for d in docs["start"])
    assert len(docs["event"]) == 2
    # one event per run, holding num_images averaged frames
    assert [d["indices"] for d in docs["stream_datum"]] == [{"start": 0, "stop": 1}] * 2
    assert docs["descriptor"][0]["data_keys"]["perkin_elmer"]["shape"][0] == 3
    assert await perkin_elmer.driver.acquire_time.get_value() == 0.2
    proc = perkin_elmer.get_plugin("proc", NDProcessIO)
    assert await proc.num_filter.get_value() == 5
    assert await photon_shutter.status.get_value() == ShutterStatus.CLOSED


async def test_xrd_dark_never_opens_the_photon_shutter(
    RE: RunEngine, perkin_elmer: ContAcqDetector, open_fe: tuple[Shutter, Shutter]
):
    fe_shutter, photon_shutter = open_fe
    set_mock_value(photon_shutter.status, ShutterStatus.OPEN)
    opened: list[object] = []
    callback_on_mock_execute(photon_shutter.open_cmd, lambda *a: opened.append(a))
    docs = _collect(
        RE,
        xrd_dark(
            0.1,
            num_images=4,
            detector=perkin_elmer,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
        ),
    )
    assert [d["plan_name"] for d in docs["start"]] == ["xrd_dark"]
    assert opened == []
    assert await photon_shutter.status.get_value() == ShutterStatus.CLOSED


async def test_xrd_scan_takes_one_averaged_image_per_point(
    RE: RunEngine, perkin_elmer: ContAcqDetector, open_fe: tuple[Shutter, Shutter]
):
    fe_shutter, photon_shutter = open_fe
    with init_devices():
        det_z = SimMotor(name="det_z", instant=True)
    docs = _collect(
        RE,
        xrd_scan(
            det_z,
            0.0,
            4.0,
            5,
            0.1,
            images_to_average=3,
            num_iterations=2,
            detector=perkin_elmer,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
        ),
    )
    assert [d["plan_name"] for d in docs["start"]] == ["xrd_scan", "xrd_scan"]
    assert len(docs["event"]) == 10
    positions = [e["data"]["det_z"] for e in docs["event"]]
    assert positions == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0] * 2)
    proc = perkin_elmer.get_plugin("proc", NDProcessIO)
    assert await proc.num_filter.get_value() == 3
    assert await det_z.user_readback.get_value() == 0.0
    assert await photon_shutter.status.get_value() == ShutterStatus.CLOSED


async def test_xrd_scan_closes_the_shutter_and_returns_the_motor_on_failure(
    RE: RunEngine, perkin_elmer: ContAcqDetector, open_fe: tuple[Shutter, Shutter]
):
    fe_shutter, photon_shutter = open_fe
    with init_devices():
        det_z = SimMotor(name="det_z", instant=True)
    hdf = perkin_elmer.get_plugin("hdf", NDFileHDF5IO)
    cb = perkin_elmer.get_plugin("cb", NDCircularBuffIO)
    triggers: list[bool] = []

    async def _fault_on_second_point(value):
        if not value:
            return
        triggers.append(value)
        if len(triggers) == 2:
            raise RuntimeError("detector fault")
        set_mock_value(hdf.num_captured, await hdf.num_captured.get_value() + 1)
        asyncio.get_running_loop().call_later(0.01, set_mock_value, cb.trigger_, False)

    callback_on_mock_put(cb.trigger_, _fault_on_second_point)
    with pytest.raises(Exception, match="detector fault"):
        RE(
            xrd_scan(
                det_z,
                1.0,
                3.0,
                3,
                0.1,
                detector=perkin_elmer,
                fe_shutter=fe_shutter,
                photon_shutter=photon_shutter,
            )
        )
    assert await det_z.user_readback.get_value() == 1.0
    assert await photon_shutter.status.get_value() == ShutterStatus.CLOSED
