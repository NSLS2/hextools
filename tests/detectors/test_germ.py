import asyncio
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import tifffile
from bluesky import RunEngine
from bluesky import plans as bp
from bluesky_tiled_plugins import TiledWriter
from ophyd_async.core import (
    DetectorTrigger,
    TriggerInfo,
    callback_on_mock_put,
    init_devices,
    set_mock_value,
)
from tiled.client.container import Container

from hextools.detectors.germ import (
    GeRMAcquireLogic,
    GeRMDetector,
    GeRMDetectorIO,
    GeRMDetectorMock,
    GeRMTriggerLogic,
)


@pytest.fixture
async def germ_io():
    async with init_devices(mock=True):
        germ_io = GeRMDetectorIO("GeRM:")
    return germ_io


@pytest.fixture
def germ_trigger_logic(germ_io):
    return GeRMTriggerLogic(germ_io)


async def test_germ_trigger_logic_default_trigger_info(
    germ_trigger_logic: GeRMTriggerLogic,
):
    set_mock_value(germ_trigger_logic.driver.acquire_time, 0.25)
    default_trig_info = await germ_trigger_logic.default_trigger_info()
    assert default_trig_info == TriggerInfo(
        livetime=0.25,
        deadtime=0.0,
        exposures_per_collection=1,
        collections_per_event=1,
        number_of_events=1,
        trigger=DetectorTrigger.INTERNAL,
    )


async def test_germ_trigger_logic_prepare_internal(
    germ_trigger_logic: GeRMTriggerLogic,
):
    with pytest.raises(ValueError, match="Only a single"):
        await germ_trigger_logic.prepare_internal(2, 0.25, 0.0)


async def test_germ_acquire_logic(germ_io: GeRMDetectorIO):
    acquire_logic = GeRMAcquireLogic(germ_io)

    async def _finish_acquiring():
        await asyncio.sleep(0.1)
        set_mock_value(germ_io.acquire, False)

    # The put writes its value after the callback returns, so finish in a separate task
    finishing: list[asyncio.Task] = []

    def _on_acquire(value: bool):
        if value:
            finishing.append(asyncio.create_task(_finish_acquiring()))

    callback_on_mock_put(germ_io.acquire, _on_acquire)
    assert await germ_io.acquire.get_value() is False
    await acquire_logic.start_acquiring()
    assert await germ_io.acquire.get_value() is True
    await acquire_logic.wait_for_idle()
    assert await germ_io.acquire.get_value() is False
    await acquire_logic.ensure_stopped()
    assert await germ_io.acquire.get_value() is False


# --- GeRMDetectorMock ---------------------------------------------------------------

NUM_ELEMENTS = GeRMDetectorMock.num_elements
NUM_BINS = 4096


def _count(
    RE: RunEngine, germ: GeRMDetector, client: Container, num: int
) -> tuple[dict[str, list], np.ndarray]:
    docs: dict[str, list] = {}
    RE.subscribe(TiledWriter(client))
    RE(
        bp.count([germ], num=num),
        lambda name, doc: docs.setdefault(name, []).append(doc),
    )
    run = client.values().last()
    return docs, run["primary"]["germ"].read()


async def test_germ_mock_defaults(germ: GeRMDetector):
    assert await germ.driver.num_elements.get_value() == NUM_ELEMENTS
    assert await germ.driver.model.get_value() == "Mock GeRM"
    assert await germ.hdf.file_path_exists.get_value()
    assert await germ.tiff.file_path_exists.get_value()


async def test_germ_mock_count_writes_hdf5(
    RE: RunEngine, germ: GeRMDetector, tiled_client: tuple[Path, Container]
):
    data_dir, client = tiled_client
    docs, data = _count(RE, germ, client, num=3)

    (resource,) = docs["stream_resource"]
    assert resource["mimetype"] == "application/x-hdf5"
    assert resource["parameters"]["dataset"] == "/entry/data/data"
    (h5_file,) = data_dir.glob("*.h5")
    with h5py.File(h5_file) as f:
        dataset = f["/entry/data/data"]
        assert isinstance(dataset, h5py.Dataset)
        written: np.ndarray = dataset[()]
    assert written.shape == (3, NUM_ELEMENTS, NUM_BINS)
    assert written.dtype == np.int32

    # tiled serves exactly the frames on disk, one per event
    np.testing.assert_array_equal(data.reshape(written.shape), written)
    totals = [event["data"]["germ-stats1-total"] for event in docs["event"]]
    assert totals == [float(frame.sum()) for frame in written]
    assert await germ.hdf.num_captured.get_value() == 3


async def test_germ_mock_count_writes_tiffs(
    RE: RunEngine, germ: GeRMDetector, tiled_client: tuple[Path, Container]
):
    data_dir, client = tiled_client
    germ.save_as_tiff()
    docs, data = _count(RE, germ, client, num=2)

    (resource,) = docs["stream_resource"]
    assert resource["mimetype"] == "multipart/related;type=image/tiff"
    tiffs = sorted(data_dir.glob("*.tiff"))
    assert [p.name[-12:] for p in tiffs] == ["_000000.tiff", "_000001.tiff"]
    written = np.stack([tifffile.imread(p) for p in tiffs])
    assert written.shape == (2, NUM_ELEMENTS, NUM_BINS)
    assert not list(data_dir.glob("*.h5"))

    np.testing.assert_array_equal(data.reshape(written.shape), written)
    totals = [event["data"]["germ-stats1-total"] for event in docs["event"]]
    assert totals == [float(frame.sum()) for frame in written]


async def test_germ_mock_count_updates_live_signals(germ: GeRMDetector):
    await germ.driver.acquire.set(True)
    await asyncio.wait_for(_wait_idle(germ), timeout=5)

    spectrum = await germ.driver.spectrum.get_value()
    intensity = await germ.driver.intensity.get_value()
    energies = await germ.driver.spectrum_x.get_value()
    assert spectrum.shape == (NUM_BINS,)
    assert intensity.shape == (NUM_ELEMENTS,)
    assert spectrum.sum() == intensity.sum() == await germ.stats1.total.get_value()
    # the default 120 keV gain sets the energy axis
    assert energies[0] == 0.0
    assert energies[-1] == pytest.approx(120.0)
    assert await germ.driver.elapsed_time.get_value() == pytest.approx(0.01)
    # nothing was capturing, so nothing was written
    assert await germ.hdf.num_captured.get_value() == 0


async def test_germ_mock_stopped_count_writes_no_frame(germ: GeRMDetector):
    set_mock_value(germ.driver.acquire_time, 5.0)
    set_mock_value(germ.hdf.capture, True)
    await germ.driver.acquire.set(True)
    await asyncio.sleep(0.05)
    await germ.driver.acquire.set(False)
    await asyncio.sleep(0.05)

    assert await germ.hdf.num_captured.get_value() == 0
    assert await germ.stats1.total.get_value() == 0.0


async def _wait_idle(germ: GeRMDetector):
    while await germ.driver.acquire.get_value():
        await asyncio.sleep(0.01)


async def _take_frame(germ: GeRMDetector) -> np.ndarray:
    frames: list[np.ndarray] = []
    mock = germ._mock
    assert isinstance(mock, GeRMDetectorMock)
    mock.register_frame_callback(frames.append)
    frames.clear()
    await germ.driver.acquire.set(True)
    await asyncio.wait_for(_wait_idle(germ), timeout=5)
    mock.unregister_frame_callback(frames.append)
    (frame,) = frames
    assert frame.shape == (NUM_ELEMENTS, NUM_BINS)
    assert float(frame.sum()) == await germ.stats1.total.get_value()
    return frame


@pytest.fixture
def germ_mock(germ: GeRMDetector) -> GeRMDetectorMock:
    assert isinstance(germ._mock, GeRMDetectorMock)
    return germ._mock


async def test_germ_mock_frame_callback_gets_each_frame(germ: GeRMDetector):
    frame = await _take_frame(germ)
    np.testing.assert_array_equal(await germ.driver.mca.get_value(), frame.ravel())
    assert frame.sum() > 0


async def test_germ_mock_sends_zeros_when_a_callback_registers(
    germ_mock: GeRMDetectorMock,
):
    frames: list[np.ndarray] = []
    germ_mock.register_frame_callback(frames.append)
    (frame,) = frames
    assert frame.shape == (NUM_ELEMENTS, NUM_BINS)
    assert not frame.any()


async def test_germ_mock_frame_updates_each_second_while_counting(
    germ: GeRMDetector, germ_mock: GeRMDetectorMock
):
    frames: list[np.ndarray] = []
    germ_mock.register_frame_callback(frames.append)
    frames.clear()
    set_mock_value(germ.driver.acquire_time, 2.5)
    await germ.driver.acquire.set(True)
    await asyncio.wait_for(_wait_idle(germ), timeout=10)

    # at 1 s, 2 s, then the final frame, each accumulating on the last
    assert len(frames) == 3
    assert (frames[1] >= frames[0]).all() and (frames[2] >= frames[1]).all()
    assert frames[0].sum() < frames[1].sum() < frames[2].sum()


async def test_germ_mock_closed_shutter_gives_zeros(
    germ: GeRMDetector, germ_mock: GeRMDetectorMock, two_shutters
):
    fe, photon = two_shutters
    germ_mock.set_shutters(fe, photon)
    await fe.set(True)
    assert not (await _take_frame(germ)).any()


async def test_germ_mock_open_shutters_count_with_exposure(
    germ: GeRMDetector, germ_mock: GeRMDetectorMock, two_shutters
):
    germ_mock.set_shutters(*two_shutters)
    for shutter in two_shutters:
        await shutter.set(True)
    short = await _take_frame(germ)
    set_mock_value(germ.driver.acquire_time, 0.1)
    long = await _take_frame(germ)
    assert long.sum() > 5 * short.sum() > 0


async def test_germ_mock_test_pulses(
    germ: GeRMDetector, germ_mock: GeRMDetectorMock, two_shutters
):
    germ_mock.set_shutters(*two_shutters)
    set_mock_value(germ.driver.test_pulse_enable, True)
    pulses = await _take_frame(germ)
    # same peak in every strip, 1000 Hz for 0.01 s, centred by the amplitude
    assert (pulses == pulses[0]).all()
    assert pulses[0].sum() == pytest.approx(10, abs=2)
    assert abs(int(pulses[0].argmax()) - round(100 / 1024 * (NUM_BINS - 1))) <= 1

    for shutter in two_shutters:
        await shutter.set(True)
    both = await _take_frame(germ)
    signal = both - pulses
    assert (signal >= 0).all()
    assert signal.sum() > 0


# Copilot on #109: the last step slept a full 0.1 s, past the exposure.
async def test_germ_mock_sleeps_no_longer_than_the_exposure(
    germ: GeRMDetector, monkeypatch
):
    import hextools.detectors.germ as germ_module

    requested: list[float] = []
    real_sleep = asyncio.sleep

    async def record(delay: float, *args, **kwargs):
        requested.append(delay)
        await real_sleep(0)

    proxy = SimpleNamespace(**vars(asyncio))
    proxy.sleep = record
    monkeypatch.setattr(germ_module, "asyncio", proxy)
    set_mock_value(germ.driver.acquire_time, 0.15)

    await germ.driver.acquire.set(True)
    await asyncio.wait_for(_wait_idle(germ), timeout=5)

    assert sum(requested) == pytest.approx(0.15)
    assert max(requested) <= 0.1


# Copilot on #109: turning test pulses off mid-count recomputed the pulses
# already counted with the new settings, so they vanished from the frame.
async def test_germ_mock_keeps_pulses_counted_before_they_were_turned_off(
    germ: GeRMDetector, germ_mock: GeRMDetectorMock, two_shutters
):
    germ_mock.set_shutters(*two_shutters)  # closed: only test pulses count
    driver = germ.driver
    set_mock_value(driver.test_pulse_enable, True)
    set_mock_value(driver.acquire_time, 0.4)
    frames: list[np.ndarray] = []
    germ_mock.register_frame_callback(frames.append)

    await driver.acquire.set(True)
    await asyncio.sleep(0.25)
    # what configure_test_pulses(False) leaves behind
    set_mock_value(driver.test_pulse_enable, False)
    for signal in (
        driver.test_pulse_amplitude,
        driver.test_pulse_frequency,
        driver.test_pulse_count,
    ):
        set_mock_value(signal, 0)
    await asyncio.wait_for(_wait_idle(germ), timeout=5)

    assert frames[-1].any()
