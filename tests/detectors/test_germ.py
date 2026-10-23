import asyncio
from pathlib import Path

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
