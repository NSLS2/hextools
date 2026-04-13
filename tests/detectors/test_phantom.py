import asyncio
from pathlib import Path

import bluesky.plans as bp
import h5py
import numpy as np
import pytest
from bluesky.run_engine import RunEngine
from bluesky_tiled_plugins import TiledWriter
from ophyd_async.core import (
    DetectorTrigger,
    StaticFilenameProvider,
    StaticPathProvider,
    TriggerInfo,
    callback_on_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.adcore import ADBaseDataType

import hextools.detectors.phantom
from hextools.detectors.phantom import (
    PhantomArmLogic,
    PhantomDetector,
    PhantomIO,
    PhantomPixelDataFormat,
    PhantomTriggerLogic,
)


@pytest.fixture
def phantom_io(RE: RunEngine) -> PhantomIO:
    with init_devices(mock=True):
        phantom = PhantomIO("TEST:PHANTOM")
    return phantom


@pytest.fixture
def phantom_trigger_logic(phantom_io: PhantomIO):
    return PhantomTriggerLogic(phantom_io)


@pytest.fixture
def phantom_arm_logic(phantom_io: PhantomIO):
    return PhantomArmLogic(phantom_io)


@pytest.fixture
def phantom_detector_factory():
    def _factory(write_path: Path):
        return PhantomDetector(
            prefix="TEST:PHANTOM",
            path_provider=StaticPathProvider(
                StaticFilenameProvider("scan"), write_path
            ),
            name="phantom",
        )

    return _factory


@pytest.fixture
def phantom_detector(RE, phantom_detector_factory, tmp_path: Path) -> PhantomDetector:
    with init_devices(mock=True):
        phantom = phantom_detector_factory(tmp_path)

    set_mock_value(phantom.writer.file_path_exists, True)
    return phantom


@pytest.mark.parametrize(
    ("pixel_format", "expected_dtype"),
    (
        (PhantomPixelDataFormat.EIGHT, ADBaseDataType.UINT8),
        (PhantomPixelDataFormat.EIGHT_R, ADBaseDataType.UINT8),
        *[
            (fmt, ADBaseDataType.UINT16)
            for fmt in PhantomPixelDataFormat
            if fmt not in (PhantomPixelDataFormat.EIGHT, PhantomPixelDataFormat.EIGHT_R)
        ],
    ),
)
def test_get_downloaded_dtype(
    phantom_io: PhantomIO,
    pixel_format: PhantomPixelDataFormat,
    expected_dtype: ADBaseDataType,
):
    assert phantom_io.get_downloaded_dtype(pixel_format) == expected_dtype


@pytest.mark.parametrize(
    ("start", "end", "expected_num"),
    ((0, 1, 2), (-15, 10, 26), (5, 15, 11), (10, 10, 1), (15, 5, 0)),
)
def test_get_downloaded_frames(
    phantom_io: PhantomIO, start: int, end: int, expected_num: int
):
    assert phantom_io.get_total_downloaded_frames(start, end) == expected_num


def test_trigger_logic_config_sigs(phantom_trigger_logic: PhantomTriggerLogic):
    config_sigs = phantom_trigger_logic.config_sigs()
    assert isinstance(config_sigs, set)
    assert len(config_sigs) == 10
    assert phantom_trigger_logic.driver.acquire_time_ms in config_sigs


@pytest.mark.parametrize("num_frames", (0, -1, -10))
async def test_trigger_logic_setup_download_req_num_greater_than_zero(
    phantom_trigger_logic: PhantomTriggerLogic, num_frames: int
):
    with pytest.raises(
        ValueError, match="Number of frames to download must be greater than 0!"
    ):
        await phantom_trigger_logic.setup_download(num_frames)


@pytest.mark.parametrize(
    ("post_trig_frames", "num", "expected_start", "expected_end"),
    ((1, 10, -9, 0), (5, 10, -5, 4), (10, 10, 0, 9), (15, 10, 0, 9)),
)
async def test_trigger_logic_setup_download(
    phantom_trigger_logic: PhantomTriggerLogic,
    post_trig_frames: int,
    num: int,
    expected_start: int,
    expected_end: int,
):
    set_mock_value(phantom_trigger_logic.driver.post_trig_frames, post_trig_frames)
    await phantom_trigger_logic.setup_download(num)
    # await asyncio.sleep(0.1)  # Allow time for async operations to complete
    assert (
        await phantom_trigger_logic.driver.download_start_frame.get_value()
        == expected_start
    )
    assert (
        await phantom_trigger_logic.driver.download_end_frame.get_value()
        == expected_end
    )


async def test_arm_logic_arm_timeout_waiting_for_trigger(
    phantom_arm_logic: PhantomArmLogic, monkeypatch
):
    monkeypatch.setattr(
        hextools.detectors.phantom, "DEFAULT_TIMEOUT", 0.1
    )  # Set a short timeout for the test

    # Pre-set waiting_for_trigger so set_and_wait_for_other_value completes immediately
    set_mock_value(phantom_arm_logic.driver.waiting_for_trigger, 1)

    # Acquisition stop after a short delay, so it takes effect after acquire.set(True)
    async def _stop_acquisition():
        await asyncio.sleep(0.05)
        set_mock_value(phantom_arm_logic.driver.acquire, False)

    stop_task = asyncio.create_task(_stop_acquisition())
    try:
        with pytest.raises(
            RuntimeError, match="Acquisition stopped while waiting for event trigger!"
        ):
            await phantom_arm_logic.arm()
    finally:
        stop_task.cancel()


async def test_arm_logic_arm_timeout_saving_to_cine(
    phantom_arm_logic: PhantomArmLogic, monkeypatch
):
    monkeypatch.setattr(
        hextools.detectors.phantom, "DEFAULT_TIMEOUT", 0.1
    )  # Set a short timeout for the test

    # Pre-set waiting_for_trigger and trigger_received so the first two
    # loops complete immediately
    set_mock_value(phantom_arm_logic.driver.waiting_for_trigger, 1)
    set_mock_value(phantom_arm_logic.driver.trigger_received, 1)
    # Set post_trig_frames to a value that won't match array_counter,
    # so the second loop times out
    set_mock_value(phantom_arm_logic.driver.post_trig_frames, 10)

    with pytest.raises(
        TimeoutError,
        match="Received event trigger, but writing to cine was not completed!",
    ):
        await phantom_arm_logic.arm()


async def test_arm_logic_arm_post_trig_frames_incorrect(
    phantom_arm_logic: PhantomArmLogic, monkeypatch
):
    monkeypatch.setattr(
        hextools.detectors.phantom, "DEFAULT_TIMEOUT", 0.1
    )  # Set a short timeout for the test

    # Pre-set waiting_for_trigger and trigger_received so the first two loops
    # complete immediately
    set_mock_value(phantom_arm_logic.driver.waiting_for_trigger, 1)
    set_mock_value(phantom_arm_logic.driver.trigger_received, 1)
    set_mock_value(phantom_arm_logic.driver.post_trig_frames, 10)
    set_mock_value(phantom_arm_logic.driver.complete_and_valid, 1)
    set_mock_value(
        phantom_arm_logic.driver.array_counter, 5
    )  # Different from post_trig_frames

    with pytest.raises(
        ValueError,
        match="Expected number of post trig frames .* does not match actual number .*",
    ):
        await phantom_arm_logic.arm()


async def test_arm_logic_arm_success(phantom_arm_logic: PhantomArmLogic, monkeypatch):
    monkeypatch.setattr(
        hextools.detectors.phantom, "DEFAULT_TIMEOUT", 0.1
    )  # Set a short timeout for the test

    # Pre-set waiting_for_trigger and trigger_received so the first two loops
    # complete immediately
    set_mock_value(phantom_arm_logic.driver.waiting_for_trigger, 1)
    set_mock_value(phantom_arm_logic.driver.trigger_received, 1)
    set_mock_value(phantom_arm_logic.driver.post_trig_frames, 10)
    set_mock_value(phantom_arm_logic.driver.complete_and_valid, 1)
    set_mock_value(
        phantom_arm_logic.driver.array_counter, 10
    )  # Matches post_trig_frames

    await phantom_arm_logic.arm()  # Should complete without exceptions
    assert await phantom_arm_logic.driver.download.get_value()


async def test_arm_logic_wait_for_idle_timeout(
    phantom_arm_logic: PhantomArmLogic, monkeypatch
):
    monkeypatch.setattr(
        hextools.detectors.phantom, "DEFAULT_TIMEOUT", 0.1
    )  # Set a short timeout for the test
    set_mock_value(phantom_arm_logic.driver.download_start_frame, -5)
    set_mock_value(phantom_arm_logic.driver.download_end_frame, 5)
    set_mock_value(phantom_arm_logic.driver.download, True)
    with pytest.raises(
        TimeoutError,
        match="Timeout waiting for download to complete! Target number of downloaded frames: 11",  # noqa: E501
    ):
        await phantom_arm_logic.wait_for_idle()


async def test_arm_logic_wait_for_idle_success(
    phantom_arm_logic: PhantomArmLogic, monkeypatch
):
    monkeypatch.setattr(
        hextools.detectors.phantom, "DEFAULT_TIMEOUT", 0.1
    )  # Set a short timeout for the test
    set_mock_value(phantom_arm_logic.driver.download_start_frame, -5)
    set_mock_value(phantom_arm_logic.driver.download_end_frame, 5)
    set_mock_value(phantom_arm_logic.driver.download, True)

    # Simulate frames being downloaded by incrementing download_start_frame
    # until it exceeds download_end_frame
    async def _simulate_download():
        while True:
            await asyncio.sleep(0.01)
            count = await phantom_arm_logic.driver.download_count.get_value()
            if count >= 11:  # Total frames to download is 11 (-5 to 5 inclusive)
                break
            set_mock_value(phantom_arm_logic.driver.download_count, count + 1)

    download_task = asyncio.create_task(_simulate_download())
    try:
        await phantom_arm_logic.wait_for_idle()  # Should complete without exceptions
    finally:
        download_task.cancel()


@pytest.mark.parametrize(
    ("pixel_format", "x_size", "y_size", "expected_shape", "expected_dtype"),
    (
        (PhantomPixelDataFormat.EIGHT, 10, 20, [1, 20, 10], "|u1"),
        (PhantomPixelDataFormat.EIGHT_R, 15, 25, [1, 25, 15], "|u1"),
        (PhantomPixelDataFormat.P_TEN, 5, 5, [1, 5, 5], "<u2"),
        (PhantomPixelDataFormat.P_SIXTEEN, 8, 12, [1, 12, 8], "<u2"),
    ),
)
async def test_detector_describe(
    phantom_detector,
    pixel_format: PhantomPixelDataFormat,
    x_size: int,
    y_size: int,
    expected_shape: tuple[int, int, int],
    expected_dtype: ADBaseDataType,
):
    set_mock_value(phantom_detector.driver.select_pixel_data_format, pixel_format)
    set_mock_value(phantom_detector.driver.array_size_x, x_size)
    set_mock_value(phantom_detector.driver.array_size_y, y_size)
    await phantom_detector.prepare(
        TriggerInfo(
            trigger=DetectorTrigger.INTERNAL,
            livetime=0,
            deadtime=0,
            exposures_per_collection=1,
            collections_per_event=1,
            number_of_events=1,
        )
    )

    desc = await phantom_detector.describe()
    assert desc["phantom"]["shape"] == expected_shape
    assert desc["phantom"]["dtype"] == "array"
    assert desc["phantom"]["dtype_numpy"] == expected_dtype
    assert desc["phantom"]["source"].endswith("scan.h5")


async def test_detector_full_stack(
    RE, phantom_detector_factory, tiled_client, monkeypatch
):

    monkeypatch.setenv(
        "OPHYD_ASYNC_PRESERVE_DETECTOR_STATE", "YES"
    )  # Ensure detector config is preserved across stages
    tmp_path, c = tiled_client
    with init_devices(mock=True):
        phantom = phantom_detector_factory(tmp_path)
    tiled_writer = TiledWriter(c)
    docs_cache: dict[str, list] = {}

    RE.subscribe(tiled_writer)
    RE.subscribe(lambda name, doc: docs_cache.setdefault(name, []).append(doc))

    set_mock_value(phantom.writer.file_path_exists, True)
    set_mock_value(
        phantom.driver.select_pixel_data_format, PhantomPixelDataFormat.P_TEN
    )
    set_mock_value(phantom.driver.array_size_x, 4)
    set_mock_value(phantom.driver.array_size_y, 3)
    set_mock_value(phantom.driver.post_trig_frames, 15)
    set_mock_value(phantom.driver.download_start_frame, -5)
    set_mock_value(phantom.driver.download_end_frame, 5)

    def _on_acquire(value, **kwargs):
        if value:
            set_mock_value(phantom.driver.waiting_for_trigger, 1)
            set_mock_value(phantom.driver.trigger_received, 1)
            set_mock_value(phantom.driver.array_counter, 15)
            set_mock_value(phantom.driver.complete_and_valid, 1)

    def _on_download(value, **kwargs):
        if value:
            for count in range(1, 12):
                set_mock_value(phantom.driver.download_count, count)
                set_mock_value(phantom.writer.num_captured, count)

        with h5py.File(tmp_path / "scan.h5", "w") as f:
            f.create_dataset(
                "/entry/data/data",
                data=np.random.randint(0, 65536, size=(11, 3, 4), dtype=np.uint16),
            )

    callback_on_mock_put(phantom.driver.acquire, _on_acquire)
    callback_on_mock_put(phantom.driver.download, _on_download)

    RE(bp.count([phantom]))

    assert "start" in docs_cache
    assert "descriptor" in docs_cache
    assert "event" in docs_cache
    assert "stream_resource" in docs_cache
    assert "stream_datum" in docs_cache
    assert "stop" in docs_cache

    assert len(docs_cache["start"]) == 1
    assert len(docs_cache["descriptor"]) == 1
    assert len(docs_cache["event"]) == 1
    assert len(docs_cache["stream_resource"]) == 1
    assert len(docs_cache["stream_datum"]) == 1
    assert len(docs_cache["stop"]) == 1

    assert "phantom" in docs_cache["descriptor"][0]["data_keys"]
    desc = docs_cache["descriptor"][0]["data_keys"]["phantom"]
    assert desc["shape"] == [11, 3, 4]
    assert desc["dtype"] == "array"
    assert desc["dtype_numpy"] == "<u2"
    assert desc["source"].endswith("scan.h5")

    sres = docs_cache["stream_resource"][0]
    assert sres["uri"] == f"file://localhost{tmp_path / 'scan.h5'}"
    assert sres["parameters"]["dataset"] == "/entry/data/data"
    assert sres["parameters"]["chunk_shape"] == (1, 3, 4)

    datum = docs_cache["stream_datum"][0]
    assert datum["stream_resource"] == sres["uid"]
    assert datum["indices"] == {"start": 0, "stop": 1}
    assert datum["seq_nums"] == {"start": 1, "stop": 2}

    # Check that the data we get from the stream is what we expect
    # (the random data we wrote to the file in _on_download)
    run = c.values().last()
    assert "primary" in run
    assert "phantom" in run["primary"]
    data = run["primary"]["phantom"].read()
    assert data.shape == (11, 3, 4)
    assert data.dtype == np.uint16
