import asyncio
from pathlib import Path

import pytest
from ophyd_async.core import (
    DetectorTrigger,
    StaticFilenameProvider,
    StaticPathProvider,
    TriggerInfo,
    init_devices,
    set_mock_value,
)

from hextools.detectors.germ import (
    GeRMAcquireLogic,
    GeRMDetector,
    GeRMDetectorIO,
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


async def test_germ_trigger_logic_prepare_internal_sets_exposure(
    germ_trigger_logic: GeRMTriggerLogic,
):
    set_mock_value(germ_trigger_logic.driver.acquire_time, 1.5)
    await germ_trigger_logic.prepare_internal(1, 0.25, 0.0)
    assert await germ_trigger_logic.driver.acquire_time.get_value() == 0.25


async def test_germ_trigger_logic_prepare_internal_zero_livetime_keeps_exposure(
    germ_trigger_logic: GeRMTriggerLogic,
):
    # livetime=0 means "leave the configured exposure alone", per the docstring.
    set_mock_value(germ_trigger_logic.driver.acquire_time, 1.5)
    await germ_trigger_logic.prepare_internal(1, 0.0, 0.0)
    assert await germ_trigger_logic.driver.acquire_time.get_value() == 1.5


async def test_germ_acquire_logic(germ_io: GeRMDetectorIO):
    acquire_logic = GeRMAcquireLogic(germ_io)
    set_mock_value(germ_io.acquire_time, 0.01)

    assert await germ_io.acquire.get_value() is False
    await acquire_logic.start_acquiring()
    assert await germ_io.acquire.get_value() is True

    # wait_for_idle returns when the IOC clears Acquire at the end of the count.
    # Drive that transition explicitly rather than sleeping past it.
    idle = asyncio.create_task(acquire_logic.wait_for_idle())
    await asyncio.sleep(0)  # let wait_for_idle subscribe before the value moves
    set_mock_value(germ_io.acquire, False)
    await asyncio.wait_for(idle, timeout=10)

    assert await germ_io.acquire.get_value() is False
    await acquire_logic.ensure_stopped()
    assert await germ_io.acquire.get_value() is False


@pytest.fixture
def germ_detector(RE, tmp_path: Path) -> GeRMDetector:
    with init_devices(mock=True):
        germ = GeRMDetector(
            "GeRM:",
            StaticPathProvider(StaticFilenameProvider("scan"), tmp_path),
            name="germ",
        )
    set_mock_value(germ.hdf.file_path_exists, True)
    return germ


async def test_germ_detector_describes_elements_then_energy_bins(
    germ_detector: GeRMDetector,
):
    """The frame is (elements, energy bins), not the other way round.

    GeRM reads out one spectrum per detector element, so the element axis comes
    first. Nothing constructed GeRMDetector before this test, which is how a
    transposed NDArrayDescription reached a branch unnoticed.
    """
    set_mock_value(germ_detector.driver.num_elements, 192)

    await germ_detector.prepare(
        TriggerInfo(
            trigger=DetectorTrigger.INTERNAL,
            livetime=0.25,
            deadtime=0,
            exposures_per_collection=1,
            collections_per_event=1,
            number_of_events=1,
        )
    )

    desc = await germ_detector.describe()
    # (frames per event, elements, energy bins). num_energy_bins is a soft
    # signal fixed at 4096. If the element and energy axes are ever swapped
    # again, this reads [1, 4096, 192] and fails.
    assert desc["germ"]["shape"] == [1, 192, 4096]
    assert desc["germ"]["dtype"] == "array"
