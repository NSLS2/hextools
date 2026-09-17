import asyncio

import pytest

from ophyd_async.core import (
    TriggerInfo,
    callback_on_mock_put,
    set_mock_value,
    DetectorTrigger,
)
from hextools.detectors.germ import (
    GeRMTriggerLogic,
    GeRMAcquireLogic,
    GeRMDetector,
    GeRMDetectorIO,
)
from ophyd_async.core import init_devices


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

    async def _mock_wait_for_idle(value: bool):
        if value:
            await asyncio.sleep(0.1)
            set_mock_value(germ_io.acquire, False)

    callback_on_mock_put(germ_io.acquire, _mock_wait_for_idle)
    assert await germ_io.acquire.get_value() is False
    await acquire_logic.start_acquiring()
    assert await germ_io.acquire.get_value() is True
    await asyncio.sleep(0.2)
    assert await germ_io.acquire.get_value() is False
    await acquire_logic.ensure_stopped()
    assert await germ_io.acquire.get_value() is False
