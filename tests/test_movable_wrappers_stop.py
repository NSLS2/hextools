"""Devices whose set() moves motors stop those motors when the RunEngine stops them."""

import asyncio
from collections.abc import Callable

import pytest
from bluesky.protocols import Stoppable
from bluesky.run_engine import RunEngine
from bluesky.utils import maybe_await
from ophyd_async.core import (
    AsyncStatus,
    Device,
    get_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.motor import Motor

from hextools.motors import CameraObjective, FOV_2_4_mm_Camera, HomeStatus
from hextools.photon_delivery_system.dclm import DCLM
from hextools.photon_delivery_system.filters import (
    Filter,
    FilterPosition,
    FilterSetting,
)
from hextools.photon_delivery_system.slits import Slits


def _slits() -> tuple[Device, list[Motor]]:
    with init_devices(mock=True):
        slits = Slits("TEST:SLT:", 1)  # type: ignore
    return slits, [slits.inboard, slits.outboard, slits.bottom, slits.top]


def _filter() -> tuple[Device, list[Motor]]:
    positions = {FilterPosition.PASS_THROUGH: FilterSetting(0.0, "pass through")}
    with init_devices(mock=True):
        filt = Filter("TEST:FLT:MTR", positions, "TEST:FLT:IN_POS", name="filt")
    return filt, [filt.filter_motor]


def _dclm() -> tuple[Device, list[Motor]]:
    with init_devices(mock=True, child_name_separator="_"):
        dclm = DCLM("TEST:DCLM{")
    return dclm, [dclm.xtal1_vertical_trans, dclm.cooled_beam_stop]


def _mock_camera() -> FOV_2_4_mm_Camera:
    with init_devices(mock=True):
        camera = FOV_2_4_mm_Camera("TEST:CAM:")
    return camera


def _camera() -> tuple[Device, list[Motor]]:
    camera = _mock_camera()
    return camera, [camera.obj_selector]


@pytest.mark.parametrize("make", [_slits, _filter, _dclm, _camera])
async def test_stop_stops_the_motors_set_moves(
    RE: RunEngine, make: Callable[[], tuple[Device, list[Motor]]]
):
    device, motors = make()
    # The RunEngine stops only what it moved, and only if it is Stoppable.
    assert isinstance(device, Stoppable)
    for motor in motors:
        set_mock_value(motor.motor_done_move, 0)

    await maybe_await(device.stop(success=False))

    for motor in motors:
        get_mock_put(motor.motor_stop).assert_called_once()


async def _start_objective_change(camera: FOV_2_4_mm_Camera) -> AsyncStatus:
    set_mock_value(camera._obj_selector_home_sts, HomeStatus.HOMED)
    status = camera.set(CameraObjective.LEFT_4MM)
    await asyncio.sleep(0.01)
    set_mock_value(camera.obj_selector.motor_done_move, 0)
    await asyncio.sleep(0.01)
    return status


# Copilot and Jakub on #118: set() waited on the objective readback with no
# timeout, so a selector stopped between objectives left its status pending.
async def test_stop_during_an_objective_change_fails_its_status(RE: RunEngine):
    camera = _mock_camera()
    status = await _start_objective_change(camera)

    await camera.stop()
    set_mock_value(camera.obj_selector.motor_done_move, 1)

    with pytest.raises(RuntimeError, match="stopped"):
        await asyncio.wait_for(status, timeout=5)


async def test_selector_stopping_short_of_the_objective_fails_the_set(RE: RunEngine):
    camera = _mock_camera()
    camera._arrival_grace = 0.1
    status = await _start_objective_change(camera)

    set_mock_value(camera.obj_selector.motor_done_move, 1)

    with pytest.raises(RuntimeError, match="stopped"):
        await asyncio.wait_for(status, timeout=5)


async def test_objective_reported_just_after_the_move_ends_completes_the_set(
    RE: RunEngine,
):
    camera = _mock_camera()
    status = await _start_objective_change(camera)

    set_mock_value(camera.obj_selector.motor_done_move, 1)
    await asyncio.sleep(0.05)
    set_mock_value(camera._at_left_objective, True)

    await asyncio.wait_for(status, timeout=5)
    assert status.success
