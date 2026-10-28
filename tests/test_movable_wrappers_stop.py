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
    callback_on_mock_put,
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


def _mock_slits() -> Slits:
    with init_devices(mock=True):
        slits = Slits("TEST:SLT:", 1)  # type: ignore
    return slits


def _slits() -> tuple[Device, list[Motor]]:
    slits = _mock_slits()
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


async def test_stop_in_the_gap_phase_fails_the_slits_move_before_the_centers(
    RE: RunEngine,
):
    slits = _mock_slits()
    blades = [slits.inboard, slits.outboard, slits.bottom, slits.top]
    for blade in blades:
        set_mock_value(blade.velocity, 1.0)
        set_mock_value(blade.acceleration_time, 0.0)
    status = slits.set((4.0, 1.0, 4.0, 1.0))
    await asyncio.sleep(0.2)

    await slits.stop()

    with pytest.raises(RuntimeError, match="stopped"):
        await asyncio.wait_for(status, timeout=5)
    for blade in blades:
        assert get_mock_put(blade.user_setpoint).call_count == 1


def _move_on_objective_put(camera: FOV_2_4_mm_Camera, arrive: bool) -> None:
    def move(value, **_):
        set_mock_value(camera.obj_selector.motor_done_move, 0)
        set_mock_value(camera._at_left_objective, arrive)
        set_mock_value(camera.obj_selector.motor_done_move, 1)

    callback_on_mock_put(camera.objective, move)


# Copilot on #118: a move that ends before the objective put completes was
# never seen moving, so set() waited forever.
@pytest.mark.parametrize("arrive", [True, False])
async def test_objective_move_that_ends_before_its_put_completes(
    RE: RunEngine, arrive: bool
):
    camera = _mock_camera()
    camera._arrival_grace = 0.1
    set_mock_value(camera._obj_selector_home_sts, HomeStatus.HOMED)
    _move_on_objective_put(camera, arrive)

    status = camera.set(CameraObjective.LEFT_4MM)

    if arrive:
        await asyncio.wait_for(status, timeout=5)
    else:
        with pytest.raises(RuntimeError, match="stopped"):
            await asyncio.wait_for(status, timeout=5)


async def test_objective_change_that_never_starts_moving_fails(RE: RunEngine):
    camera = _mock_camera()
    camera._start_timeout = 0.1
    set_mock_value(camera._obj_selector_home_sts, HomeStatus.HOMED)

    with pytest.raises(RuntimeError, match="stopped"):
        await asyncio.wait_for(camera.set(CameraObjective.LEFT_4MM), timeout=5)


# Copilot on #118 (f07a001): a stop() that came before the objective put did not
# prevent it, so the selector could start moving after being stopped.
@pytest.mark.parametrize("ticks_before_stop", [0, 1, 2, 3])
async def test_no_objective_command_is_sent_after_stop(
    RE: RunEngine, ticks_before_stop: int
):
    camera = _mock_camera()
    set_mock_value(camera._obj_selector_home_sts, HomeStatus.HOMED)
    set_mock_value(camera.obj_selector.motor_done_move, 1)
    command = get_mock_put(camera.objective)
    motor_stop = get_mock_put(camera.obj_selector.motor_stop)

    status = camera.set(CameraObjective.LEFT_4MM)
    for _ in range(ticks_before_stop):
        await asyncio.sleep(0)
    commands_before_stop = command.call_count
    await camera.stop()

    with pytest.raises(RuntimeError, match="stopped"):
        await asyncio.wait_for(status, timeout=5)
    if command.call_count > commands_before_stop:
        # A command already in flight is followed by a motor stop.
        motor_stop.assert_called_once()
    else:
        assert command.call_count == commands_before_stop
