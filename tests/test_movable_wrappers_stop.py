"""Devices whose set() moves motors stop those motors when the RunEngine stops them."""

from collections.abc import Callable

import pytest
from bluesky.protocols import Stoppable
from bluesky.run_engine import RunEngine
from bluesky.utils import maybe_await
from ophyd_async.core import Device, get_mock_put, init_devices, set_mock_value
from ophyd_async.epics.motor import Motor

from hextools.motors import FOV_2_4_mm_Camera
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


def _camera() -> tuple[Device, list[Motor]]:
    with init_devices(mock=True):
        camera = FOV_2_4_mm_Camera("TEST:CAM:")
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
