import bluesky.plan_stubs as bps
import pytest
from bluesky.run_engine import RunEngine
from ophyd_async.core import callback_on_mock_put, init_devices, set_mock_value

from hextools.photon_delivery_system.slits import Slits


@pytest.fixture
async def slits() -> Slits:
    async with init_devices(mock=True):
        device = Slits("XF:TEST:", 1)
    # Mirror each motor setpoint to its readback so moves complete in mock mode.
    for motor in (device.inboard, device.outboard, device.bottom, device.top):
        callback_on_mock_put(
            motor.user_setpoint,
            lambda value, motor=motor, **_: set_mock_value(motor.user_readback, value),
        )
    return device


# axis label, low motor, high motor, gap signal, center signal
_AXES = [
    ("horizontal", "inboard", "outboard", "horizontal_gap", "horizontal_center"),
    ("vertical", "bottom", "top", "vertical_gap", "vertical_center"),
]


@pytest.mark.parametrize("axis, low, high, gap, center", _AXES)
async def test_slits_gap_and_center_readback(
    slits: Slits, axis: str, low: str, high: str, gap: str, center: str
):
    set_mock_value(getattr(slits, low).user_readback, -2.0)
    set_mock_value(getattr(slits, high).user_readback, 4.0)

    assert await getattr(slits, gap).get_value() == pytest.approx(6.0)
    assert await getattr(slits, center).get_value() == pytest.approx(1.0)


@pytest.mark.parametrize("axis, low, high, gap, center", _AXES)
async def test_slits_set_gap_keeps_center(
    slits: Slits, axis: str, low: str, high: str, gap: str, center: str
):
    low_motor, high_motor = getattr(slits, low), getattr(slits, high)
    set_mock_value(low_motor.user_readback, -1.0)
    set_mock_value(high_motor.user_readback, 3.0)  # center 1.0, gap 4.0

    await getattr(slits, gap).set(10.0)

    # Blades move symmetrically about the unchanged center.
    assert await low_motor.user_readback.get_value() == pytest.approx(-4.0)
    assert await high_motor.user_readback.get_value() == pytest.approx(6.0)
    assert await getattr(slits, center).get_value() == pytest.approx(1.0)
    assert await getattr(slits, gap).get_value() == pytest.approx(10.0)


@pytest.mark.parametrize("axis, low, high, gap, center", _AXES)
async def test_slits_set_center_keeps_gap(
    slits: Slits, axis: str, low: str, high: str, gap: str, center: str
):
    low_motor, high_motor = getattr(slits, low), getattr(slits, high)
    set_mock_value(low_motor.user_readback, -1.0)
    set_mock_value(high_motor.user_readback, 3.0)  # center 1.0, gap 4.0

    await getattr(slits, center).set(5.0)

    # Both blades shift by the same amount, preserving the gap.
    assert await low_motor.user_readback.get_value() == pytest.approx(3.0)
    assert await high_motor.user_readback.get_value() == pytest.approx(7.0)
    assert await getattr(slits, gap).get_value() == pytest.approx(4.0)
    assert await getattr(slits, center).get_value() == pytest.approx(5.0)


async def test_slits_set_all_gaps_and_centers_via_mv(RE: RunEngine):
    # Build under the RE loop so bps.mv can drive the device.
    with init_devices(mock=True):
        slits = Slits("XF:TEST:", 1)
    for motor in (slits.inboard, slits.outboard, slits.bottom, slits.top):
        callback_on_mock_put(
            motor.user_setpoint,
            lambda value, motor=motor, **_: set_mock_value(motor.user_readback, value),
        )
        set_mock_value(motor.user_readback, 0.0)

    # Flat (h_gap, h_center, v_gap, v_center).
    RE(bps.mv(slits, (10, 10, 4, 2)))

    # Horizontal: gap 10, center 10 -> inboard 5, outboard 15.
    assert await slits.inboard.user_readback.get_value() == pytest.approx(5.0)
    assert await slits.outboard.user_readback.get_value() == pytest.approx(15.0)
    # Vertical: gap 4, center 2 -> bottom 0, top 4.
    assert await slits.bottom.user_readback.get_value() == pytest.approx(0.0)
    assert await slits.top.user_readback.get_value() == pytest.approx(4.0)

    assert await slits.horizontal_gap.get_value() == pytest.approx(10.0)
    assert await slits.horizontal_center.get_value() == pytest.approx(10.0)
    assert await slits.vertical_gap.get_value() == pytest.approx(4.0)
    assert await slits.vertical_center.get_value() == pytest.approx(2.0)
