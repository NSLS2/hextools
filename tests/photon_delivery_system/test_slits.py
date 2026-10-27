import bluesky.plan_stubs as bps
import pytest
from bluesky.run_engine import RunEngine
from ophyd_async.core import callback_on_mock_put, init_devices, set_mock_value

from hextools.photon_delivery_system.slits import Slits, move_slits


@pytest.fixture
async def slits() -> Slits:
    async with init_devices(mock=True):
        device = Slits("XF:TEST:", 1)  # type: ignore
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
        slits = Slits("XF:TEST:", 1)  # type: ignore
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


async def test_slits_set_reports_blade_progress_to_watchers(slits: Slits):
    for motor in (slits.inboard, slits.outboard, slits.bottom, slits.top):
        set_mock_value(motor.user_readback, 0.0)
    updates = []

    status = slits.set((10, 10, 4, 2))
    status.watch(lambda **kw: updates.append(kw["name"]))  # type: ignore
    await status

    assert {
        slits.inboard.name,
        slits.outboard.name,
        slits.bottom.name,
        slits.top.name,
    } <= set(updates)


def _re_slits(blades: tuple[float, float, float, float]) -> Slits:
    """Mock slits under the RE loop; blades at (inboard, outboard, bottom, top)."""
    with init_devices(mock=True):
        slits = Slits("XF:TEST:", 1)  # type: ignore
    for motor, value in zip(_motors(slits), blades, strict=True):
        callback_on_mock_put(
            motor.user_setpoint,
            lambda value, motor=motor, **_: set_mock_value(motor.user_readback, value),
        )
        set_mock_value(motor.user_readback, value)
    return slits


def _motors(slits: Slits):
    return (slits.inboard, slits.outboard, slits.bottom, slits.top)


async def _blades(slits: Slits) -> list[float]:
    return [await m.user_readback.get_value() for m in _motors(slits)]


async def test_move_slits_moves_only_the_given_values(RE: RunEngine):
    slits = _re_slits((-1.0, 3.0, -2.0, 2.0))

    RE(move_slits(slits, horizontal_center=2.0))

    # Both horizontal blades shift by +1 (gap 4 kept); the vertical pair stays put.
    assert await _blades(slits) == pytest.approx([0.0, 4.0, -2.0, 2.0])


async def test_move_slits_sets_gap_before_center(RE: RunEngine):
    slits = _re_slits((-1.0, 1.0, 0.0, 0.0))

    RE(move_slits(slits, horizontal_gap=4.0, horizontal_center=5.0))

    assert (await _blades(slits))[:2] == pytest.approx([3.0, 7.0])


async def test_move_slits_relative_shifts_from_current(RE: RunEngine):
    slits = _re_slits((-1.0, 3.0, -2.0, 2.0))

    RE(move_slits(slits, vertical_center=0.5, horizontal_gap=-1.0, relative=True))

    # Horizontal gap 4 -> 3 about center 1; vertical pair shifted up by 0.5.
    assert await _blades(slits) == pytest.approx([-0.5, 2.5, -1.5, 2.5])


def test_move_slits_needs_a_value(RE: RunEngine):
    slits = _re_slits((0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="at least one"):
        RE(move_slits(slits))
