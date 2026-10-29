from typing import cast

import pytest
from bluesky.protocols import Stoppable
from bluesky.run_engine import RunEngine
from ophyd_async.core import (
    callback_on_mock_put,
    get_mock_put,
    init_devices,
    set_mock_value,
)

from hextools.photon_delivery_system.slits import (
    FrontEndSlit,
    SlitAperture,
    set_fe_aperture,
)


def _re_slit() -> FrontEndSlit:
    """Mock slit built under the RE loop; moves complete instantly."""
    with init_devices(mock=True):
        fe_slit1 = FrontEndSlit("FE:TEST{Slt:1-Ax:Y}", name="fe-slit1")
    callback_on_mock_put(
        fe_slit1.y.user_setpoint,
        lambda value, **_: set_mock_value(fe_slit1.y.user_readback, value),
    )
    return fe_slit1


def test_aperture_positions_match_the_hex_table(RE: RunEngine):
    assert {a.value: y for a, y in _re_slit()._settings.items()} == {
        "Fully open (24 x 4.55 mm)": -30.0,
        "Large (7.13 x 3.56 mm)": -10.0,
        "Fully closed": 0.0,
        "Small (3.56 x 3.56 mm)": 10.0,
        "Pinhole (0.89 mm)": 30.0,
    }


@pytest.mark.parametrize(
    "y, expected",
    [
        (-30.0, "Fully open (24 x 4.55 mm)"),
        (-10.02, "Large (7.13 x 3.56 mm)"),
        (0.0, "Fully closed"),
        (30.0, "Pinhole (0.89 mm)"),
        (4.0, "Fully closed"),
        (-30.67, "Fully open (24 x 4.55 mm)"),
    ],
)
async def test_aperture_reads_closest_to_y(RE: RunEngine, y: float, expected: str):
    fe_slit1 = _re_slit()
    set_mock_value(fe_slit1.y.user_readback, y)
    assert await fe_slit1.aperture.get_value() == expected


@pytest.mark.parametrize("aperture", list(SlitAperture))
async def test_set_fe_aperture_moves_y(RE: RunEngine, aperture: SlitAperture):
    fe_slit1 = _re_slit()

    # The GUI submits the dropdown's string value.
    RE(set_fe_aperture(cast(SlitAperture, aperture.value), fe_slit1=fe_slit1))

    assert await fe_slit1.y.user_readback.get_value() == pytest.approx(
        fe_slit1._settings[aperture]
    )
    assert await fe_slit1.aperture.get_value() == aperture.value


def test_set_fe_aperture_refuses_unknown_aperture(RE: RunEngine):
    with pytest.raises(ValueError):
        RE(set_fe_aperture(cast(SlitAperture, "2 x 2 mm"), fe_slit1=_re_slit()))


async def test_stop_stops_y_while_moving(RE: RunEngine):
    fe_slit1 = _re_slit()
    # The RunEngine stops only what it moved, and only if it is Stoppable.
    assert isinstance(fe_slit1, Stoppable)
    set_mock_value(fe_slit1.y.motor_done_move, 0)

    await fe_slit1.stop(success=False)

    get_mock_put(fe_slit1.y.motor_stop).assert_called_once()
