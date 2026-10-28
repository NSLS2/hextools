"""Front-end slit with fixed apertures selected by one axis (27-ID Slit1 Y)."""

from enum import StrEnum

import bluesky.plan_stubs as bps
from ophyd_async.core import (
    AsyncMovable,
    AsyncStatus,
    StandardReadable,
    derived_signal_r,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor

from ..utils import ensure_available


class SlitAperture(StrEnum):
    """Apertures of front-end Slit1; the values are what the GUI offers."""

    FULLY_OPEN = "Fully open (24 x 4.55 mm)"
    LARGE = "7.13 x 3.56 mm"
    FULLY_CLOSED = "Fully closed"
    SMALL = "3.56 x 3.56 mm"
    PINHOLE = "Pinhole 0.89 mm"


# Y positions in mm (HEX, AJ 2026-10-27); the limit switch is at -30.67 mm.
APERTURE_POSITIONS: dict[SlitAperture, float] = {
    SlitAperture.FULLY_OPEN: -30.0,
    SlitAperture.LARGE: -10.0,
    SlitAperture.FULLY_CLOSED: 0.0,
    SlitAperture.SMALL: 10.0,
    SlitAperture.PINHOLE: 30.0,
}

BETWEEN_APERTURES = "between apertures"


class FrontEndSlit(StandardReadable, AsyncMovable[SlitAperture]):
    """A front-end slit whose aperture is chosen by moving its Y axis.

    Parameters
    ----------
    prefix : str
        Axis PV prefix, e.g. ``"FE:C27A-OP{Slt:1-Ax:Y}"``; the motor record is
        ``<prefix>Mtr``.
    tolerance : float
        How far, in mm, Y may be from an aperture's position and still count as it.
    name : str
        Device name.
    """

    def __init__(self, prefix: str, tolerance: float = 0.05, name: str = ""):
        self.tolerance = tolerance
        with self.add_children_as_readables():
            self.y = AsyncEpicsMotor(prefix + "Mtr")
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.aperture = derived_signal_r(self._aperture, y=self.y.user_readback)
        super().__init__(name=name)

    def _aperture(self, y: float) -> str:
        closest = min(APERTURE_POSITIONS, key=lambda a: abs(APERTURE_POSITIONS[a] - y))
        if abs(APERTURE_POSITIONS[closest] - y) <= self.tolerance:
            return closest.value
        return BETWEEN_APERTURES

    @AsyncStatus.wrap
    async def set(self, value: SlitAperture):
        """Move Y to the position of an aperture.

        Parameters
        ----------
        value : SlitAperture
            The aperture, or its value as a string.
        """
        await self.y.set(APERTURE_POSITIONS[SlitAperture(value)])

    async def stop(self, success: bool = False):
        """Stop Y, so pausing or aborting the RunEngine stops a move."""
        await self.y.stop()


def set_fe_slit1_aperture(aperture: SlitAperture, fe_slit1: FrontEndSlit | None = None):
    """Move front-end Slit1 Y to a named aperture.

    Parameters
    ----------
    aperture : SlitAperture
        The aperture: fully open (24 x 4.55 mm), 7.13 x 3.56 mm, fully closed,
        3.56 x 3.56 mm, or the 0.89 mm pinhole.
    fe_slit1 : FrontEndSlit, optional
        The slit. If None, taken from the session namespace.
    """
    fe_slit1 = ensure_available(FrontEndSlit, fe_slit1=fe_slit1)
    yield from bps.mv(fe_slit1, SlitAperture(aperture))
