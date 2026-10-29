"""Slit device for the photon delivery system."""

import asyncio
from enum import StrEnum
from functools import partial

import bluesky.plan_stubs as bps
from ophyd_async.core import (
    AsyncMovable,
    StandardReadable,
    WatchableAsyncStatus,
    derived_signal_r,
    derived_signal_rw,
    forward_watcher_updates,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor

from hextools.bl_constants import fetch_beamline_constants
from hextools.utils import ensure_available


class SlitAperture(StrEnum):
    """Apertures of front-end Slit1; the values are what the GUI offers."""

    FULLY_OPEN = "Fully open (24 x 4.55 mm)"
    LARGE = "Large (7.13 x 3.56 mm)"
    FULLY_CLOSED = "Fully closed"
    SMALL = "Small (3.56 x 3.56 mm)"
    PINHOLE = "Pinhole (0.89 mm)"


class FrontEndSlit(StandardReadable, AsyncMovable[SlitAperture]):
    """A front-end slit whose aperture is chosen by moving its Y axis.

    Parameters
    ----------
    prefix : str
        Axis PV prefix, e.g. ``"FE:C27A-OP{Slt:1-Ax:Y}"``; the motor record is
        ``<prefix>Mtr``.
    name : str
        Device name.
    """

    def __init__(self, prefix: str, name: str = ""):
        self._settings = {
            SlitAperture[k.upper()]: v
            for k, v in fetch_beamline_constants("fe_aperture").items()
        }
        with self.add_children_as_readables():
            self.y = AsyncEpicsMotor(prefix + "Mtr")
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.aperture = derived_signal_r(self._aperture, y=self.y.user_readback)
            self.in_position = derived_signal_r(self._in_position, y=self.y.user_readback, aperture=self.aperture)
        super().__init__(name=name)

    def _aperture(self, y: float) -> str:
        return min(self._settings, key=lambda a: abs(self._settings[a] - y)).value

    def _in_position(self, y: float, aperture: str) -> bool:
        return abs(y - self._settings[SlitAperture(aperture)]) < 1e-2

    @WatchableAsyncStatus.wrap
    async def set(self, value: SlitAperture):
        """Move Y to the position of an aperture.

        Parameters
        ----------
        value : SlitAperture
            The aperture, or its value as a string.
        """
        if value not in self._settings:
            raise ValueError(
                f"Aperture setting not found for {value}, "
                f"available settings: {list(self._settings.keys())}"
            )
        move_status = self.y.set(self._settings[value])
        async for update in forward_watcher_updates(
            [move_status], self.name, combine=False
        ):
            yield update

    async def stop(self, success: bool = False):
        """Stop Y, so pausing or aborting the RunEngine stops a move."""
        await self.y.stop()


def set_fe_aperture(aperture: SlitAperture, fe_slit1: FrontEndSlit | None = None):
    """Move front-end slit Y to a named aperture.

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


class Slits(
    StandardReadable,
    EpicsDevice,
    AsyncMovable[
        tuple[tuple[float, float], tuple[float, float]]
        | tuple[float, float, float, float]
    ],
):
    """Generic slits with inboard, outboard, bottom, and top motors.

    Exposes gap and center as read-write derived signals for both the horizontal
    (inboard/outboard) and vertical (bottom/top) axes. Setting a gap keeps the
    center fixed; setting a center keeps the gap fixed.
    """

    def __init__(self, prefix: str, name: str = ""):
        with self.add_children_as_readables(Format.CHILD):
            self.inboard = AsyncEpicsMotor(prefix + "I}Mtr")
            self.outboard = AsyncEpicsMotor(prefix + "O}Mtr")
            self.bottom = AsyncEpicsMotor(prefix + "B}Mtr")
            self.top = AsyncEpicsMotor(prefix + "T}Mtr")

        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.horizontal_gap = derived_signal_rw(
                self._get_gap,
                partial(self._set_gap, self.inboard, self.outboard),
                derived_units="mm",
                derived_precision=3,
                low=self.inboard,
                high=self.outboard,
            )
            self.horizontal_center = derived_signal_rw(
                self._get_center,
                partial(self._set_center, self.inboard, self.outboard),
                derived_units="mm",
                derived_precision=3,
                low=self.inboard,
                high=self.outboard,
            )
            self.vertical_gap = derived_signal_rw(
                self._get_gap,
                partial(self._set_gap, self.bottom, self.top),
                derived_units="mm",
                derived_precision=3,
                low=self.bottom,
                high=self.top,
            )
            self.vertical_center = derived_signal_rw(
                self._get_center,
                partial(self._set_center, self.bottom, self.top),
                derived_units="mm",
                derived_precision=3,
                low=self.bottom,
                high=self.top,
            )

        super().__init__(name=name)

    def _get_gap(self, low: float, high: float) -> float:
        """Gap is the separation between the low and high blades.

        Parameters
        ----------
        low : float
            The position of the low blade (inboard or bottom).
        high : float
            The position of the high blade (outboard or top).

        Returns
        -------
        float
            The gap between the low and high blades.
        """
        return high - low

    def _get_center(self, low: float, high: float) -> float:
        """Center is the midpoint between the low and high blades.

        Parameters
        ----------
        low : float
            The position of the low blade (inboard or bottom).
        high : float
            The position of the high blade (outboard or top).

        Returns
        -------
        float
            The center position between the low and high blades.
        """
        return (low + high) / 2

    @WatchableAsyncStatus.wrap
    async def _set_gap(
        self, low_motor: AsyncEpicsMotor, high_motor: AsyncEpicsMotor, gap: float
    ):
        """Set the gap while holding the center fixed.

        Parameters
        ----------
        low_motor : AsyncEpicsMotor
            The motor controlling the low blade (inboard or bottom).
        high_motor : AsyncEpicsMotor
            The motor controlling the high blade (outboard or top).
        gap : float
            The desired gap between the low and high blades.
        """
        low, high = await asyncio.gather(
            low_motor.user_readback.get_value(),
            high_motor.user_readback.get_value(),
        )
        center = (low + high) / 2
        statuses = [
            low_motor.set(center - gap / 2),
            high_motor.set(center + gap / 2),
        ]
        async for update in forward_watcher_updates(statuses, self.name, combine=False):
            yield update

    @WatchableAsyncStatus.wrap
    async def _set_center(
        self, low_motor: AsyncEpicsMotor, high_motor: AsyncEpicsMotor, center: float
    ):
        """Set the center while holding the gap fixed.

        Parameters
        ----------
        low_motor : AsyncEpicsMotor
            The motor controlling the low blade (inboard or bottom).
        high_motor : AsyncEpicsMotor
            The motor controlling the high blade (outboard or top).
        center : float
            The desired center position for the blades.
        """
        low, high = await asyncio.gather(
            low_motor.user_readback.get_value(),
            high_motor.user_readback.get_value(),
        )
        shift = center - (low + high) / 2
        statuses = [
            low_motor.set(low + shift),
            high_motor.set(high + shift),
        ]
        async for update in forward_watcher_updates(statuses, self.name, combine=False):
            yield update

    @WatchableAsyncStatus.wrap
    async def set(
        self,
        value: tuple[tuple[float, float], tuple[float, float]]
        | tuple[float, float, float, float],
    ):
        """Set the horizontal and vertical gaps and centers.

        Parameters
        ----------
        value : tuple[tuple[float, float], tuple[float, float]]
            ((horizontal_gap, horizontal_center), (vertical_gap, vertical_center))
        """
        if len(value) == 2 and all(len(v) == 2 for v in value):
            (h_gap, h_center), (v_gap, v_center) = value
        elif len(value) == 4:
            h_gap, h_center, v_gap, v_center = value
        else:
            raise ValueError(
                "Slit set value must be ((h_gap, h_center), (v_gap, v_center)) "
                "or (h_gap, h_center, v_gap, v_center)"
            )

        # Set gaps first to preserve centers, then set centers to preserve gaps.
        gaps = [
            self._set_gap(self.inboard, self.outboard, h_gap),
            self._set_gap(self.bottom, self.top, v_gap),
        ]
        async for update in forward_watcher_updates(gaps, self.name, combine=False):
            yield update

        centers = [
            self._set_center(self.inboard, self.outboard, h_center),
            self._set_center(self.bottom, self.top, v_center),
        ]
        async for update in forward_watcher_updates(centers, self.name, combine=False):
            yield update

    async def stop(self, success: bool = False):
        """Stop all four blades, so pausing or aborting the RunEngine stops a move."""
        await asyncio.gather(
            self.inboard.stop(),
            self.outboard.stop(),
            self.bottom.stop(),
            self.top.stop(),
        )


def move_slits(
    slits: Slits,
    horizontal_center: float | None = None,
    vertical_center: float | None = None,
    horizontal_gap: float | None = None,
    vertical_gap: float | None = None,
    relative: bool = False,
):
    """Move a pair of slit blades by their center line and gap.

    Only the values given move; the others stay where they are. A center move
    shifts both blades of the pair together, keeping the gap, and a gap move
    opens or closes both blades about the center line.

    Parameters
    ----------
    slits : Slits
        The slits to move.
    horizontal_center : float, optional
        Center of the inboard/outboard pair, in mm.
    vertical_center : float, optional
        Center of the bottom/top pair, in mm.
    horizontal_gap : float, optional
        Gap between the inboard and outboard blades, in mm.
    vertical_gap : float, optional
        Gap between the bottom and top blades, in mm.
    relative : bool, default False
        Treat the given values as shifts from the current positions.

    Raises
    ------
    ValueError
        If no center or gap is given.
    """
    # A gap and a center on one axis drive the same two blades, so gaps go first
    stages = [
        [(slits.horizontal_gap, horizontal_gap), (slits.vertical_gap, vertical_gap)],
        [
            (slits.horizontal_center, horizontal_center),
            (slits.vertical_center, vertical_center),
        ],
    ]
    if all(value is None for stage in stages for _, value in stage):
        raise ValueError("Give at least one center or gap to move the slits.")
    move = bps.mvr if relative else bps.mv
    for stage in stages:
        args = [
            x for signal, value in stage if value is not None for x in (signal, value)
        ]
        if args:
            yield from move(*args)
