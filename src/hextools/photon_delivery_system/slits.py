"""Slit device for the photon delivery system."""

import asyncio
from functools import partial

from ophyd_async.core import (
    AsyncMovable,
    AsyncStatus,
    StandardReadable,
    derived_signal_rw,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor


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
                low=self.inboard.user_readback,
                high=self.outboard.user_readback,
            )
            self.horizontal_center = derived_signal_rw(
                self._get_center,
                partial(self._set_center, self.inboard, self.outboard),
                derived_units="mm",
                derived_precision=3,
                low=self.inboard.user_readback,
                high=self.outboard.user_readback,
            )
            self.vertical_gap = derived_signal_rw(
                self._get_gap,
                partial(self._set_gap, self.bottom, self.top),
                derived_units="mm",
                derived_precision=3,
                low=self.bottom.user_readback,
                high=self.top.user_readback,
            )
            self.vertical_center = derived_signal_rw(
                self._get_center,
                partial(self._set_center, self.bottom, self.top),
                derived_units="mm",
                derived_precision=3,
                low=self.bottom.user_readback,
                high=self.top.user_readback,
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

    @AsyncStatus.wrap
    async def _set_gap(
        self, low_motor: AsyncEpicsMotor, high_motor: AsyncEpicsMotor, gap: float
    ) -> None:
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
        await asyncio.gather(
            low_motor.set(center - gap / 2),
            high_motor.set(center + gap / 2),
        )

    @AsyncStatus.wrap
    async def _set_center(
        self, low_motor: AsyncEpicsMotor, high_motor: AsyncEpicsMotor, center: float
    ) -> None:
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
        await asyncio.gather(
            low_motor.set(low + shift),
            high_motor.set(high + shift),
        )

    @AsyncStatus.wrap
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
        await asyncio.gather(
            self.horizontal_gap.set(h_gap),
            self.vertical_gap.set(v_gap),
        )

        await asyncio.gather(
            self.horizontal_center.set(h_center),
            self.vertical_center.set(v_center),
        )
