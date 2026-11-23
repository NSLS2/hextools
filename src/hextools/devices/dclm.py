"""Double crystal Laue monochromator (DCLM) device and energy change plan."""

import asyncio
from dataclasses import dataclass

import numpy as np
from ophyd_async.core import (
    AsyncMovable,
    StandardReadable,
    StrictEnum,
    WatchableAsyncStatus,
    derived_signal_r,
    forward_watcher_updates,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor

from hextools.bl_constants import fetch_beamline_constants


class BeamMode(StrictEnum):
    """Beam modes."""

    MONOCHROMATIC = "Monochromatic"
    WHITE = "White"


@dataclass(frozen=True)
class DCLMConstants:
    """Monochromator positions and geometry, from ``bl_constants.yml``."""

    beam_stop_in: float
    beam_stop_out: float
    xtal1_in: float
    xtal1_out: float
    fs_in: float
    fs_out: float
    bragg_factor: float  # Energy in keV for Si(111) at 2d = 6.271 Å
    bragg_angle_offset: float  # Bragg angle offset in degrees for Si(111)
    fixed_beam_offset: float  # Crystal 2 z offset in mm for fixed beam offset
    fs_distance: float  # Distance to fluorescence screen in mm


class DCLM(StandardReadable, EpicsDevice, AsyncMovable[BeamMode]):
    """Double crystal Laue monochromator device.

    This class represents a double crystal laue monochromator (DCLM) device,
    which is used to select a specific energy of X-rays from a white beam.
    It provides methods to set the beam mode (white or monochromatic) and to
    change the energy of the monochromatic beam.

    Attributes
    ----------
    xtal1_vertical_trans : AsyncEpicsMotor
        Motor for the vertical translation of the first crystal.
    cooled_beam_stop : AsyncEpicsMotor
        Motor for the cooled beam stop.
    xtal1_pitch : AsyncEpicsMotor
        Motor for the pitch of the first crystal.
    xtal2_pitch : AsyncEpicsMotor
        Motor for the pitch of the second crystal.
    flourescence_screen : AsyncEpicsMotor
        Motor for the fluorescence screen.
    xtal2_z : AsyncEpicsMotor
        Motor for the z-position of the second crystal.
    beam_mode : derived_signal_r
        Derived signal representing the current beam mode (white or monochromatic).
    energy : derived_signal_r
        Derived signal representing the current energy of the monochromatic beam.
    constants : DCLMConstants
        In/out positions and geometry of the monochromator.

    Methods
    -------
    set(value: BeamMode)
        Set the beam mode to either white or monochromatic.
    """

    def __init__(self, prefix: str, name: str = ""):
        self.constants = DCLMConstants(**fetch_beamline_constants("monochromator"))

        self.xtal1_vertical_trans = AsyncEpicsMotor(prefix + "C1Y}Mtr")
        self.cooled_beam_stop = AsyncEpicsMotor(
            prefix + "BS}Mtr",
        )
        self.xtal1_pitch = AsyncEpicsMotor(prefix + "C1P}Mtr")
        self.xtal2_pitch = AsyncEpicsMotor(prefix + "C2P}Mtr")
        self.flourescence_screen = AsyncEpicsMotor(prefix + "FS}Mtr")
        self.xtal2_z = AsyncEpicsMotor(prefix + "Z2}Mtr")
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.beam_mode = derived_signal_r(
                self._get_beam_mode,
                xtal1_pos=self.xtal1_vertical_trans.user_readback,
                beam_stop_pos=self.cooled_beam_stop.user_readback,
            )
            self.energy = derived_signal_r(
                self._get_energy,
                pitch_angle=self.xtal1_pitch.user_readback,
            )
        super().__init__(name=name)

    def _get_beam_mode(self, xtal1_pos: float, beam_stop_pos: float) -> BeamMode:
        """Determine the current beam mode."""
        xtal1_out = abs(xtal1_pos - self.constants.xtal1_out) < 0.01
        beam_stop_out = abs(beam_stop_pos - self.constants.beam_stop_out) < 0.01
        if xtal1_out and beam_stop_out:
            return BeamMode.WHITE
        return BeamMode.MONOCHROMATIC

    def _get_energy(self, pitch_angle: float) -> float:
        """Compute the energy of the monochromatic beam based on the pitch angle."""
        bragg_angle = np.deg2rad(self.constants.bragg_angle_offset - pitch_angle)
        return float(self.constants.bragg_factor / np.sin(bragg_angle))

    @WatchableAsyncStatus.wrap
    async def set(self, value: BeamMode):
        c = self.constants
        if value == BeamMode.WHITE:
            xtal1_target, bs_target = c.xtal1_out, c.beam_stop_out
        else:
            xtal1_target, bs_target = c.xtal1_in, c.beam_stop_in
        statuses = [
            self.xtal1_vertical_trans.set(xtal1_target),
            self.cooled_beam_stop.set(bs_target),
        ]
        async for update in forward_watcher_updates(statuses, self.name, combine=False):
            yield update

    async def stop(self, success: bool = False):
        """Stop the motors a beam-mode change moves."""
        await asyncio.gather(
            self.xtal1_vertical_trans.stop(success=success),
            self.cooled_beam_stop.stop(success=success),
        )


__all__ = ["BeamMode", "DCLM"]
