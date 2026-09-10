"""Double crystal Laue monochromator (DCLM) device and energy change plan."""

import asyncio
from typing import Final

import numpy as np
from bluesky import plan_stubs as bps
from bluesky import plans as bp
from bluesky.callbacks.fitting import PeakStats
from bluesky.preprocessors import finalize_wrapper, subs_decorator
from ophyd_async.core import (
    AsyncMovable,
    AsyncStatus,
    StandardReadable,
    StrictEnum,
    derived_signal_r,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.adcore import AreaDetector, NDStatsIO
from ophyd_async.epics.core import EpicsDevice
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor

from ..utils import get_obj_from_ipython_ns
from .shutter import Shutter


class BeamMode(StrictEnum):
    """Beam modes."""

    MONOCHROMATIC = "Monochromatic"
    WHITE = "White"


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
    beam_stop_in : float
        Position of the beam stop when it is in the beam path.
    beam_stop_out : float
        Position of the beam stop when it is out of the beam path.
    crystal_1_in : float
        Position of the first crystal when it is in the beam path.
    crystal_1_out : float
        Position of the first crystal when it is out of the beam path.
    fluo_y_direct : float
        Position of the fluorescence screen when it is in the direct beam path.
    fluo_y_direct_out : float
        Position of the fluorescence screen when it is out of the beam path.

    Methods
    -------
    set(value: BeamMode)
        Set the beam mode to either white or monochromatic.
    """

    # Constants for in/out positions of the monochromator components
    beam_stop_in: Final[float] = 0.0
    beam_stop_out: Final[float] = 48.3
    xtal1_in: Final[float] = 0.0
    xtal1_out: Final[float] = -36.0
    fs_in: Final[float] = 25.0
    fs_out: Final[float] = 46.0

    # Constants for the monochromator geometry
    bragg_factor: Final[float] = 1.977  # Energy in keV for Si(111) at 2d = 6.271 Å
    bragg_angle_offset: Final[float] = (
        35.2544  # Bragg angle offset in degrees for Si(111)
    )
    fixed_beam_offset: Final[float] = (
        25.0  # Crystal 2 z offset in mm for fixed beam offset
    )
    fs_distance: Final[float] = 1428.0  # Distance to fluorescence screen in mm

    def __init__(self, prefix: str, name: str = ""):

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
        xtal1_out = abs(xtal1_pos - self.xtal1_out) < 0.01
        beam_stop_out = abs(beam_stop_pos - self.beam_stop_out) < 0.01
        if xtal1_out and beam_stop_out:
            return BeamMode.WHITE
        return BeamMode.MONOCHROMATIC

    def _get_energy(self, pitch_angle: float) -> float:
        """Compute the energy of the monochromatic beam based on the pitch angle."""
        bragg_angle = np.deg2rad(self.bragg_angle_offset - pitch_angle)
        return self.bragg_factor / np.sin(bragg_angle)

    @AsyncStatus.wrap
    async def set(self, value: BeamMode):
        if value == BeamMode.WHITE:
            xtal1_target, bs_target = self.xtal1_out, self.beam_stop_out
        else:
            xtal1_target, bs_target = self.xtal1_in, self.beam_stop_in
        coros = (
            self.xtal1_vertical_trans.set(xtal1_target),
            self.cooled_beam_stop.set(bs_target),
        )
        await asyncio.gather(*coros)


def change_energy(
    energy: float,
    dclm: DCLM | None = None,
    fs_camera: AreaDetector | None = None,
    coarse_angle_range: float = 0.1,
    coarse_num_steps: int = 41,
    fine_angle_range: float = 0.025,
    fine_num_steps: int = 26,
    fs_stats_plugin_name: str = "stats1",
    photon_shutter: Shutter | None = None,
):
    """Bluesky plan to change monochromator energy for Si(111).

    Computes Bragg geometry and moves all crystal and beam stop motors.
    If an area detector is provided, performs coarse and fine pitch scans
    to auto-tune the crystal 2 pitch to the fluorescence peak.

    Parameters
    ----------
    energy : float
        Target energy in keV.
    dclm : DCLM, optional
        Must be in monochromatic mode. If None, the function will attempt to
        retrieve it from the IPython namespace.
    fs_camera : AreaDetector, optional
        Fluorescence screen camera for auto-tuning. If None, motors are
        moved without feedback.
    coarse_angle_range : float
        Half-width of the coarse pitch scan in degrees.
    coarse_num_steps : int
        Number of points in the coarse scan.
    fine_angle_range : float
        Half-width of the fine pitch scan in degrees.
    fine_num_steps : int
        Number of points in the fine scan.
    photon_shutter : Shutter, optional
        Shutter to close on exit. Falls back to the ``photon_shutter`` in the
        IPython namespace when not provided.

    Raises
    ------
    RuntimeError
        If the monochromator is not in monochromatic mode.
    """
    if dclm is None:
        dclm = get_obj_from_ipython_ns("dclm", DCLM)
        if dclm is None:
            raise RuntimeError(
                "No DCLM provided and no valid 'dclm' var found in the IPython ns."
            )

    if photon_shutter is None:
        photon_shutter = get_obj_from_ipython_ns("photon_shutter", Shutter)
        if photon_shutter is None and fs_camera is not None:
            raise RuntimeError(
                "No photon shutter provided and no valid 'photon_shutter' var found "
                "in the IPython ns. Photon shutter is required for auto-tuning with "
                "a camera."
            )

    def _reset_and_close():
        # Reset the fluorescence screen and close the shutter on the way out.
        yield from bps.mv(dclm.flourescence_screen, dclm.fs_out)
        if photon_shutter is not None:
            yield from bps.mv(photon_shutter, False)  # Close the photon shutter on exit

    def _inner_change_energy():
        mode = yield from bps.rd(dclm.beam_mode)
        if mode != BeamMode.MONOCHROMATIC:
            raise RuntimeError("Monochromator is not in monochromatic mode.")

        if not np.isfinite(energy) or energy <= dclm.bragg_factor:
            raise ValueError(
                f"Energy must be finite and greater than {dclm.bragg_factor} keV."
            )
        bragg_angle = np.arcsin(
            dclm.bragg_factor / energy
        )  # Si(111), 2d = 6.271 Å, energy in keV
        angle = dclm.bragg_angle_offset - np.rad2deg(
            bragg_angle
        )  # Bragg angle to motor coordinate
        z = dclm.fixed_beam_offset / np.tan(
            2 * bragg_angle
        )  # Crystal 2 z for fixed beam offset of 25 mm
        if fs_camera is not None:
            fluo_y = dclm.fs_distance * np.tan(
                2 * bragg_angle
            )  # Fluorescence screen y at 1428 mm downstream
        else:
            fluo_y = dclm.fs_in

        # fmt: off
        yield from bps.mv(
            dclm.xtal2_z, z,
            dclm.xtal2_pitch, angle,
            dclm.xtal1_pitch, angle,
            dclm.xtal1_vertical_trans, dclm.xtal1_in,
            dclm.cooled_beam_stop, dclm.beam_stop_in,
            dclm.flourescence_screen, fluo_y,
        )
        # fmt: on

        # No feedback available without a camera, so just move the motors and return.
        if fs_camera is None or photon_shutter is None:
            return

        fs_stats = fs_camera.get_plugin_by_name(fs_stats_plugin_name, NDStatsIO)
        if fs_stats is None:
            raise RuntimeError(
                "Fluorescence screen camera does not have a "
                f"'{fs_stats_plugin_name}' plugin configured!"
            )

        photon_shutter_sts = yield from bps.rd(photon_shutter.status)
        if not photon_shutter_sts:
            yield from bps.mv(photon_shutter, True)

        # Create a PeakStats object to monitor the fluorescence screen camera signal
        # and find the position of the crystal 2 pitch that produces a peak.
        ps = PeakStats(
            dclm.xtal2_pitch.name,
            fs_camera.get_plugin_by_name(fs_stats_plugin_name, NDStatsIO).total.name,
        )

        # Perform a scan around the current position of the crystal 2 pitch,
        # and feed the produced events into the PeakStats object to find the peak
        # position.
        @subs_decorator(ps)
        def auto_tune(angle_range: float, num_steps: int):
            yield from bp.scan(
                [fs_camera],
                dclm.xtal2_pitch,
                angle - angle_range,
                angle + angle_range,
                num_steps,
                md={"plan_name": "change_energy_auto_tune"},
            )

        yield from auto_tune(coarse_angle_range, coarse_num_steps)  # Coarse scan

        peak: float | None = ps.com  # ty: ignore[unresolved-attribute]
        if peak is None:
            raise RuntimeError(
                "No peak found in coarse scan. Check the fluorescence screen."
            )

        # Move to the peak found by the coarse scan
        yield from bps.mv(dclm.xtal2_pitch, peak)

        # Clear coarse scan events so the fine scan computes from its own data only
        ps.reset()

        yield from auto_tune(fine_angle_range, fine_num_steps)  # Fine scan

        peak = ps.com  # ty: ignore[unresolved-attribute]
        if peak is None:
            raise RuntimeError(
                "No peak found in fine scan. Check the fluorescence screen."
            )

        # To minimize issues with backlash, always approach the final position
        # from below.
        yield from bps.mv(dclm.xtal2_pitch, peak - 0.05)
        yield from bps.mv(dclm.xtal2_pitch, peak)

    yield from finalize_wrapper(
        _inner_change_energy(),
        _reset_and_close(),
    )
