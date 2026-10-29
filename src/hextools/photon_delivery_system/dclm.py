"""Double crystal Laue monochromator (DCLM) device and energy change plan."""

import asyncio
from dataclasses import dataclass

import numpy as np
from bluesky import plan_stubs as bps
from bluesky import plans as bp
from bluesky import preprocessors as bpp
from bluesky.callbacks.fitting import PeakStats
from bluesky.preprocessors import finalize_wrapper, subs_wrapper
from ophyd_async.core import (
    AsyncMovable,
    StandardReadable,
    StrictEnum,
    WatchableAsyncStatus,
    derived_signal_r,
    forward_watcher_updates,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.adcore import NDStatsIO
from ophyd_async.epics.advimba import VimbaDetector
from ophyd_async.epics.core import EpicsDevice
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor

from hextools.bl_constants import fetch_beamline_constants
from hextools.detectors.utils import enable_preview_mode

from ..utils import ensure_available
from .shutter import Shutter, ensure_shutter_closed, ensure_shutter_open


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
            self.xtal1_vertical_trans.stop(), self.cooled_beam_stop.stop()
        )


def change_beam_mode(
    mode: BeamMode,
    dclm: DCLM | None = None,
    fe_shutter: Shutter | None = None,
):
    """Bluesky plan to change the beam mode of the DCLM.

    Parameters
    ----------
    mode : BeamMode
        The target beam mode for the DCLM
    dclm : DCLM, optional
        The DCLM instance to control. If None, the function will attempt to
        retrieve it from the IPython namespace.
    """
    dclm = ensure_available(DCLM, dclm=dclm)
    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    current_mode = yield from bps.rd(dclm.beam_mode)
    if current_mode == mode:
        return
    if mode == BeamMode.WHITE:
        yield from ensure_shutter_closed(fe_shutter)

    yield from bps.mv(dclm, mode)
    yield from ensure_shutter_open(fe_shutter, allow_actuation=True)


def change_energy(
    energy: float,
    coarse_angle_range: float = 0.1,
    coarse_num_steps: int = 41,
    fine_angle_range: float = 0.025,
    fine_num_steps: int = 26,
    fs_stats_plugin_name: str = "stats1",
    exposure_time: float = 0.5,
    auto_tune: bool = True,
    dclm: DCLM | None = None,
    fs_window_cam: VimbaDetector | None = None,
):
    """Bluesky plan to change monochromator energy for Si(111).

    Computes Bragg geometry and moves all crystal and beam stop motors.
    If an area detector is provided, performs coarse and fine pitch scans
    to auto-tune the crystal 2 pitch to the fluorescence peak.

    Parameters
    ----------
    energy : float
        Target energy in keV.
    auto_tune : bool, optional
        Whether to perform auto-tuning using the fluorescence screen camera.
    coarse_angle_range : float
        Half-width of the coarse pitch auto-tune scan in degrees.
    coarse_num_steps : int
        Number of points in the coarse auto-tune scan.
    fine_angle_range : float
        Half-width of the fine pitch auto-tune scan in degrees.
    fine_num_steps : int
        Number of points in the fine auto-tune scan.
    dclm : DCLM, optional
        Must be in monochromatic mode. If None, the function will attempt to
        retrieve it from the IPython namespace.
    fs_window_cam : VimbaDetector, optional
        Fluorescence screen camera for auto-tuning. If None, with auto_tune
        enabled, it will attempt to be retrieved from the IPython namespace.
        Otherwise, motors are moved without feedback.
    exposure_time : float, optional
        Exposure time for the fluorescence screen camera during auto-tuning.

    Raises
    ------
    RuntimeError
        If the monochromator is not in monochromatic mode.
    """
    # If not provided
    dclm = ensure_available(DCLM, monochromator=dclm)
    if auto_tune:
        fs_window_cam = ensure_available(VimbaDetector, fs_window_cam=fs_window_cam)

    @bpp.reset_positions_decorator(
        [fs_window_cam.driver.acquire_time, fs_window_cam.driver.acquire_period]
        if fs_window_cam is not None
        else []
    )
    def _body():
        mode = yield from bps.rd(dclm.beam_mode)
        if mode != BeamMode.MONOCHROMATIC:
            raise RuntimeError("Monochromator is not in monochromatic mode.")

        c = dclm.constants
        if not np.isfinite(energy) or energy <= c.bragg_factor:
            raise ValueError(
                f"Energy must be finite and greater than {c.bragg_factor} keV."
            )
        bragg_angle = np.arcsin(c.bragg_factor / energy)
        angle = c.bragg_angle_offset - np.rad2deg(bragg_angle)
        z = c.fixed_beam_offset / np.tan(2 * bragg_angle)
        if not auto_tune:
            fluo_y = c.fs_distance * np.tan(2 * bragg_angle)
        else:
            fluo_y = c.fs_in

        # fmt: off
        yield from bps.mv(
            dclm.xtal2_z, z,
            dclm.xtal2_pitch, angle,
            dclm.xtal1_pitch, angle,
            dclm.xtal1_vertical_trans, c.xtal1_in,
            dclm.cooled_beam_stop, c.beam_stop_in,
            dclm.flourescence_screen, fluo_y,
        )
        # fmt: on

        # No feedback available without a camera, so just move the motors and return.
        if fs_window_cam is None or not auto_tune:
            return

        fs_stats = fs_window_cam.get_plugin(fs_stats_plugin_name, NDStatsIO)
        if fs_stats is None:
            raise RuntimeError(
                "Fluorescence screen camera does not have a "
                f"'{fs_stats_plugin_name}' plugin configured!"
            )

        # set camera exposure time
        yield from bps.mv(fs_window_cam.driver.acquire_time, exposure_time)

        # Scan the crystal 2 pitch and return the center of the fluorescence peak.
        # A fresh PeakStats per scan, so each fit sees only its own scan's data.
        def energy_auto_tune(scan_name: str, angle_range: float, num_steps: int):
            ps = PeakStats(dclm.xtal2_pitch.name, fs_stats.mean_value.name)
            yield from subs_wrapper(
                bp.scan(
                    [fs_window_cam],  # type: ignore  # TODO: allow AsyncReadable
                    dclm.xtal2_pitch,
                    angle - angle_range,
                    angle + angle_range,
                    num_steps,
                    md={"plan_name": "energy_auto_tune"},
                ),
                [ps],
            )
            if ps.cen is None:  # type: ignore
                raise RuntimeError(
                    f"No peak found in {scan_name} scan. Check the fluorescence screen."
                )
            return float(ps.cen)  # type: ignore

        coarse_center = yield from energy_auto_tune(
            "coarse", coarse_angle_range, coarse_num_steps
        )
        yield from bps.mv(dclm.xtal2_pitch, coarse_center)

        fine_center = yield from energy_auto_tune(
            "fine", fine_angle_range, fine_num_steps
        )

        # To minimize issues with backlash, always approach the final position
        # from below.
        yield from bps.mv(dclm.xtal2_pitch, fine_center - 0.05)
        yield from bps.mv(dclm.xtal2_pitch, fine_center)

    def _cleanup():
        # Reset the fluorescence screen and close the shutter on the way out.
        yield from bps.mv(dclm.flourescence_screen, dclm.constants.fs_out)
        if fs_window_cam is not None:
            yield from enable_preview_mode(fs_window_cam)

    yield from (
        finalize_wrapper(
            _body(),
            _cleanup(),
        )
    )
