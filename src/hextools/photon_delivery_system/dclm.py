"""Double crystal Laue monochromator (DCLM) device and energy change plan."""

import asyncio
from collections.abc import Callable
from functools import partial
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from bluesky import plan_stubs as bps
from bluesky import plans as bp
from bluesky import preprocessors as bpp
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.callbacks.fitting import PeakStats
from bluesky.callbacks.mpl_plotting import QtAwareCallback, plot_peak_stats
from bluesky.preprocessors import finalize_wrapper, subs_wrapper
from IPython.core.getipython import get_ipython
from matplotlib.figure import Figure
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

from hextools.detectors.utils import enable_preview_mode

from ..utils import ensure_available
from .shutter import Shutter, ensure_shutter_closed, ensure_shutter_open


class BeamMode(StrictEnum):
    """Beam modes."""

    MONOCHROMATIC = "Monochromatic"
    WHITE = "White"


def _figure_factory() -> Callable[[str], Figure]:
    """Return the session BestEffortCallback's figure factory, which the GUI embeds."""
    ipython = get_ipython()
    if ipython is not None:
        for obj in ipython.user_ns.values():
            if isinstance(obj, BestEffortCallback):
                return obj._fig_factory  # noqa: SLF001
    return partial(plt.figure, layout="constrained")


class _PeakStatsPlot(QtAwareCallback):
    """Plot a PeakStats' results when its run stops, on the Qt thread if any."""

    def __init__(
        self, peak_stats: PeakStats, title: str, fig_factory: Callable[[str], Figure]
    ):
        super().__init__()
        self._peak_stats = peak_stats
        self._title = title
        self._fig_factory = fig_factory
        self._scan_id = None

    def start(self, doc):
        self._scan_id = doc.get("scan_id")

    def stop(self, doc):
        if getattr(self._peak_stats, "x_data", None) is None:
            return
        title = f"{self._title} (scan {self._scan_id})"
        figure = self._fig_factory(title)
        ax = figure.add_subplot()
        plot_peak_stats(self._peak_stats, ax=ax)
        ax.set_title(title)
        figure.canvas.draw_idle()


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
        return float(self.bragg_factor / np.sin(bragg_angle))

    @WatchableAsyncStatus.wrap
    async def set(self, value: BeamMode):
        if value == BeamMode.WHITE:
            xtal1_target, bs_target = self.xtal1_out, self.beam_stop_out
        else:
            xtal1_target, bs_target = self.xtal1_in, self.beam_stop_in
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
        if not auto_tune:
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

        fig_factory = _figure_factory()

        # Scan the crystal 2 pitch and return the center of the fluorescence peak.
        # A fresh PeakStats per scan, so the GUI can plot one while the next runs.
        def energy_auto_tune(scan_name: str, angle_range: float, num_steps: int):
            ps = PeakStats(dclm.xtal2_pitch.name, fs_stats.mean_value.name)
            plotter = _PeakStatsPlot(ps, f"energy_auto_tune {scan_name}", fig_factory)
            # ps must come first so its stats are computed before plotter sees 'stop'.
            yield from subs_wrapper(
                bp.scan(
                    [fs_window_cam],  # type: ignore  # TODO: allow AsyncReadable
                    dclm.xtal2_pitch,
                    angle - angle_range,
                    angle + angle_range,
                    num_steps,
                    md={"plan_name": "energy_auto_tune"},
                ),
                [ps, plotter],
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
        yield from bps.mv(dclm.flourescence_screen, dclm.fs_out)
        if fs_window_cam is not None:
            yield from enable_preview_mode(fs_window_cam)

    yield from (
        finalize_wrapper(
            _body(),
            _cleanup(),
        )
    )
