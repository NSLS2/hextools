"""Plans for the photon delivery system: shutters, slits, filters and DCLM."""

import logging
from collections.abc import Hashable

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import bluesky.preprocessors as bpp
import numpy as np
from bluesky.callbacks.fitting import PeakStats
from bluesky.preprocessors import finalize_wrapper, subs_wrapper
from ophyd_async.core import DEFAULT_TIMEOUT
from ophyd_async.epics.adcore import NDStatsIO
from ophyd_async.epics.advimba import VimbaDetector

from hextools.devices.dclm import DCLM, BeamMode
from hextools.devices.filters import (
    Filter,
    Filter1DownstreamPosition,
    Filter1UpstreamPosition,
    Filter2Position,
    FilterPosition,
)
from hextools.devices.shutter import PPSShutter, Shutter, ShutterStatus
from hextools.devices.slits import FrontEndSlit, SlitAperture, Slits
from hextools.plans.detectors import enable_preview_mode

logger = logging.getLogger("hextools")

__all__ = [
    "ensure_shutter_state",
    "ensure_shutter_open",
    "ensure_shutter_closed",
    "set_fe_aperture",
    "move_slits",
    "set_filters",
    "move_filters",
    "change_beam_mode",
    "change_energy",
]


def ensure_shutter_state(
    shutter: Shutter | list[Shutter],
    desired_state: bool,
    allow_actuation: bool = False,
    group: Hashable | None = None,
    wait: bool = True,
    sleep_time: float = 4.0,
):
    """Plan stub to guarantees that the shutter(s) is/are in the desired state.

    Parameters
    ----------
    shutter : Shutter | list[Shutter]
        shutter(s) to guarantee the state of.
    desired_state : bool
        the state of the shutter (True for open, False for closed)
    allow_actuation : bool, default False
        whether to allow the plan to actuate the shutter if it is not in the
        desired state
    group : Hashable | None, optional
        the Bluesky group to use for the actuation, if any
    wait : bool, default True
        whether to wait for the shutter to reach the desired state after actuation
    sleep_time : float, default 4.0
        the time to sleep to ensure the shutter has reached the desired state
    """
    shutters = shutter if isinstance(shutter, list) else [shutter]

    logger.info(
        f"Ensuring shutter(s) {', '.join(s.name for s in shutters)} are in the "
        f"desired state: {'open' if desired_state else 'closed'}"
    )

    shutter_statuses = {}
    for s in shutters:
        shutter_statuses[s.name] = yield from bps.rd(s.status)

    actuated = False
    for s in shutters:
        if (shutter_statuses[s.name] == ShutterStatus.OPEN) != desired_state:
            if allow_actuation:
                yield from bps.abs_set(s, desired_state, group=group, wait=False)
                actuated = True
            else:
                raise ValueError(
                    f"Shutter {s.name} is {shutter_statuses[s.name].name.lower()}, but"
                    f" the desired state is {'open' if desired_state else 'closed'}!"
                )

    if wait and actuated:
        yield from bps.wait(group=group, timeout=DEFAULT_TIMEOUT)

        # It seems that the shutter status in EPICS is not reliable enough to
        # consider the shutter fully opened/closed, so wait some extra time
        # if requested.
        if sleep_time > 0:
            yield from bps.sleep(sleep_time)


def ensure_shutter_open(
    shutter: Shutter | list[Shutter],
    allow_actuation: bool = False,
    group: Hashable | None = None,
    wait: bool = True,
    sleep_time: float = 4.0,
):
    """Plan stub to guarantee that the shutter(s) is/are open.

    Parameters
    ----------
    shutter : Shutter | list[Shutter]
        shutter(s) to guarantee the state of.
    allow_actuation : bool, default False
        whether to allow the plan to actuate the shutter if it is not in the
        desired state
    group : Hashable | None, optional
        the Bluesky group to use for the actuation, if any
    wait : bool, default True
        whether to wait for the shutter to reach the desired state after actuation
    sleep_time : float, default 4.0
        the time to sleep to ensure the shutter has reached the desired state
    """
    yield from ensure_shutter_state(
        shutter,
        True,
        allow_actuation=allow_actuation,
        wait=wait,
        group=group,
        sleep_time=sleep_time,
    )


def ensure_shutter_closed(
    shutter: Shutter | list[Shutter],
    allow_actuation: bool = True,
    group: Hashable | None = None,
    wait: bool = True,
    sleep_time: float = 4.0,
):
    """Plan stub to guarantee that the shutter is closed.

    Parameters
    ----------
    shutter : Shutter | list[Shutter]
        shutter(s) to guarantee the state of.
    allow_actuation : bool, default True
        whether to allow the plan to actuate the shutter if it is not in the
        desired state
    group : Hashable | None, optional
        the Bluesky group to use for the actuation, if any
    wait : bool, default True
            whether to wait for the shutter to reach the desired state after actuation
    sleep_time : float, default 4.0
        the time to sleep to ensure the shutter has reached the desired state
    """
    yield from ensure_shutter_state(
        shutter,
        False,
        allow_actuation=allow_actuation,
        wait=wait,
        group=group,
        sleep_time=sleep_time,
    )


def set_fe_aperture(fe_slit1: FrontEndSlit, aperture: SlitAperture):
    """Move front-end slit Y to a named aperture.

    Parameters
    ----------
    fe_slit1 : FrontEndSlit
        The slit.
    aperture : SlitAperture
        The aperture: fully open (24 x 4.55 mm), 7.13 x 3.56 mm, fully closed,
        3.56 x 3.56 mm, or the 0.89 mm pinhole.
    """
    # Resuming after a pause rewinds here, so the slit is moved again.
    yield from bps.checkpoint()
    yield from bps.mv(fe_slit1, SlitAperture(aperture))


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
            # Resuming after a pause rewinds here, so this stage is moved again.
            yield from bps.checkpoint()
            yield from move(*args)


def set_filters(
    filter1_upstream: Filter,
    filter1_downstream: Filter,
    filter2: Filter,
    fe_shutter: PPSShutter,
    filter1_upstream_setting: Filter1UpstreamPosition | None = None,
    filter1_downstream_setting: Filter1DownstreamPosition | None = None,
    filter2_setting: Filter2Position | None = None,
):
    """Set filter 1 upstream, filter 1 downstream and filter 2 by material.

    Each setting is one of that filter's materials from ``bl_constants.yml`` (for
    example ``"6 mm Cu"``) or ``"Pass Through"``. A filter with no setting does
    not move; the others move together.

    Moving a filter closes the front-end shutter through its interlock. Once the
    filters are in position the interlock is acknowledged, and the front-end
    shutter is reopened if it was open before.

    Parameters
    ----------
    filter1_upstream, filter1_downstream, filter2 : Filter
        The filter devices.
    fe_shutter : PPSShutter
        The front-end shutter.
    filter1_upstream_setting : Filter1UpstreamPosition, optional
        Material for filter 1 upstream.
    filter1_downstream_setting : Filter1DownstreamPosition, optional
        Material for filter 1 downstream.
    filter2_setting : Filter2Position, optional
        Material for filter 2.

    Raises
    ------
    ValueError
        If no setting is given, or a setting is not one of that filter's choices.
    """
    args = []
    for device, setting, choices in (
        (filter1_upstream, filter1_upstream_setting, Filter1UpstreamPosition),
        (filter1_downstream, filter1_downstream_setting, Filter1DownstreamPosition),
        (filter2, filter2_setting, Filter2Position),
    ):
        if setting is not None:
            # Lookup by value accepts the member or its material string, and raises
            # ValueError for anything that isn't one of this filter's choices.
            args += [device, FilterPosition[choices(setting).name]]
    if not args:
        raise ValueError("Choose a setting for at least one filter.")
    yield from move_filters(*args, fe_shutter=fe_shutter)


def move_filters(*args: Filter | FilterPosition | str, fe_shutter: PPSShutter):
    """Move filters, given as ``filter1, setting1, filter2, setting2, ...``.

    Each setting is a :class:`FilterPosition` or that filter's label for it (for
    example ``"6 mm Cu"`` or ``"Pass Through"``). The filters move together, and
    the plan returns once they are in position.

    Moving a filter closes the front-end shutter through its interlock. Once the
    filters are in position the interlock is acknowledged, and the front-end
    shutter is reopened if it was open before.

    Parameters
    ----------
    *args : Filter, FilterPosition or str
        Alternating filters and settings.
    fe_shutter : PPSShutter
        The front-end shutter.

    Raises
    ------
    ValueError
        If no filters are given, the arguments don't pair up, or a setting is not
        one of its filter's.
    """
    if not args or len(args) % 2:
        raise ValueError("Give one or more filter, setting pairs.")
    moves: list = []
    for device, setting in zip(args[::2], args[1::2], strict=True):
        if not isinstance(device, Filter):
            raise ValueError(f"{device!r} is not a filter")
        moves += [device, device.position_for(setting)]  # type: ignore[arg-type]

    fe_was_open = (yield from bps.rd(fe_shutter.status)) == ShutterStatus.OPEN
    # Resuming after a pause rewinds here, so the filters are moved again.
    yield from bps.checkpoint()
    yield from bps.mv(*moves)
    yield from bps.mv(fe_shutter.interlock_ack, True)
    if fe_was_open:
        yield from ensure_shutter_open(fe_shutter, allow_actuation=True, sleep_time=0)


def change_beam_mode(dclm: DCLM, fe_shutter: Shutter, mode: BeamMode):
    """Bluesky plan to change the beam mode of the DCLM.

    Parameters
    ----------
    dclm : DCLM
        The DCLM to control.
    fe_shutter : Shutter
        The front-end shutter, closed for white beam and reopened after.
    mode : BeamMode
        The target beam mode for the DCLM
    """
    current_mode = yield from bps.rd(dclm.beam_mode)
    if current_mode == mode:
        return
    if mode == BeamMode.WHITE:
        yield from ensure_shutter_closed(fe_shutter)

    # Resuming after a pause rewinds here, so the beam mode change is made again.
    yield from bps.checkpoint()
    yield from bps.mv(dclm, mode)
    yield from ensure_shutter_open(fe_shutter, allow_actuation=True)


def change_energy(
    dclm: DCLM,
    fs_window_cam: VimbaDetector,
    energy: float,
    coarse_angle_range: float = 0.1,
    coarse_num_steps: int = 41,
    fine_angle_range: float = 0.025,
    fine_num_steps: int = 26,
    fs_stats_plugin_name: str = "stats1",
    exposure_time: float = 0.5,
    auto_tune: bool = True,
):
    """Bluesky plan to change monochromator energy for Si(111).

    Computes Bragg geometry and moves all crystal and beam stop motors.
    With ``auto_tune``, performs coarse and fine pitch scans to auto-tune the
    crystal 2 pitch to the fluorescence peak.

    Parameters
    ----------
    dclm : DCLM
        Must be in monochromatic mode.
    fs_window_cam : VimbaDetector
        Fluorescence screen camera for auto-tuning. Unused without
        ``auto_tune``, when motors are moved without feedback.
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
    exposure_time : float, optional
        Exposure time for the fluorescence screen camera during auto-tuning.

    Raises
    ------
    RuntimeError
        If the monochromator is not in monochromatic mode.
    """

    @bpp.reset_positions_decorator(
        [fs_window_cam.driver.acquire_time, fs_window_cam.driver.acquire_period]
        if auto_tune
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

        # Resuming after a pause rewinds to the checkpoint before each move.
        yield from bps.checkpoint()
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

        # Without auto-tuning, just move the motors and return.
        if not auto_tune:
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
        yield from bps.checkpoint()
        yield from bps.mv(dclm.xtal2_pitch, coarse_center)

        fine_center = yield from energy_auto_tune(
            "fine", fine_angle_range, fine_num_steps
        )

        # To minimize issues with backlash, always approach the final position
        # from below.
        yield from bps.checkpoint()
        yield from bps.mv(dclm.xtal2_pitch, fine_center - 0.05)
        yield from bps.mv(dclm.xtal2_pitch, fine_center)

    def _cleanup():
        # Reset the fluorescence screen and close the shutter on the way out.
        yield from bps.mv(dclm.flourescence_screen, dclm.constants.fs_out)
        if auto_tune:
            yield from enable_preview_mode(fs_window_cam)

    yield from (
        finalize_wrapper(
            _body(),
            _cleanup(),
        )
    )
