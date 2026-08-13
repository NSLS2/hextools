"""
Hardware-triggered fly-scan tomography plan for HEX beamline.

Equivalent of the old pyepics script:
    hex-acq-pyepics/techniques/tomography/kinetix/tomo_flyscan.py
and the parametrized successor of hex-profile-collection's
``startup/85-fly-plans.py::tomo_flyscan`` (devices are arguments here, not
profile globals).

What this plan does
-------------------
1.  Optionally checks the front-end status and opens the photon shutter.
2.  Moves the rotation stage to *start_deg* (to read the PandA encoder
    value), backs off by *lead_angle*, and sets the scan velocity computed
    from the angular range and frame period.
3.  Configures the PandA: PCOMP armed at the start-angle encoder count; in
    time-trigger mode PCOMP fires once and PULSE1 emits *num_projections*
    pulses at the frame period (position-trigger mode: one PCOMP pulse per
    projection).  ``calc[2]`` (the angle calculation) is captured to the
    PandA HDF file as the ``Angle`` dataset.
4.  Stages+prepares detector(s) (external-edge triggering) and the PandA,
    kicks everything off, sweeps the rotation stage to *stop_deg* +
    *lead_angle*, and collects into the ``tomo`` stream while completing.
5.  Finalizer: closes the shutter (when used) and restores the reset
    velocity — runs even on error/interrupt.

Outputs: the detector HDF file (``proj``) and the PandA HDF file
(``panda`` — with the per-trigger ``Angle`` dataset) under *output_dir*.

Usage
-----
    RE(tomo_flyscan(
        kinetix1, panda1, rot_stage,
        ph_open_cmd=ph_open_cmd, ph_close_cmd=ph_close_cmd,
        output_dir=".../raw_data/scan_00001",
        exposure_time=0.015, num_projections=1801,
        start_deg=0, stop_deg=180,
    ))
"""

from collections.abc import Sequence

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from ophyd_async.core import DetectorTrigger, TriggerInfo
from ophyd_async.epics.adkinetix import KinetixReadoutMode

from hextools.detectors.kinetix import set_output_dir
from hextools.plans.shutter import close_photon_shutter, open_photon_shutter

# Kinetix per-readout-mode frame-rate ceilings (fps) — from the profile.
DETECTOR_MAX_FRAMERATES = {
    KinetixReadoutMode.SENSITIVITY: 50,
    KinetixReadoutMode.SPEED: 250,
    KinetixReadoutMode.DYNAMIC_RANGE: 75,
}

TOMO_ROTARY_STAGE_VELO_RESET_MAX = 30  # deg/s, non-scan moves
TOMO_ROTARY_STAGE_VELO_SCAN_MAX = 60   # deg/s, during the sweep
OVERHEAD = 0.005                       # s, minimum frame-period overhead
ENCODER_SETTLE_TOL = 20                # counts between reads == stationary
ENCODER_SETTLE_TIMEOUT = 30.0          # s


def tomo_flyscan(
    detectors,
    panda,
    rot_stage,
    *,
    output_dir: str,
    exposure_time: float,
    num_projections: int,
    start_deg: float = 0.0,
    stop_deg: float = 180.0,
    lead_angle: float = 10.0,
    acquire_period: float = 0.0,
    time_trigger: bool = True,
    raw_frames_per_projection: int = 1,
    reset_speed: float = TOMO_ROTARY_STAGE_VELO_RESET_MAX,
    use_shutter: bool = True,
    ph_open_cmd=None,
    ph_close_cmd=None,
    fe_shutter_status=None,
    md: dict | None = None,
):
    """
    Fly-scan tomography: rotate continuously, PandA-triggered frames.

    Parameters
    ----------
    detectors : device or sequence of devices
        The camera(s), built by ``hextools.detectors.kinetix.make_kinetix`` (dual-cam
        later passes two).
    panda : HDFPanda
        Built by ``hextools.detectors.panda.make_panda`` — trigger source + Angle capture.
    rot_stage : Motor
        The tomography rotation stage.
    output_dir : str
        Directory for both HDF files (the old script's scan_NNNNN folder).
    exposure_time : float
        Camera exposure per projection, seconds.
    num_projections : int
        Number of projections (e.g. 1801).
    start_deg / stop_deg : float, optional
        Angular range of the scan. Defaults 0 / 180.
    lead_angle : float, optional
        Run-up/run-out angle so the stage is at constant speed across the
        range. Default 10.
    acquire_period : float, optional
        Frame period; 0 (default) derives exposure + overhead.
    time_trigger : bool, optional
        True (default): PCOMP fires once, PULSE1 emits the time train.
        False: one PCOMP (position) pulse per projection.
    raw_frames_per_projection : int, optional
        Raw camera frames per delivered projection (default 1).  Used by
        the frame-averaging variant: the camera acquires
        num_projections x this many frames (one per PandA pulse) while the
        Proc plugin averages them down, so the HDF writer still sees
        num_projections frames.  Plain scans leave this at 1.
    reset_speed : float, optional
        Stage velocity for non-scan moves (capped at the reset max).
    use_shutter : bool, optional
        Check front-end (when *fe_shutter_status* given) and drive the
        photon shutter via *ph_open_cmd*/*ph_close_cmd*. Default True.
    ph_open_cmd / ph_close_cmd : signal, optional
        Photon-shutter command signals (required when *use_shutter*).
    fe_shutter_status : signal, optional
        Front-end status readback; checked == 1 when provided.
    md : dict, optional
        Extra run metadata.
    """
    if not isinstance(detectors, Sequence):
        detectors = [detectors]
    detectors = list(detectors)

    if not detectors:
        raise ValueError("detectors must contain at least one detector")
    if stop_deg <= start_deg:
        raise ValueError(
            f"stop_deg ({stop_deg}) must be > start_deg ({start_deg}) — "
            "reverse scans are not supported (the sweep-wait logic assumes "
            "a forward scan)."
        )
    if num_projections < 2:
        raise ValueError(f"num_projections must be >= 2, got {num_projections}")
    if raw_frames_per_projection < 1:
        raise ValueError(
            f"raw_frames_per_projection must be >= 1, got {raw_frames_per_projection}"
        )
    if exposure_time <= 0:
        raise ValueError(f"exposure_time must be > 0, got {exposure_time}")
    if use_shutter and (ph_open_cmd is None or ph_close_cmd is None):
        raise ValueError(
            "use_shutter=True needs ph_open_cmd and ph_close_cmd "
            "(or pass use_shutter=False)."
        )

    # ------------------------------------------------------------------
    # Kinematics (same rules as the pyepics script / profile plan)
    # ------------------------------------------------------------------
    if (acquire_period + OVERHEAD) < exposure_time or acquire_period == 0.0:
        acquire_period = exposure_time + OVERHEAD

    raw_frames = num_projections * raw_frames_per_projection

    reset_speed = min(reset_speed, TOMO_ROTARY_STAGE_VELO_RESET_MAX)

    output_path = None
    for det in detectors:
        output_path = set_output_dir(det, output_dir, "proj")
    set_output_dir(panda, output_dir, "panda")

    _md = {
        "plan_name": "tomo_flyscan",
        "detectors": [det.name for det in detectors],
        "motors": [rot_stage.name],
        "output_dir": str(output_path),
        "exposure_time": exposure_time,
        "num_projections": num_projections,
        "start_deg": start_deg,
        "stop_deg": stop_deg,
        "lead_angle": lead_angle,
        "time_trigger": time_trigger,
        "hints": {},
    }
    _md.update(md or {})

    panda_pcomp = panda.pcomp[1]
    panda_pulser = panda.pulse[1]

    # Tracks whether devices are currently staged, so the finalizer can
    # unstage on error/interrupt without double-unstaging the happy path.
    staged: dict = {"devices": None}

    def _scan():
        if use_shutter:
            if fe_shutter_status is not None:
                if (yield from bps.rd(fe_shutter_status)) != 1:
                    raise RuntimeError("Front-end shutter is closed. Reopen it!")
            yield from open_photon_shutter(ph_open_cmd)

        # Frame period / sweep velocity / angular span form one consistent
        # triangle, resolved HERE so every cap feeds back into the others
        # (the profile plan adjusted the pulse period for the readout-mode
        # cap after computing the velocity, letting the sweep desync from
        # the trigger train; the pyepics script's recompute-from-the-cap
        # behavior is restored):
        #   1. step_time from acquire_period, capped by the camera's
        #      readout-mode max frame rate (Kinetix-specific; skipped for
        #      detectors without the signal);
        #   2. sweep velocity from the span and RAW frame count;
        #   3. if the velocity cap engages, stretch step_time to match;
        #   4. the exposure must fit the final step_time.
        step_time = acquire_period
        for det in detectors:
            if hasattr(det.driver, "readout_port_idx"):
                readout_mode = yield from bps.rd(det.driver.readout_port_idx)
                max_rate = DETECTOR_MAX_FRAMERATES.get(readout_mode)
                if max_rate and 1 / step_time > max_rate:
                    step_time = 1 / max_rate

        scan_time = (raw_frames - 1) * step_time
        rot_motor_vel = abs(stop_deg - start_deg) / scan_time
        if rot_motor_vel > TOMO_ROTARY_STAGE_VELO_SCAN_MAX:
            rot_motor_vel = TOMO_ROTARY_STAGE_VELO_SCAN_MAX
            scan_time = abs(stop_deg - start_deg) / rot_motor_vel
            step_time = scan_time / (raw_frames - 1)

        if exposure_time > step_time:
            raise RuntimeError(
                f"Exposure time {exposure_time} s is longer than the final "
                f"time per step {step_time:.6g} s (after readout-rate and "
                f"velocity caps)!"
            )

        det_trigger_info = TriggerInfo(
            number_of_events=num_projections,
            trigger=DetectorTrigger.EXTERNAL_EDGE,
            livetime=exposure_time,
            deadtime=0.001,
        )
        # The PandA pulses/captures once per RAW frame (the Angle series has
        # one entry per pulse, like the legacy scripts).
        panda_trigger_info = TriggerInfo(
            number_of_events=raw_frames,
            trigger=DetectorTrigger.EXTERNAL_LEVEL,
            livetime=step_time,
            deadtime=0.0001,
        )

        def _wait_encoder_settled(label):
            # The PandA must see the stage AT its commanded position: a
            # lagging encoder (or the sim's slew-limited motor->INENC
            # bridge after big repositioning moves) yields stale reads —
            # a stale start_encoder mis-programs PCOMP.START, and arming
            # with the apparent position past START fires immediately.
            # Calibration-free: settled == two reads 0.2 s apart agree.
            last = yield from bps.rd(panda.calc[2].out)
            for _ in range(int(ENCODER_SETTLE_TIMEOUT / 0.2)):
                yield from bps.sleep(0.2)
                now = yield from bps.rd(panda.calc[2].out)
                if abs(now - last) < ENCODER_SETTLE_TOL:
                    return now
                last = now
            raise RuntimeError(
                f"Encoder still moving at the {label} position "
                f"(last read {last:.0f} counts)"
            )

        # -- position the stage and read the start-angle encoder count ----
        yield from bps.mv(rot_stage.velocity, reset_speed)
        yield from bps.mv(rot_stage, start_deg)
        start_encoder = yield from _wait_encoder_settled("start")
        yield from bps.mv(rot_stage, start_deg - lead_angle)
        yield from _wait_encoder_settled("lead")
        yield from bps.mv(rot_stage.velocity, rot_motor_vel)

        # -- PandA block configuration ------------------------------------
        yield from bps.mv(panda_pcomp.start, int(start_encoder))
        yield from bps.mv(panda_pcomp.width, 3)
        if time_trigger:
            yield from bps.mv(
                panda_pcomp.pulses, 1,
                panda_pulser.pulses, raw_frames,
                panda_pulser.step, step_time,
                panda_pulser.width, exposure_time / 5,
            )
        else:
            yield from bps.mv(panda_pcomp.pulses, raw_frames)
        yield from bps.mv(panda.calc[2].out_dataset, "Angle")

        yield from bps.open_run(md=_md)
        print(f"\nExecuting tomography fly scan: {num_projections} projections, "
              f"{start_deg:g} -> {stop_deg:g} deg at {rot_motor_vel:.3g} deg/s")

        all_devices = [panda] + detectors
        for det in detectors:
            if hasattr(det.hdf, "queue_size"):
                yield from bps.mv(det.hdf.queue_size, raw_frames * 2)

        yield from bps.stage_all(*all_devices)
        staged["devices"] = all_devices
        for det in detectors:
            yield from bps.prepare(det, det_trigger_info, wait=True)
            if raw_frames_per_projection > 1:
                # prepare() sized the camera for num_projections (what the
                # writer will see after Proc averaging); the camera itself
                # must acquire every raw frame.  (KinetixTriggerLogic on
                # 0.19 has no exposures_per_collection hook, so we widen
                # num_images explicitly.)
                yield from bps.mv(det.driver.num_images, raw_frames)
        yield from bps.prepare(panda, panda_trigger_info, wait=True)

        yield from bps.declare_stream(*all_devices, name="tomo")
        yield from bps.kickoff_all(*all_devices, wait=True)

        # Sweep — non-blocking group move so the RunEngine keeps handling
        # messages (pause/abort) while we collect.
        yield from bps.abs_set(rot_stage, stop_deg + lead_angle, group="tomo_sweep")

        current_pos = yield from bps.rd(rot_stage)
        while current_pos < start_deg:
            yield from bps.sleep(0.1)
            current_pos = yield from bps.rd(rot_stage)

        print("Completing...")
        yield from bps.collect_while_completing(
            all_devices, all_devices,
            flush_period=max(1, exposure_time),
            stream_name="tomo",
        )
        yield from bps.unstage_all(*all_devices)
        staged["devices"] = None

        # Make sure the sweep move is done before closing out.
        yield from bps.wait(group="tomo_sweep")
        yield from bps.close_run()

        # Report captured counts (parity with the profile plan).
        captured = {panda.name: (yield from bps.rd(panda.data.num_captured))}
        for det in detectors:
            captured[det.name] = yield from bps.rd(det.hdf.num_captured)
        print("Number of frames captured:")
        for name, count in captured.items():
            print(f"    {name:15}: {count}")

    def _cleanup():
        # Unstage anything still staged (error/interrupt path) — disarms
        # the detectors/PandA and restores the Kinetix live view.
        if staged["devices"] is not None:
            yield from bps.unstage_all(*staged["devices"])
            staged["devices"] = None
        if use_shutter and ph_close_cmd is not None:
            yield from close_photon_shutter(ph_close_cmd)
        yield from bps.mv(rot_stage.velocity, reset_speed)

    return (yield from bpp.finalize_wrapper(_scan(), _cleanup()))
