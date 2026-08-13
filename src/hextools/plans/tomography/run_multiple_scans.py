"""
Batch tomography runner for HEX beamline.

Equivalent of the old pyepics scripts:
    hex-acq-pyepics/techniques/tomography/kinetix/run_multiple_scans.py
    hex-acq-pyepics/techniques/tomography/kinetix/run_multiple_scans_average.py

(The ``_average`` variant differed only in which scan script it launched —
here that's the *scan_plan* argument, so one runner covers both rows.)

What this plan does
-------------------
Runs a series of tomography fly scans, either

- **motor mode**: step *scan_motor* through linspace(start, stop,
  num_points), one tomo scan per position, restoring the motor afterwards; or
- **time mode**: *num_scans* scans separated by *sleep_time* seconds,

with dark/flat handling exactly like the original ``tomo_skip`` semantics
(*dark_flat_every*): ``-1`` = one dark/flat scan at the start only; ``0`` =
never; ``N > 0`` = after every N tomo scans (no initial).

Each sub-scan writes to its own sub-directory of *output_base_dir*
(``scan_00001``, ... / ``dark_flat_00001``, ...).  The old scripts used
globally-numbered scan_NNNNN folders allocated by scanning the filesystem;
here numbering is per-run (Bluesky scan ids / Tiled carry global identity),
which also keeps the plan free of control-host filesystem access.

Usage
-----
    RE(run_multiple_scans(
        kinetix1, panda1, rot_stage, sample_x,
        ph_open_cmd=ph_open_cmd, ph_close_cmd=ph_close_cmd,
        output_base_dir=".../raw_data/batch_A",
        exposure_time=0.015, num_projections=1801, flat_x_offset=5.0,
        scan_motor=sample_y, start=-2.0, stop=2.0, num_points=5,
        dark_flat_every=2,
    ))
"""

import bluesky.plan_stubs as bps
import numpy as np

from .take_dark_flat import take_dark_flat
from .tomo_flyscan import tomo_flyscan


def run_multiple_scans(
    detectors,
    panda,
    rot_stage,
    sample_x,
    *,
    output_base_dir: str,
    exposure_time: float,
    num_projections: int,
    flat_x_offset: float,
    # motor mode
    scan_motor=None,
    start: float | None = None,
    stop: float | None = None,
    num_points: int | None = None,
    # time mode
    num_scans: int | None = None,
    sleep_time: float = 0.0,
    # dark/flat cadence (the old tomo_skip semantics)
    dark_flat_every: int = -1,
    num_dark: int = 20,
    num_flat: int = 50,
    # per-scan parameters passed through to tomo_flyscan
    start_deg: float = 0.0,
    stop_deg: float = 180.0,
    lead_angle: float = 10.0,
    acquire_period: float = 0.0,
    time_trigger: bool = True,
    use_shutter: bool = True,
    ph_open_cmd=None,
    ph_close_cmd=None,
    fe_shutter_status=None,
    md: dict | None = None,
    # injection points: callables (output_dir, index) -> plan.  Defaults run
    # tomo_flyscan / take_dark_flat with the arguments above; the _average
    # variant passes an averaging scan_plan here.
    scan_plan=None,
    dark_flat_plan=None,
):
    """Run multiple tomography scans with periodic dark/flat collection."""
    motor_mode = scan_motor is not None
    if motor_mode:
        if None in (start, stop, num_points) or num_points < 1:
            raise ValueError(
                "motor mode needs scan_motor, start, stop and num_points >= 1"
            )
        if num_scans is not None:
            raise ValueError("pass either scan_motor+range OR num_scans, not both")
        positions = np.linspace(start, stop, num_points)
        n_iterations = num_points
    else:
        if num_scans is None or num_scans < 2:
            raise ValueError("time mode needs num_scans > 1 (or pass scan_motor)")
        if sleep_time <= 0.0:
            raise ValueError(
                f"time mode needs a strictly positive sleep_time, got {sleep_time}"
            )
        positions = None
        n_iterations = num_scans

    if dark_flat_every < -1:
        raise ValueError(
            f"dark_flat_every must be -1 (initial only), 0 (never) or N > 0 "
            f"(every N scans), got {dark_flat_every}"
        )

    base = output_base_dir.rstrip("/")

    if scan_plan is None:
        def scan_plan(output_dir, index):
            return tomo_flyscan(
                detectors, panda, rot_stage,
                output_dir=output_dir,
                exposure_time=exposure_time,
                num_projections=num_projections,
                start_deg=start_deg, stop_deg=stop_deg,
                lead_angle=lead_angle, acquire_period=acquire_period,
                time_trigger=time_trigger, use_shutter=use_shutter,
                ph_open_cmd=ph_open_cmd, ph_close_cmd=ph_close_cmd,
                fe_shutter_status=fe_shutter_status,
                md=dict(md or {}, batch_index=index),
            )

    if dark_flat_plan is None:
        def dark_flat_plan(output_dir, index):
            first = detectors[0] if isinstance(detectors, (list, tuple)) else detectors
            return take_dark_flat(
                first, sample_x, ph_open_cmd, ph_close_cmd,
                output_dir=output_dir,
                exposure_time=exposure_time,
                num_dark=num_dark, num_flat=num_flat,
                flat_x_offset=flat_x_offset,
                md=dict(md or {}, batch_index=index),
            )

    n_dark_flat = 0

    def _dark_flat():
        nonlocal n_dark_flat
        n_dark_flat += 1
        yield from dark_flat_plan(f"{base}/dark_flat_{n_dark_flat:05d}", n_dark_flat)

    if dark_flat_every == -1:
        print("Taking the initial dark/flat scan...")
        yield from _dark_flat()

    if motor_mode:
        initial_position = yield from bps.rd(scan_motor)

    for i in range(n_iterations):
        if motor_mode:
            print(f"\nTomo scan {i + 1}/{n_iterations}: "
                  f"moving {scan_motor.name} to {positions[i]:g}")
            yield from bps.mv(scan_motor, float(positions[i]))
        else:
            print(f"\nTomo scan {i + 1}/{n_iterations} (time series)")

        yield from scan_plan(f"{base}/scan_{i + 1:05d}", i + 1)

        if not motor_mode and i < n_iterations - 1:
            print(f"Pausing {sleep_time} s before the next scan...")
            yield from bps.sleep(sleep_time)

        if dark_flat_every > 0 and (i + 1) % dark_flat_every == 0:
            print(f"Dark/flat after scan {i + 1}...")
            yield from _dark_flat()

    if motor_mode:
        print(f"Restoring {scan_motor.name} to {initial_position:g}")
        yield from bps.mv(scan_motor, initial_position)
