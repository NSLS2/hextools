"""
2-D grid tomography runner for HEX beamline.

Equivalent of the old pyepics scripts:
    hex-acq-pyepics/techniques/tomography/kinetix/run_multiple_2d_scans.py
    hex-acq-pyepics/techniques/tomography/kinetix/run_multiple_2d_scans_average.py

(As with run_multiple_scans, the ``_average`` variant differed only in which
scan script it launched — the *scan_plan* injection point covers it.)

What this plan does
-------------------
Outer loop over *motor2* (rows), inner loop over *motor1* (columns):
a dark/flat scan at the start of every row (matching the original), then one
tomography fly scan per grid point.  Both motors are restored to their
pre-scan positions at the end.

Sub-directories under *output_base_dir*: ``scan_00001``... in grid order
(row-major), ``dark_flat_00001``... per row.

Usage
-----
    RE(run_multiple_2d_scans(
        kinetix1, panda1, rot_stage, sample_x,
        motor1=sample_tower_x, start1=-2.0, stop1=2.0, num_points1=5,
        motor2=sample_tower_y, start2=0.0, stop2=10.0, num_points2=3,
        ph_open_cmd=ph_open_cmd, ph_close_cmd=ph_close_cmd,
        output_base_dir=".../raw_data/grid_A",
        exposure_time=0.015, num_projections=1801, flat_x_offset=5.0,
    ))
"""

import bluesky.plan_stubs as bps
import numpy as np

from .take_dark_flat import take_dark_flat
from .tomo_flyscan import tomo_flyscan


def run_multiple_2d_scans(
    detectors,
    panda,
    rot_stage,
    sample_x,
    *,
    motor1,
    start1: float,
    stop1: float,
    num_points1: int,
    motor2,
    start2: float,
    stop2: float,
    num_points2: int,
    output_base_dir: str,
    exposure_time: float,
    num_projections: int,
    flat_x_offset: float,
    num_dark: int = 20,
    num_flat: int = 50,
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
    scan_plan=None,
    dark_flat_plan=None,
):
    """Grid of tomography scans: motor2 rows x motor1 columns, dark/flat per row."""
    if num_points1 < 1 or num_points2 < 1:
        raise ValueError("num_points1 and num_points2 must be >= 1")

    positions1 = np.linspace(start1, stop1, num_points1)
    positions2 = np.linspace(start2, stop2, num_points2)
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

    initial1 = yield from bps.rd(motor1)
    initial2 = yield from bps.rd(motor2)

    n_scan = 0
    n_dark_flat = 0
    for row, pos2 in enumerate(positions2):
        print(f"\nRow {row + 1}/{num_points2}: moving {motor2.name} to {pos2:g}")
        # Move BOTH motors to the row's starting point before the row's
        # dark/flat — otherwise motor1 sits at the previous row's last
        # column while flats are taken. (The pyepics original had that
        # quirk; the profile's beamline-run tomo_grid_scan already moved
        # motor1 first, and we follow it.)
        yield from bps.mv(motor2, float(pos2), motor1, float(positions1[0]))

        n_dark_flat += 1
        print(f"Dark/flat for row {row + 1}...")
        yield from dark_flat_plan(f"{base}/dark_flat_{n_dark_flat:05d}", n_dark_flat)

        for col, pos1 in enumerate(positions1):
            print(f"Grid point row {row + 1}, col {col + 1}: "
                  f"moving {motor1.name} to {pos1:g}")
            yield from bps.mv(motor1, float(pos1))
            n_scan += 1
            yield from scan_plan(f"{base}/scan_{n_scan:05d}", n_scan)

    print(f"Restoring {motor1.name} -> {initial1:g}, {motor2.name} -> {initial2:g}")
    yield from bps.mv(motor1, initial1, motor2, initial2)
