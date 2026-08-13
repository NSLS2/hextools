"""
Frame-averaged batch tomography runner for HEX beamline.

Equivalent of the old pyepics script:
    hex-acq-pyepics/techniques/tomography/kinetix/run_multiple_scans_average.py

The original differed from run_multiple_scans.py only in launching the
averaging scan script — here it is run_multiple_scans with
tomo_flyscan_average injected as the per-scan plan.  All batch semantics
(modes, dark/flat cadence, restores) are documented there.
"""

from .run_multiple_scans import run_multiple_scans
from .tomo_flyscan_average import tomo_flyscan_average


def run_multiple_scans_average(
    detectors,
    panda,
    rot_stage,
    sample_x,
    *,
    frames_to_average: int,
    output_base_dir: str,
    exposure_time: float,
    num_projections: int,
    md: dict | None = None,
    **runner_kwargs,
):
    """run_multiple_scans with frame-averaged tomography scans."""
    passthrough = {
        key: runner_kwargs[key]
        for key in (
            "start_deg", "stop_deg", "lead_angle", "acquire_period",
            "time_trigger", "use_shutter", "ph_open_cmd", "ph_close_cmd",
            "fe_shutter_status",
        )
        if key in runner_kwargs
    }

    def scan_plan(output_dir, index):
        return tomo_flyscan_average(
            detectors, panda, rot_stage,
            frames_to_average=frames_to_average,
            output_dir=output_dir,
            exposure_time=exposure_time,
            num_projections=num_projections,
            md=dict(md or {}, batch_index=index),
            **passthrough,
        )

    return (yield from run_multiple_scans(
        detectors, panda, rot_stage, sample_x,
        output_base_dir=output_base_dir,
        exposure_time=exposure_time,
        num_projections=num_projections,
        md=md,
        scan_plan=scan_plan,
        **runner_kwargs,
    ))
