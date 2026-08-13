"""
Frame-averaged fly-scan tomography plan for HEX beamline.

Equivalent of the old pyepics script:
    hex-acq-pyepics/techniques/tomography/kinetix/tomo_flyscan_average.py

How averaging works (same mechanism as the original)
----------------------------------------------------
Averaging happens IOC-side in the AreaDetector process plugin: the camera
acquires ``num_projections x frames_to_average`` raw frames (one per PandA
pulse), ``Proc1``'s recursive filter averages every *frames_to_average* of
them, and the HDF plugin — re-pointed to consume from ``Proc1`` — writes the
``num_projections`` averaged frames.  The PandA still captures one ``Angle``
entry per raw frame.

Differences from the original (flagged for beamline review)
-----------------------------------------------------------
- ``num_projections`` here is the number of DELIVERED (averaged) projections
  — the old script's ``-n`` was the raw frame count (``-n / -m`` delivered).
- Plugin routing is port-name agnostic: the plan reads the HDF plugin's
  current source port, splices Proc1 in front of it, and RESTORES the
  original routing in the finalizer.  (The old script left HDF1 wired to
  PROC1 with the filter disabled, which is what util/reset_detector.py
  existed to clean up.)

Usage
-----
    RE(tomo_flyscan_average(
        kinetix1, panda1, rot_stage,
        ph_open_cmd=ph_open_cmd, ph_close_cmd=ph_close_cmd,
        output_dir=".../raw_data/scan_00001",
        exposure_time=0.015, num_projections=1801, frames_to_average=4,
        start_deg=0, stop_deg=180,
    ))
"""

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from ophyd_async.core import EnableDisable

from .tomo_flyscan import tomo_flyscan


def configure_averaging(detector, frames_to_average: int):
    """
    Wire the detector's Proc plugin for recursive-average filtering and
    route the HDF plugin to consume from it.  Returns the HDF plugin's
    previous source port (for restore_averaging).

    Field sequence mirrors the legacy ``set_number_frame_for_averaging`` +
    ``enable_filter`` helpers.
    """
    proc = detector.proc
    proc_port = yield from bps.rd(proc.port_name)
    hdf_port = yield from bps.rd(detector.hdf.nd_array_port)
    if hdf_port == proc_port:
        # Already routed through the proc plugin (e.g. a previous run died
        # before restore). The true upstream is proc's current source —
        # using hdf_port here would wire proc to itself.
        camera_port = yield from bps.rd(proc.nd_array_port)
    else:
        camera_port = hdf_port
    saved_port = camera_port

    yield from bps.mv(proc.nd_array_port, camera_port)
    yield from bps.mv(proc.enable_callbacks, EnableDisable.ENABLE)
    yield from bps.mv(proc.num_filter, 1)
    yield from bps.mv(proc.reset_filter, "Yes")
    yield from bps.mv(proc.enable_filter, "Enable")
    yield from bps.mv(proc.num_filter, frames_to_average)
    yield from bps.mv(proc.filter_type, "RecursiveAve")
    yield from bps.mv(proc.auto_reset_filter, "Yes")
    yield from bps.mv(proc.filter_callbacks, "Array N only")

    yield from bps.mv(detector.hdf.nd_array_port, proc_port)
    return saved_port


def restore_averaging(detector, saved_port):
    """Disable the filter and point the HDF plugin back at *saved_port*."""
    proc = detector.proc
    yield from bps.mv(proc.enable_filter, "Disable")
    yield from bps.mv(proc.num_filter, 1)
    yield from bps.mv(detector.hdf.nd_array_port, saved_port)
    # Nothing consumes from proc anymore — stop it processing every frame
    # (the legacy scripts' disable_plugin("Proc1")).
    yield from bps.mv(proc.enable_callbacks, EnableDisable.DISABLE)


def tomo_flyscan_average(
    detectors,
    panda,
    rot_stage,
    *,
    frames_to_average: int,
    md: dict | None = None,
    **tomo_kwargs,
):
    """
    Frame-averaged tomography fly scan.

    Parameters are those of :func:`tomo_flyscan` plus:

    frames_to_average : int
        Raw frames averaged into each delivered projection (>= 2; the old
        script's ``-m``/mean_frame).  ``num_projections`` counts the
        DELIVERED averaged projections.
    """
    if frames_to_average < 2:
        raise ValueError(
            f"frames_to_average must be >= 2, got {frames_to_average} "
            "(use tomo_flyscan for unaveraged scans)"
        )
    if not isinstance(detectors, (list, tuple)):
        detectors = [detectors]
    for det in detectors:
        if not hasattr(det, "proc"):
            raise TypeError(
                f"{getattr(det, 'name', type(det).__name__)} has no 'proc' "
                "plugin; build it with hextools.detectors.kinetix.make_kinetix."
            )

    saved_ports: dict = {}

    def _body():
        # Stop the live view BEFORE re-routing (the legacy stop_preview):
        # otherwise streaming arrays sit in the plugin queues mid-rewire and
        # leak into the new capture as phantom pre-kickoff frames.
        for det in detectors:
            yield from bps.abs_set(det.driver.acquire, 0, wait=True)
        for det in detectors:
            saved_ports[det.name] = yield from configure_averaging(
                det, frames_to_average
            )
        yield from tomo_flyscan(
            detectors, panda, rot_stage,
            raw_frames_per_projection=frames_to_average,
            md=dict(md or {}, frames_to_average=frames_to_average,
                    plan_name="tomo_flyscan_average"),
            **tomo_kwargs,
        )

    def _cleanup():
        for det in detectors:
            if det.name in saved_ports:
                yield from restore_averaging(det, saved_ports[det.name])
        # After a completed scan the detector's unstage has already
        # restarted the live view; but a failure BEFORE staging (e.g. in
        # configure_averaging) leaves the camera stopped by _body's
        # stop-preview. Restart any camera found stopped so the session
        # stays usable without manual intervention.
        for det in detectors:
            acquiring = yield from bps.rd(det.driver.acquire)
            if not acquiring:
                yield from bps.wait_for([det.start_live_view])

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))
