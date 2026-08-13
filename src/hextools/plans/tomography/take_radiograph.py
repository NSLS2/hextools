"""
Radiograph acquisition plan for HEX beamline.

Equivalent of the old pyepics script:
    hex-acq-pyepics/techniques/tomography/kinetix/take_radiograph.py

What this plan does
-------------------
1.  Opens the photon shutter.  (Front-end status check not yet ported — see
    hextools/plans/shutter.py.)
2.  Acquires *num_images* frames at a fixed sample position into the
    ``primary`` event stream.
3.  Closes the photon shutter (unless *keep_shutter_open* is True).

Everything from shutter-open onward runs under a finalizer, so an error or
interrupt still closes the shutter.

Trigger model
-------------
The old script paced frames with a PandA PULSE2 train (external trigger,
BITS:A gate).  This port uses the camera's internal trigger with the same
exposure/period — functionally equivalent frames for a static radiograph;
PandA-paced acquisition arrives with the tomo_flyscan port.  Frame pacing:
``prepare(TriggerInfo(livetime=exposure_time, deadtime=...))``.

Output: one HDF file per scan under *output_dir* (see alignment_scan for the
one-file-per-scan note vs. the old per-frame files).

Usage
-----
    RE(take_radiograph(
        kinetix1, ph_open_cmd, ph_close_cmd,
        output_dir=".../radiograph/scan_00001",
        exposure_time=0.05,
        num_images=10,
    ))
"""

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from ophyd_async.core import TriggerInfo

from hextools.detectors.kinetix import set_output_dir
from hextools.plans.shutter import close_photon_shutter, open_photon_shutter


def take_radiograph(
    detector,
    ph_open_cmd,
    ph_close_cmd,
    *,
    output_dir: str,
    exposure_time: float,
    num_images: int = 1,
    acquire_period: float | None = None,
    keep_shutter_open: bool = False,
    md: dict | None = None,
):
    """
    Take *num_images* radiographs at the current sample position.

    Parameters
    ----------
    detector : HEXKinetixDetector
        Built by ``hextools.detectors.kinetix.make_kinetix`` (e.g. kinetix1 or kinetix3).
    ph_open_cmd / ph_close_cmd : signal
        Photon-shutter command signals (write 1 to actuate).
    output_dir : str
        Directory for the HDF file (the old script's scan_NNNNN folder).
    exposure_time : float
        Camera exposure time in seconds.
    num_images : int, optional
        Number of frames. Default 1.
    acquire_period : float, optional
        Frame period in seconds; defaults to exposure_time + a small
        overhead (the old script's behavior when period < exposure).
    keep_shutter_open : bool, optional
        Leave the shutter open afterwards. Default False.
    md : dict, optional
        Extra run metadata.
    """
    output_path = set_output_dir(detector, output_dir, "img")

    deadtime = 0.001
    if acquire_period is not None and acquire_period > exposure_time:
        deadtime = acquire_period - exposure_time

    _md = {
        "plan_name": "take_radiograph",
        "detectors": [detector.name],
        "output_dir": str(output_path),
        "exposure_time": exposure_time,
        "num_images": num_images,
    }
    _md.update(md or {})

    @bpp.stage_decorator([detector])
    @bpp.run_decorator(md=_md)
    def _inner():
        yield from bps.prepare(
            detector,
            TriggerInfo(livetime=exposure_time, deadtime=deadtime),
            wait=True,
        )
        print(f"  Acquiring {num_images} radiograph(s) -> {output_path}")
        for _ in range(num_images):
            yield from bps.trigger_and_read([detector])

    def _body():
        yield from open_photon_shutter(ph_open_cmd)
        yield from _inner()

    def _cleanup():
        if not keep_shutter_open:
            yield from close_photon_shutter(ph_close_cmd)

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))
