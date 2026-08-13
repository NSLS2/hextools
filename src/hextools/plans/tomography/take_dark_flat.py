"""
Dark + flat-field acquisition plan for HEX beamline.

Equivalent of the old pyepics script:
    hex-acq-pyepics/techniques/tomography/kinetix/take_dark_flat.py

What this plan does
-------------------
1.  Darks: closes the photon shutter, acquires *num_dark* frames into the
    ``dark`` event stream.
2.  Flats: opens the shutter, moves sample_x out by *flat_x_offset*,
    acquires *num_flat* frames into the ``flat`` event stream, moves the
    sample back.
3.  Closes the photon shutter when finished (unless *keep_shutter_open*).

The sample move-back and shutter close run under a finalizer, so an error or
interrupt still restores the sample position and closes the shutter.

Differences from the old script (flagged for beamline review)
-------------------------------------------------------------
- Internal trigger instead of the PandA PULSE2 train (same exposure/period;
  PandA pacing arrives with the tomo_flyscan port).
- One HDF file per scan: darks and flats are separate *event streams*
  (``dark`` / ``flat``) of the same run, not separate dark_*.h5 / flat_*.h5
  files.  The .nxs metadata sidecar is not written by the plan — metadata
  lives in the run documents (Tiled); NeXus export is downstream tooling.

Usage
-----
    RE(take_dark_flat(
        kinetix1, sample_x, ph_open_cmd, ph_close_cmd,
        output_dir=".../raw_data/scan_00001",
        exposure_time=0.05,
        num_dark=20, num_flat=50,
        flat_x_offset=5.0,
    ))
"""

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from ophyd_async.core import TriggerInfo

from hextools.detectors.kinetix import set_output_dir
from hextools.plans.shutter import close_photon_shutter, open_photon_shutter

OVERHEAD = 0.05  # same frame-period overhead as the old script


def take_dark_flat(
    detector,
    sample_x,
    ph_open_cmd,
    ph_close_cmd,
    *,
    output_dir: str,
    exposure_time: float,
    num_dark: int = 20,
    num_flat: int = 50,
    flat_x_offset: float,
    keep_shutter_open: bool = False,
    md: dict | None = None,
):
    """
    Collect dark and flat-field images.

    Parameters
    ----------
    detector : HEXKinetixDetector
        Built by ``hextools.detectors.kinetix.make_kinetix`` (e.g. kinetix1 or kinetix3).
    sample_x : Motor
        Sample X-axis, moved by *flat_x_offset* to take flats out of beam.
    ph_open_cmd / ph_close_cmd : signal
        Photon-shutter command signals (write 1 to actuate).
    output_dir : str
        Directory for the HDF file (the old script's scan_NNNNN folder).
    exposure_time : float
        Camera exposure time in seconds.
    num_dark / num_flat : int, optional
        Frame counts; either may be 0 to skip that half. Defaults 20 / 50.
    flat_x_offset : float
        Relative X move (mm) that takes the sample out of the beam.
    keep_shutter_open : bool, optional
        Leave the shutter open afterwards. Default False.
    md : dict, optional
        Extra run metadata.
    """
    output_path = set_output_dir(detector, output_dir, "dark_flat")

    _md = {
        "plan_name": "take_dark_flat",
        "detectors": [detector.name],
        "output_dir": str(output_path),
        "exposure_time": exposure_time,
        "num_dark": num_dark,
        "num_flat": num_flat,
        "flat_x_offset": flat_x_offset,
    }
    _md.update(md or {})

    # Set inside _body once the sample has actually moved, read by _cleanup —
    # so an early failure doesn't "move back" a sample that never moved out.
    moved_out = False

    @bpp.stage_decorator([detector])
    @bpp.run_decorator(md=_md)
    def _inner():
        nonlocal moved_out
        yield from bps.prepare(
            detector,
            TriggerInfo(livetime=exposure_time, deadtime=OVERHEAD),
            wait=True,
        )

        if num_dark > 0:
            print("-" * 52)
            print(f"  Shutter closed — acquiring {num_dark} dark image(s)")
            yield from close_photon_shutter(ph_close_cmd)
            for _ in range(num_dark):
                yield from bps.trigger_and_read([detector], name="dark")

        if num_flat > 0:
            print("-" * 52)
            print(f"  Moving sample out ({flat_x_offset:+g}) — "
                  f"acquiring {num_flat} flat image(s)")
            yield from open_photon_shutter(ph_open_cmd)
            yield from bps.mvr(sample_x, flat_x_offset)
            moved_out = True
            for _ in range(num_flat):
                yield from bps.trigger_and_read([detector], name="flat")

    def _cleanup():
        if moved_out:
            print("  Moving sample back...")
            yield from bps.mvr(sample_x, -flat_x_offset)
        if not keep_shutter_open:
            yield from close_photon_shutter(ph_close_cmd)

    return (yield from bpp.finalize_wrapper(_inner(), _cleanup()))
