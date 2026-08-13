"""
1-D motor step-scan plan for HEX beamline.

Equivalent of the old pyepics script:
    hex-acq-pyepics/techniques/tomography/kinetix/scan_1d.py

What this plan does
-------------------
1.  Opens the photon shutter.  (Front-end status check not yet ported — see
    hextools/plans/shutter.py.)
2.  Steps *motor* through linspace(start, stop, num_points); at each point
    optionally rests *rest_time* seconds, then takes one image into the
    ``primary`` event stream (motor readback recorded per point).
3.  Closes the photon shutter (unless *keep_shutter_open* is True).

The motor is any movable passed in — the old script's "--motor <PV>"
becomes passing the device (start == stop repeats num_points images at one
position, like the original).  As in the original, the motor is left at its
final position (no restore); alignment_scan is the restoring variant.

Trigger model: internal (see take_radiograph docstring for the PandA note).

Usage
-----
    RE(scan_1d(
        kinetix1, sample_x, ph_open_cmd, ph_close_cmd,
        output_dir=".../raw_data/scan_00001",
        exposure_time=0.05,
        start=-1.0, stop=1.0, num_points=21,
        rest_time=0.5,
    ))
"""

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
import numpy as np
from ophyd_async.core import TriggerInfo

from hextools.detectors.kinetix import set_output_dir
from hextools.plans.shutter import close_photon_shutter, open_photon_shutter


def scan_1d(
    detector,
    motor,
    ph_open_cmd,
    ph_close_cmd,
    *,
    output_dir: str,
    exposure_time: float,
    start: float,
    stop: float,
    num_points: int,
    rest_time: float = 0.0,
    keep_shutter_open: bool = False,
    md: dict | None = None,
):
    """
    Step *motor* from *start* to *stop*, one image per point.

    Parameters
    ----------
    detector : HEXKinetixDetector
        Built by ``hextools.detectors.kinetix.make_kinetix`` (e.g. kinetix1 or kinetix3).
    motor : Movable
        The motor to scan (any ophyd-async Motor — the old script took a PV).
    ph_open_cmd / ph_close_cmd : signal
        Photon-shutter command signals (write 1 to actuate).
    output_dir : str
        Directory for the HDF file (the old script's scan_NNNNN folder).
    exposure_time : float
        Camera exposure time in seconds.
    start / stop : float
        Scan range; start == stop repeats images at one position.
    num_points : int
        Number of positions (must be > 1 unless start == stop).
    rest_time : float, optional
        Settle time after each move, seconds. Default 0.
    keep_shutter_open : bool, optional
        Leave the shutter open afterwards. Default False.
    md : dict, optional
        Extra run metadata.
    """
    if num_points < 1:
        raise ValueError(f"num_points must be >= 1, got {num_points}")
    if start == stop:
        positions = np.full(num_points, start)
    else:
        if num_points <= 1:
            raise ValueError(
                "num_points must be > 1 when start != stop "
                "(one image per position)."
            )
        positions = np.linspace(start, stop, num_points)

    output_path = set_output_dir(detector, output_dir, "img")

    _md = {
        "plan_name": "scan_1d",
        "detectors": [detector.name],
        "motors": [motor.name],
        "output_dir": str(output_path),
        "exposure_time": exposure_time,
        "start": start,
        "stop": stop,
        "num_points": num_points,
        "rest_time": rest_time,
        "hints": {"dimensions": [([motor.name], "primary")]},
    }
    _md.update(md or {})

    @bpp.stage_decorator([detector])
    @bpp.run_decorator(md=_md)
    def _inner():
        yield from bps.prepare(
            detector,
            TriggerInfo(livetime=exposure_time, deadtime=0.002),
            wait=True,
        )
        print(f"  1-D scan of {motor.name}: {start:g} -> {stop:g} "
              f"({num_points} points) -> {output_path}")
        for position in positions:
            yield from bps.mv(motor, float(position))
            if rest_time > 0:
                yield from bps.sleep(rest_time)
            yield from bps.trigger_and_read([detector, motor])

    def _body():
        yield from open_photon_shutter(ph_open_cmd)
        yield from _inner()

    def _cleanup():
        if not keep_shutter_open:
            yield from close_photon_shutter(ph_close_cmd)

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))
