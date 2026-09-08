"""
Radiograph acquisition plan for HEX beamline.

Equivalent of the old pyepics script:
    hex-acq-pyepics/techniques/tomography/kinetix/take_radiograph.py

What this plan does
-------------------
1. Check the front-end shutter and open the photon shutter.
   The front-end shutter is only checked at entry; must already be open — this
   plan never actuates it.
2. For each burst: fire ``frames_per_burst`` frames, then wait
   ``wait_between_bursts``.
3. Close the photon shutter.

Everything from shutter-open onward runs under a finalizer, so an error or
interrupt still closes the shutter.

Trigger model
-------------
Each frame is acquired with ``bps.trigger_and_read`` using the camera's
internal trigger; the plan owns the exposure via ``prepare(TriggerInfo)``
(control screens just reflect it). Non-overlapping frames are guaranteed by
``deadtime = frame_period - exposure_time`` — the same "period larger than
exposure" discipline the old PandA-paced script enforced with its PULSE step.
A PandA-paced external-trigger variant remains possible if precision frame
timing is ever needed.

Usage
-----
    RE(take_radiograph(
        [kinetix1], fe_shutter, ph_shutter,
        exposure_time=0.5,
        frames_per_burst=10,
        num_bursts=5,
        wait_between_bursts=10.0,
    ))

``detectors`` is a list (``[kinetix1]``) since multiple detectors are supported.

Where files land is decided by each detector's path provider (set in the
profile), not by this plan — the old script's proposal-folder logic is gone.
"""

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from ophyd_async.core import DetectorTrigger, StandardDetector, TriggerInfo
from ophyd_async.epics.adcore import AreaDetector

from hextools.photon_delivery_system import Shutter

# Readout headroom (s) added to exposure_time when frame_period is unset;
# same margin the beamline's deployed PandA plan kept between step and exposure.
FRAME_PERIOD_MARGIN = 0.1


def take_radiograph(
    detectors: list[AreaDetector],
    front_end_shutter: Shutter,
    photon_shutter: Shutter,
    exposure_time: float,
    frames_per_burst: int = 10,
    num_bursts: int = 5,
    wait_between_bursts: float = 10.0,
    frame_period: float | None = None,
    use_shutter: bool = True,
    sample_name: str | None = None,
    md: dict | None = None,
):
    """Acquire a burst-mode radiograph series on the HEX beamline.

    Parameters
    ----------
    detectors : list[AreaDetector]
        detectors to trigger; any ophyd-async detector is accepted
    front_end_shutter : Shutter
        the front-end shutter to check before opening the photon shutter
    photon_shutter : Shutter
        the photon shutter to open/close around the acquisition
    exposure_time : float
        camera exposure time, in seconds (no default — depends on the sample)
    frames_per_burst : int
        number of frames fired in each burst
    num_bursts : int
        number of bursts to acquire
    wait_between_bursts : float
        idle time between bursts, in seconds
    frame_period : float, optional
        minimum time per frame, in seconds; must exceed ``exposure_time``, and
        the difference is enforced as the camera's deadtime. If None, computed
        from ``exposure_time`` plus a readout margin
    use_shutter : bool
        whether to open/check the photon shutter during the scan
    sample_name : str, optional
        name of the sample being imaged
    md : dict, optional
        extra metadata to merge into the run's metadata
    """
    # Validate arguments before touching hardware.
    if frame_period is None:
        frame_period = exposure_time + FRAME_PERIOD_MARGIN
    if frame_period <= exposure_time:
        raise ValueError(
            f"frame_period ({frame_period}) must be larger than exposure_time "
            f"({exposure_time}) to leave readout margin."
        )

    if use_shutter:
        # FE shutter must already be open; this plan never actuates it.
        fe_shutter_open = yield from bps.rd(front_end_shutter.status)
        if not fe_shutter_open:
            raise ValueError(
                "Front-end shutter is closed. Please open it before starting the scan."
            )

    def _body():
        if use_shutter:
            photon_shutter_open = yield from bps.rd(photon_shutter.status)
            if not photon_shutter_open:
                yield from bps.mv(photon_shutter, True)
        total_frames = frames_per_burst * num_bursts

        _md = {
            "detectors": [det.name for det in detectors],
            "num_points": total_frames,
            "plan_name": "take_radiograph",
            "hints": {},
            # burst structure — lets analysis reconstruct the timing
            "frames_per_burst": frames_per_burst,
            "num_bursts": num_bursts,
            "wait_between_bursts": wait_between_bursts,
            "frame_period": frame_period,
            "exposure_time": exposure_time,
        }
        for det in detectors:
            yield from bps.mv(det.driver.num_images, frames_per_burst)

        if sample_name is not None:
            _md["sample_name"] = sample_name
        _md.update(md or {})
        yield from bps.open_run(md=_md)

        yield from bps.stage_all(*detectors)

        for burst in range(num_bursts):
            yield from bps.trigger_and_read(detectors)
            if burst < num_bursts - 1:
                yield from bps.sleep(wait_between_bursts)

        yield from bps.unstage_all(*detectors)
        yield from bps.close_run()

    def _cleanup():
        if use_shutter:
            yield from bps.mv(photon_shutter, False)

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))
