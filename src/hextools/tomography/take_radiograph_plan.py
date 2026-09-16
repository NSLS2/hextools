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
Each burst is a single ``bps.trigger_and_read`` on the camera's internal
trigger, with ``num_images`` set to ``frames_per_burst`` so one trigger fires
the whole burst. The plan owns the timing directly: ``acquire_time`` is set to
``exposure_time`` and ``acquire_period`` to ``frame_period``, so
``frame_period - exposure_time`` is the readout margin that keeps frames
non-overlapping — the same "period larger than exposure" discipline the old
PandA-paced script enforced with its PULSE step. A PandA-paced external-trigger
variant remains possible if precision frame timing is ever needed.

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

from  bluesky import plan_stubs as bps, plans as bp
import bluesky.preprocessors as bpp
from nslsii import detectors
from ophyd_async.epics.adcore import AreaDetector
from ophyd_async.core import DetectorTrigger, TriggerInfo
from hextools.utils import get_obj_from_ipython_ns

from hextools.photon_delivery_system import Shutter

# Readout headroom (s) added to exposure_time when frame_period is unset;
# same margin the beamline's deployed PandA plan kept between step and exposure.
FRAME_PERIOD_MARGIN = 0.1

# @bpp.stage_decorator([])
# @bpp.run_decorator()
def take_radiograph(
    detectors: list[AreaDetector],
    exposure_time: float,
    external_trigger: bool = False,
    frames_per_burst: int = 10, # image_number
    num_bursts: int = 1, # Iteration
    wait_between_bursts: float = 0.0, # Sleep time between bursts
    frame_period: float | None = None, # acquire period
    sample_name: str | None = None,
    md: dict | None = None,
    use_shutter: bool = False,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
):
    """Acquire a burst-mode radiograph series on the HEX beamline.

    Parameters
    ----------
    detectors : list[AreaDetector]
        detectors to trigger; any ophyd-async detector is accepted
    fe_shutter : Shutter
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

    if fe_shutter is None:
        fe_shutter = get_obj_from_ipython_ns("fe_shutter", Shutter)
    if photon_shutter is None:
        photon_shutter = get_obj_from_ipython_ns("photon_shutter", Shutter)
    if fe_shutter is None or photon_shutter is None:
        raise ValueError(
            "Both fe_shutter and photon_shutter must be specified or available in the IPython namespace."
        )

    # Validate arguments before touching hardware.
    if frame_period is None:
        frame_period = exposure_time + FRAME_PERIOD_MARGIN
    if frame_period <= exposure_time:
        raise ValueError(
            f"frame_period ({frame_period}) must be larger than exposure_time "
            f"({exposure_time}) to leave readout margin."
        )

    trigger_info = TriggerInfo(
        trigger=DetectorTrigger.EXTERNAL_EDGE if external_trigger else DetectorTrigger.INTERNAL,
        livetime=exposure_time,
        deadtime=frame_period - exposure_time,
        exposures_per_collection=1,
        collections_per_event=frames_per_burst,
        number_of_events=1,
    )

    if use_shutter:
        # FE shutter must already be open; this plan never actuates it.
        fe_shutter_open = yield from bps.rd(fe_shutter.status)
        if not fe_shutter_open:
            raise ValueError(
                "Front-end shutter is closed. Please open it before starting the scan."
            )

    def _body():
        if use_shutter:
            photon_shutter_open = yield from bps.rd(photon_shutter.status)
            if not photon_shutter_open:
                yield from bps.mv(photon_shutter, True)

        _md = {
            "detectors": [det.name for det in detectors],
            "num_points": num_bursts,
            "plan_name": "take_radiograph",
            "hints": {},
            # burst structure — lets analysis reconstruct the timing
            "frames_per_burst": frames_per_burst,
            "num_bursts": num_bursts,
            "wait_between_bursts": wait_between_bursts,
            "frame_period": frame_period,
            "exposure_time": exposure_time,
        }

        if sample_name is not None:
            _md["sample_name"] = sample_name
        _md.update(md or {})

        # for burst in range(num_bursts):
        #     yield from bps.trigger_and_read(detectors)
        #     if burst < num_bursts - 1:
        #         yield from bps.sleep(wait_between_bursts)

        for det in detectors:
            yield from bps.prepare(det, trigger_info, group="prepare")
        yield from bps.wait(group="prepare")

        yield from bp.count(detectors, num_bursts, delay=wait_between_bursts, md=_md)


    def _cleanup():
        if use_shutter:
            yield from bps.mv(photon_shutter, False)

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))
