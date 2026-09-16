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

from bluesky import plan_stubs as bps, plans as bp
import bluesky.preprocessors as bpp
from nslsii import detectors
from ophyd_async.epics.adcore import AreaDetector
from ophyd_async.core import DetectorTrigger, TriggerInfo
from hextools.photon_delivery_system.shutter import ensure_shutter_closed, ensure_shutter_open
from hextools.utils import ensure_available, get_obj_from_ipython_ns

from hextools.photon_delivery_system import Shutter

from hextools.detectors import FRAME_PERIOD_MARGIN


def take_radiograph(
    detectors: list[AreaDetector],
    exposure_time: float,
    external_trigger: bool = False,
    num_images: int = 10,
    images_to_average: int = 1,
    num_acquisitions: int = 1,  # Iteration
    wait_between_acquisitions: float = 0.0,  # Sleep time between bursts
    frame_period: float | None = None,  # acquire period
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
    num_images : int
        number of images to acquire in each acquisition
    images_to_average : int
        number of images to average for each acquired frame
    frame_period: float | None = None,
        time between exposures, in seconds
    wait_between_acquisitions: float = 0.0,
        idle time between acquisitions, in seconds
    num_acquisitions : int
        number of acquisitions to perform
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

    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)

    # Validate arguments before touching hardware.
    if frame_period is None:
        frame_period = exposure_time + FRAME_PERIOD_MARGIN
    if frame_period <= exposure_time:
        raise ValueError(
            f"frame_period ({frame_period}) must be larger than exposure_time "
            f"({exposure_time}) to leave readout margin."
        )

    trigger_info = TriggerInfo(
        trigger=DetectorTrigger.EXTERNAL_EDGE
        if external_trigger
        else DetectorTrigger.INTERNAL,
        livetime=exposure_time,
        deadtime=frame_period - exposure_time,
        exposures_per_collection=images_to_average,
        collections_per_event=num_images,
        number_of_events=1,
    )

    if use_shutter:
        yield from ensure_shutter_open(fe_shutter)

    def _body():

        if use_shutter:
            yield from ensure_shutter_open(photon_shutter, allow_actuation=True)

        # Prepare all detectors for the upcoming acquisition, with the specified
        # triggering configuration
        for det in detectors:
            yield from bps.prepare(det, trigger_info, group="prepare")
        yield from bps.wait(group="prepare")

        # Attach additional metadata
        _md = {
            "plan_name": "take_radiograph",
        }
        if sample_name is not None:
            _md["sample_name"] = sample_name
        _md.update(md or {})
        yield from bp.count(
            detectors, num_acquisitions, delay=wait_between_acquisitions, md=_md
        )

    def _cleanup():
        if use_shutter:
            ensure_shutter_closed(photon_shutter, allow_actuation=True)

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))
