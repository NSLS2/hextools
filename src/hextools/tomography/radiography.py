"""
Radiograph acquisition plan for HEX beamline.

Equivalent of the old pyepics script:
    hex-acq-pyepics/techniques/tomography/kinetix/take_radiograph.py

What this plan does
-------------------
1. Check the front-end shutter and open the photon shutter.
   The front-end shutter is only checked at entry; must already be open — this
   plan never actuates it.
2. For each acquisition: fire ``num_images`` images, then wait
   ``wait_between_acquisitions``.
3. Close the photon shutter.

Everything from shutter-open onward runs under a finalizer, so an error or
interrupt still closes the shutter.

Trigger model
-------------
Each acquisition prepares the detector with a ``TriggerInfo`` capturing
``num_images`` images (each averaged over ``num_exposures`` exposures). The plan
owns the timing directly: ``exposure_time`` sets the livetime and
``acquire_period`` sets the frame period, so ``acquire_period - exposure_time``
is the readout margin (deadtime) that keeps frames non-overlapping — the same
"period larger than exposure" discipline the old PandA-paced script enforced
with its PULSE step. Set ``external_trigger`` to pace frames from an external
edge if precision frame timing is ever needed.

Usage
-----
    RE(take_radiograph(
        [kinetix1],
        exposure_time=0.5,
        num_images=10,
        num_acquisitions=5,
        wait_between_acquisitions=10.0,
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
    exposure_time: float,  # screen: Exposure Time
    acquire_period: float | None = None,  # screen: Acquire Time
    num_images: int = 10,  # screen: Num Images
    num_exposures: int = 1,  # screen: Exp / Image
    external_trigger: bool = False,  # screen: Trigger Mode
    num_acquisitions: int = 1,  # screen: Number of acquisitions
    wait_between_acquisitions: float = 0.0,  # plan-level: idle between repeats
    sample_name: str | None = None,  # Name of the sample being imaged
    md: dict | None = None,  # Extra metadata to merge into the run's metadata
    use_shutter: bool = False,  # Whether to open/check the photon shutter during the scan
    fe_shutter: Shutter | None = None,  # Front-end shutter to check before opening the photon shutter
    photon_shutter: Shutter | None = None,  # Photon shutter to open/close around the acquisition
):
    """Acquire a burst-mode radiograph series on the HEX beamline.

    Parameters
    ----------
    detectors : list[AreaDetector]
        detectors to trigger; any ophyd-async detector is accepted
    exposure_time : float
        camera exposure time, in seconds (no default — depends on the sample)
    acquire_period : float, optional
        minimum time per frame, in seconds; must exceed ``exposure_time``, and
        the difference is enforced as the camera's deadtime. If None, computed
        from ``exposure_time`` plus a readout margin
    num_images : int
        number of images to acquire in each acquisition
    num_exposures : int
        number of exposures to average for each acquired image
    external_trigger : bool
        whether to pace frames from an external edge instead of the camera's
        internal trigger
    num_acquisitions : int
        number of acquisitions to perform
    wait_between_acquisitions : float
        idle time between acquisitions, in seconds
    sample_name : str, optional
        name of the sample being imaged
    md : dict, optional
        extra metadata to merge into the run's metadata
    use_shutter : bool
        whether to open/check the photon shutter during the scan
    fe_shutter : Shutter
        the front-end shutter to check before opening the photon shutter
    photon_shutter : Shutter
        the photon shutter to open/close around the acquisition
    """

    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)

    # Validate arguments before touching hardware.
    if acquire_period is None:
        acquire_period = exposure_time + FRAME_PERIOD_MARGIN
    if acquire_period <= exposure_time:
        raise ValueError(
            f"acquire_period ({acquire_period}) must be larger than exposure_time "
            f"({exposure_time}) to leave readout margin."
        )

    trigger_info = TriggerInfo(
        trigger=DetectorTrigger.EXTERNAL_EDGE
        if external_trigger
        else DetectorTrigger.INTERNAL,
        livetime=exposure_time,
        deadtime=acquire_period - exposure_time,
        exposures_per_collection=num_exposures,
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
            yield from ensure_shutter_closed(photon_shutter, allow_actuation=True)

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))
