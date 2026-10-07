"""Radiograph acquisition plan for the HEX beamline."""

from bluesky import plan_stubs as bps, plans as bp
import bluesky.preprocessors as bpp
from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.epics.adcore import ContAcqDetector
from ophyd_async.core import DetectorTrigger, TriggerInfo
from hextools.detectors import PhantomDetector
from hextools.photon_delivery_system.shutter import ensure_shutter_closed, ensure_shutter_open
from hextools.utils import ensure_available
    
from hextools.photon_delivery_system import Shutter

from hextools.detectors import FRAME_PERIOD_MARGIN


def take_radiograph(
    detectors: list[KinetixDetector | PhantomDetector | ContAcqDetector],
    exposure_time: float,
    num_images: int,
    num_acquisitions: int = 1,
    acquire_period: float = 0.0,
    external_trigger: bool = False,
    time_gap: float = 0.0,
    num_exposures: int = 1,
    sample_name: str | None = None,
    description: str | None = None,
    use_shutter: bool = False,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
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
    time_gap : float
        idle time between acquisitions, in seconds
    sample_name : str, optional
        name of the sample being imaged
    description : str, optional
        description of the acquisition
    use_shutter : bool
        whether to open/check the photon shutter during the scan
    fe_shutter : Shutter
        the front-end shutter to check before opening the photon shutter
    photon_shutter : Shutter
        the photon shutter to open/close around the acquisition
    """

    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)

    if acquire_period <= exposure_time:
        acquire_period = exposure_time + FRAME_PERIOD_MARGIN

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

    def _body():

        if use_shutter:
            yield from ensure_shutter_open(fe_shutter)
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
        if description is not None:
            _md["description"] = description

        yield from bp.count(
            detectors, num_acquisitions, delay=time_gap, md=_md
        )

    def _cleanup():
        yield from ensure_shutter_closed(photon_shutter, allow_actuation=True)

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))
