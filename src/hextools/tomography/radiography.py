"""Radiograph acquisition plan for the HEX beamline."""

import bluesky.preprocessors as bpp
from bluesky import plan_stubs as bps
from bluesky import plans as bp
from ophyd_async.core import DetectorTrigger, TriggerInfo
from ophyd_async.epics.adcore import ContAcqDetector
from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.fastcs.panda import HDFPanda

from hextools.detectors import FRAME_PERIOD_MARGIN, PhantomDetector
from hextools.detectors.utils import enable_preview_mode
from hextools.photon_delivery_system import Shutter
from hextools.photon_delivery_system.shutter import (
    ensure_shutter_closed,
    ensure_shutter_open,
)
from hextools.utils import ensure_available


def take_radiograph(
    detectors: list[KinetixDetector | PhantomDetector | ContAcqDetector],
    exposure_time: float,
    num_images: int,
    num_iterations: int = 1,
    acquire_period: float = 0.0,
    external_trigger: bool = True,
    time_gap: float = 0.0,
    images_to_average: int = 1,
    sample_name: str | None = None,
    description: str | None = None,
    use_shutter: bool = True,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
    panda: HDFPanda | None = None,
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
    images_to_average : int
        The number of images to average for each saved image.
    external_trigger : bool
        pace frames with the PandA: each burst arms the PandA and the cameras,
        then raises BITS.B, which starts PULSE1 (``num_images`` x
        ``images_to_average`` pulses, one every ``acquire_period``) on TTLOUT1.
        Records one PandA position capture per frame. Otherwise the cameras use
        their internal trigger
    num_iterations : int
        number of iterations to perform
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
    panda : HDFPanda
        the PandA that triggers the cameras when ``external_trigger`` is set
    """
    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)
    bits = None
    pulse = None
    if external_trigger:
        if any(isinstance(det, ContAcqDetector) for det in detectors):
            raise ValueError("External triggering is not supported for ContAcqDetector")
        panda = ensure_available(HDFPanda, panda=panda)
        bits = panda.bits  # type: ignore[attr-defined]  # discovered from the IOC  # ty: ignore[unresolved-attribute]
        pulse = panda.pulse[1]

    if acquire_period <= exposure_time:
        acquire_period = exposure_time + FRAME_PERIOD_MARGIN

    trigger_info = TriggerInfo(
        trigger=DetectorTrigger.EXTERNAL_EDGE
        if external_trigger
        else DetectorTrigger.INTERNAL,
        livetime=exposure_time,
        deadtime=acquire_period - exposure_time,
        exposures_per_collection=images_to_average,
        collections_per_event=num_images,
        number_of_events=1,
    )
    num_pulses = num_images * images_to_average

    def _body():

        if use_shutter:
            yield from ensure_shutter_open(fe_shutter)
            yield from ensure_shutter_open(photon_shutter, allow_actuation=True)

        if (
            external_trigger
            and panda is not None
            and pulse is not None
            and bits is not None
        ):
            yield from bps.mv(
                bits.b,
                0,
                pulse.pulses,
                num_pulses,
                pulse.step,
                acquire_period,
                pulse.width,
                exposure_time,
            )
            yield from bps.mv(panda.pcap.arm, True)

        # Attach additional metadata
        _md = {
            "plan_name": "take_radiograph",
        }
        if sample_name is not None:
            _md["sample_name"] = sample_name
        if description is not None:
            _md["description"] = description

        def _per_shot(dets, take_reading=None):
            # prepare() arms externally triggered cameras, so it must run every shot
            for det in dets:
                yield from bps.prepare(det, trigger_info, group="prepare")
            yield from bps.wait(group="prepare")

            for det in dets:
                yield from bps.trigger(det, group="trigger")

            # Trigger the pulse block in external trigger mode
            if external_trigger and bits is not None:
                yield from bps.mv(bits.b, True)

            yield from bps.wait(group="trigger")

            # PULSE1 fires on a rising edge, so drop BITS.B before the next shot
            if external_trigger and bits is not None:
                yield from bps.mv(bits.b, False)

            yield from bps.create("primary")
            for det in dets:
                yield from bps.read(det)
            yield from bps.save()

        # Each count is its own run, and its stage() makes prepare() open a new file
        for i in range(num_iterations):
            if i > 0:
                yield from bps.sleep(time_gap)
            yield from bp.count(detectors, num=1, per_shot=_per_shot, md=_md)

    def _cleanup():
        yield from ensure_shutter_closed(photon_shutter, allow_actuation=True)
        if external_trigger and bits is not None:
            yield from bps.mv(bits.b, False)
        if external_trigger and panda is not None:
            yield from bps.mv(panda.pcap.arm, False)
        for det in detectors:
            if isinstance(det, KinetixDetector):
                yield from enable_preview_mode(det)

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))
