"""Radiograph acquisition plan for the HEX beamline."""

from bluesky import plan_stubs as bps, plans as bp
import bluesky.preprocessors as bpp
from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.epics.adcore import ContAcqDetector
from ophyd_async.core import DetectorTrigger, TriggerInfo
from ophyd_async.fastcs.panda import HDFPanda
from hextools.detectors import PhantomDetector
from hextools.photon_delivery_system.shutter import ensure_shutter_closed, ensure_shutter_open
from hextools.utils import ensure_available
    
from hextools.photon_delivery_system import Shutter

from hextools.detectors import FRAME_PERIOD_MARGIN


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
    if external_trigger:
        panda = ensure_available(HDFPanda, panda=panda)

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

    def _body():

        if use_shutter:
            yield from ensure_shutter_open(fe_shutter)
            yield from ensure_shutter_open(photon_shutter, allow_actuation=True)

        # Attach additional metadata
        _md = {
            "plan_name": "take_radiograph",
        }
        if sample_name is not None:
            _md["sample_name"] = sample_name
        if description is not None:
            _md["description"] = description

        if external_trigger:
            yield from _panda_paced_bursts(_md)
            return

        # Prepare all detectors for the upcoming acquisition, with the specified
        # triggering configuration
        for det in detectors:
            yield from bps.prepare(det, trigger_info, group="prepare")
        yield from bps.wait(group="prepare")

        yield from bp.count(
            detectors, num_iterations, delay=time_gap, md=_md
        )

    def _panda_paced_bursts(md):
        if panda is None:
            raise ValueError("external_trigger needs a PandA to pace the frames")
        pacer: HDFPanda = panda
        num_pulses = num_images * images_to_average
        pulse = pacer.pulse[1]
        bits = pacer.bits  # type: ignore[attr-defined]  # discovered from the IOC
        # PCAP captures one position per PULSE1 pulse
        panda_info = TriggerInfo(
            trigger=DetectorTrigger.EXTERNAL_LEVEL,
            livetime=exposure_time,
            collections_per_event=num_pulses,
            number_of_events=1,
        )
        devices = [*detectors, pacer]

        @bpp.stage_decorator(devices)
        @bpp.run_decorator(md=md)
        def _bursts():
            # BITS.A enables PCOMP1, the fly-scan start; keep it off
            yield from bps.mv(bits.a, 0, bits.b, 0)
            yield from bps.mv(
                pulse.pulses, num_pulses,
                pulse.step, acquire_period,
                pulse.width, exposure_time,
                pulse.delay, 0,
            )
            for burst in range(num_iterations):
                if burst and time_gap > 0:
                    yield from bps.sleep(time_gap)
                # External-trigger prepare also arms: cameras wait for edges,
                # PCAP goes active
                for det in detectors:
                    yield from bps.prepare(det, trigger_info, group="arm")
                yield from bps.prepare(pacer, panda_info, group="arm")
                yield from bps.wait(group="arm")
                if burst == 0:
                    yield from bps.declare_stream(*devices, name="primary")
                yield from bps.kickoff_all(*devices, wait=True)
                yield from bps.mv(bits.b, 1)
                yield from bps.complete_all(*devices, wait=True)
                yield from bps.collect(*devices, name="primary")
                yield from bps.mv(bits.b, 0)

        def _stop_pulses():
            yield from bps.mv(bits.b, 0)

        return (yield from bpp.finalize_wrapper(_bursts(), _stop_pulses()))

    def _cleanup():
        yield from ensure_shutter_closed(photon_shutter, allow_actuation=True)

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))
