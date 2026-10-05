"""Operator-triggered camera captures for the HEX beamline."""

from typing import Annotated as A

from bluesky import plan_stubs as bps
from bluesky import plans as bp
from bluesky import preprocessors as bpp
from ophyd_async.core import DetectorTrigger, TriggerInfo

from hextools.detectors.phantom import PhantomDetector


def phantom_capture(
    camera: PhantomDetector,
    num_images: int,
    exposure_time: A[float, "s"],
    trigger_timeout: A[float, "s"] = 600,
    sample_name: str | None = None,
    stream_name: str = "primary",
):
    """Arm the Phantom, wait for the operator's event trigger, then read out.

    Once armed the camera records into its cine. The operator watches the live
    image and sends the event trigger at the moment that matters: the HEX GUI's
    Trigger button, a hardware trigger, or the camera's own screen. The camera then
    records its post-trigger frames
    and the first ``num_images`` of them are downloaded into this run. The plan
    waits in ``complete`` meanwhile, so the RunEngine shows it as running.

    Parameters
    ----------
    camera : PhantomDetector
        The Phantom to capture with.
    num_images : int
        Post-trigger images to download into the run, at least 1.
    exposure_time : float
        Exposure time, in seconds.
    trigger_timeout : float, optional
        How long to wait for the event trigger, in seconds. Default 600.
    sample_name : str, optional
        Recorded in the run's metadata.
    stream_name : str, optional
        Name of the event stream the images are recorded in. Default "primary".
    """
    if not isinstance(camera, PhantomDetector):
        raise TypeError(f"camera: {getattr(camera, 'name', camera)!r} is not a Phantom")
    info = TriggerInfo(
        number_of_events=num_images,
        livetime=exposure_time,
        trigger=DetectorTrigger.INTERNAL,
        exposure_timeout=trigger_timeout,
    )
    md = {
        "detectors": [camera.name],
        "num_points": num_images,
        "plan_name": "phantom_capture",
    }
    if sample_name is not None:
        md["sample_name"] = sample_name

    @bpp.stage_decorator([camera])
    def _capture():
        yield from bps.prepare(camera, info, wait=True)
        return (
            yield from bp.fly(
                [camera], md=md, collect_flush_period=1.0, stream_name=stream_name
            )
        )

    return (yield from _capture())
