"""XRD acquisition plans for the HEX beamline.

Thin wrappers over :func:`~hextools.tomography.radiography.take_radiograph` and
``bluesky.plans.scan`` that add XRD metadata, shutter handling and the Perkin
Elmer's continuous-acquisition exposure setup.
"""

from collections.abc import Generator, Mapping, Sequence
from typing import Any

from bluesky import Msg
from bluesky import plan_stubs as bps
from bluesky import plans as bp
from bluesky import preprocessors as bpp
from bluesky.protocols import Movable, Readable
from ophyd_async.core import DetectorTrigger, TriggerInfo
from ophyd_async.epics.adcore import ADImageMode, AreaDetector, ContAcqDetector

from hextools.photon_delivery_system import Shutter
from hextools.photon_delivery_system.shutter import (
    ensure_shutter_closed,
    ensure_shutter_open,
)
from hextools.tomography.radiography import take_radiograph
from hextools.utils import Steppable, ensure_available


def _resolve_detector(detector: ContAcqDetector | None) -> ContAcqDetector:
    if detector is not None:
        return detector
    return ensure_available(ContAcqDetector, perkin_elmer=None)


def _with_plan_name(plan: Generator[Msg, Any, Any], plan_name: str):
    """Run ``plan`` with ``plan_name`` in each of its start documents."""

    def _rename(msg: Msg) -> Msg:
        if msg.command == "open_run":
            return msg._replace(kwargs={**msg.kwargs, "plan_name": plan_name})
        return msg

    return (yield from bpp.msg_mutator(plan, _rename))


def ensure_cont_acq_exposure(detector: AreaDetector, exposure_time: float):
    """Have a continuous-acquisition detector acquiring at ``exposure_time``.

    A ``ContAcqDetector`` only checks the exposure time it is prepared with, so
    the driver is stopped, set and restarted here, as the pyepics scripts did.
    Other detectors set their exposure when prepared, so this does nothing.
    """
    if not isinstance(detector, ContAcqDetector):
        return
    acquiring = yield from bps.rd(detector.driver.acquire)
    acquire_time = yield from bps.rd(detector.driver.acquire_time)
    image_mode = yield from bps.rd(detector.driver.image_mode)
    if (
        acquiring
        and acquire_time == exposure_time
        and image_mode == ADImageMode.CONTINUOUS
    ):
        return
    if acquiring:
        yield from bps.abs_set(detector.driver.acquire, False, wait=True)
    yield from bps.mv(
        detector.driver.acquire_time,
        exposure_time,
        detector.driver.image_mode,
        ADImageMode.CONTINUOUS,
    )
    # A busy record: waiting on it would block until acquisition stops.
    yield from bps.abs_set(detector.driver.acquire, True, wait=False)


def xrd_count(
    exposure_time: float,
    num_images: int = 1,
    images_to_average: int = 1,
    num_iterations: int = 1,
    time_gap: float = 0.0,
    sample_name: str | None = None,
    description: str | None = None,
    use_shutter: bool = True,
    detector: ContAcqDetector | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
):
    """Take XRD images without moving anything.

    Parameters
    ----------
    exposure_time : float
        exposure time per frame, in seconds
    num_images : int
        number of images saved per run
    images_to_average : int
        number of frames averaged into each saved image
    num_iterations : int
        number of runs to take, each its own file
    time_gap : float
        idle time between runs, in seconds
    sample_name : str, optional
        name of the sample
    description : str, optional
        description of the acquisition
    use_shutter : bool
        whether to open the photon shutter for the acquisition
    detector : ContAcqDetector, optional
        the XRD detector; the ``perkin_elmer`` in the session if not given
    fe_shutter : Shutter, optional
        the front-end shutter to check before opening the photon shutter
    photon_shutter : Shutter, optional
        the photon shutter to open and close around the acquisition
    """
    detector = _resolve_detector(detector)
    yield from ensure_cont_acq_exposure(detector, exposure_time)
    return (
        yield from _with_plan_name(
            take_radiograph(
                [detector],
                exposure_time,
                num_images=num_images,
                num_iterations=num_iterations,
                external_trigger=False,
                time_gap=time_gap,
                images_to_average=images_to_average,
                sample_name=sample_name,
                description=description,
                use_shutter=use_shutter,
                fe_shutter=fe_shutter,
                photon_shutter=photon_shutter,
            ),
            "xrd_count",
        )
    )


def xrd_dark(
    exposure_time: float,
    num_images: int = 1,
    images_to_average: int = 1,
    sample_name: str | None = None,
    description: str | None = None,
    detector: ContAcqDetector | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
):
    """Take XRD dark images with the photon shutter closed.

    Parameters
    ----------
    exposure_time : float
        exposure time per frame, in seconds; use the one the data will use
    num_images : int
        number of dark images saved
    images_to_average : int
        number of frames averaged into each saved image
    sample_name : str, optional
        name of the sample
    description : str, optional
        description of the acquisition
    detector : ContAcqDetector, optional
        the XRD detector; the ``perkin_elmer`` in the session if not given
    fe_shutter : Shutter, optional
        the front-end shutter (not opened for darks)
    photon_shutter : Shutter, optional
        the photon shutter, closed before the darks are taken
    """
    detector = _resolve_detector(detector)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)
    yield from ensure_shutter_closed(photon_shutter, allow_actuation=True)
    yield from ensure_cont_acq_exposure(detector, exposure_time)
    return (
        yield from _with_plan_name(
            take_radiograph(
                [detector],
                exposure_time,
                num_images=num_images,
                external_trigger=False,
                images_to_average=images_to_average,
                sample_name=sample_name,
                description=description,
                use_shutter=False,
                fe_shutter=fe_shutter,
                photon_shutter=photon_shutter,
            ),
            "xrd_dark",
        )
    )


def xrd_scan(
    motor: Steppable,
    start: float,
    stop: float,
    num_points: int,
    exposure_time: float,
    images_to_average: int = 1,
    num_iterations: int = 1,
    time_between_iterations: float = 0.0,
    sample_name: str | None = None,
    description: str | None = None,
    reset_position: bool = True,
    use_shutter: bool = True,
    detector: ContAcqDetector | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
):
    """Step one motor and take an XRD image at each point.

    Parameters
    ----------
    motor : Steppable
        the motor to step
    start : float
        first motor position
    stop : float
        last motor position
    num_points : int
        number of points, including start and stop
    exposure_time : float
        exposure time per frame, in seconds
    images_to_average : int
        number of frames averaged into the image saved at each point
    num_iterations : int
        number of times to repeat the scan, each its own run
    time_between_iterations : float
        idle time between repeats, in seconds
    sample_name : str, optional
        name of the sample
    description : str, optional
        description of the scan
    reset_position : bool
        whether to move the motor back to ``start`` afterwards
    use_shutter : bool
        whether to open the photon shutter for the scan
    detector : ContAcqDetector, optional
        the XRD detector; the ``perkin_elmer`` in the session if not given
    fe_shutter : Shutter, optional
        the front-end shutter to check before opening the photon shutter
    photon_shutter : Shutter, optional
        the photon shutter to open and close around the scan
    """
    detector = _resolve_detector(detector)
    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)
    trigger_info = TriggerInfo(
        trigger=DetectorTrigger.INTERNAL,
        livetime=exposure_time,
        exposures_per_collection=images_to_average,
    )

    def _per_step(
        detectors: Sequence[Readable],
        step: Mapping[Movable, Any],
        pos_cache: dict[Movable, Any],
        take_reading: bps.TakeReading | None = None,
    ):
        # stage() at the start of each run clears what prepare() set
        for det in detectors:
            yield from bps.prepare(det, trigger_info, wait=True)
        return (
            yield from bps.one_nd_step(
                detectors,
                step,
                pos_cache,
                take_reading=take_reading or bps.trigger_and_read,
            )
        )

    _md: dict[str, Any] = {"plan_name": "xrd_scan"}
    if sample_name is not None:
        _md["sample_name"] = sample_name
    if description is not None:
        _md["description"] = description

    def _body():
        if use_shutter:
            yield from ensure_shutter_open(fe_shutter)
            yield from ensure_shutter_open(photon_shutter, allow_actuation=True)
        yield from ensure_cont_acq_exposure(detector, exposure_time)
        yield from bp.scan(
            [detector], motor, start, stop, num_points, per_step=_per_step, md=_md
        )

    def _cleanup():
        if use_shutter:
            yield from ensure_shutter_closed(photon_shutter, allow_actuation=True)
        if reset_position:
            yield from bps.mv(motor, start)

    return (
        yield from bps.repeat(
            lambda: bpp.finalize_wrapper(_body(), _cleanup()),
            num=num_iterations,
            delay=time_between_iterations,
        )
    )
