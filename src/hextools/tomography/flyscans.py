"""Tomography plans for HEX beamline."""

from collections import defaultdict
from collections.abc import Mapping, Sequence

from bluesky import plan_stubs as bps
from bluesky import plans as bp
from bluesky import preprocessors as bpp
from bluesky.protocols import Collectable, Flyable, HasName, Movable, Readable
from bluesky.utils import CustomPlanMetadata, MsgGenerator, plan
from nslsii import detectors
from ophyd_async.core import AsyncMovable, DetectorTrigger, StandardFlyable, TriggerInfo
from ophyd_async.epics.adkinetix import KinetixDetector, KinetixTriggerMode
from ophyd_async.fastcs.panda import HDFPanda
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor
from typing import Any

from hextools.detectors import FRAME_PERIOD_MARGIN
from hextools.photon_delivery_system import Shutter
from hextools.photon_delivery_system.shutter import (
    ensure_shutter_closed,
    ensure_shutter_open,
)
from hextools.utils import ensure_available

from ..detectors.phantom import PhantomDetector
from ..flyers import SingleAxisFlyableLogic, construct_fly_info_models
from ..motors import RotationMotor


from hextools import flyers


# TODO: This is vendored from bluesky.plans, because we need to be able to fly
# either with or without creating a run.
def fly_stub(
    flyers: list[Flyable],
    *,
    collect_flush_period: float | None = None,
    stream_name: str | None = None,
    watch: Sequence[str] = (),
) -> MsgGenerator[str | None]:
    """
    Perform a fly scan with one or more 'flyers'.

    Parameters
    ----------
    flyers : collection
        objects that support the flyer interface
    md : dict, optional
        metadata
    collect_flush_period : float, optional
        If set, will use `collect_while_completing` with the given flush period
    stream_name : str, optional
        If set, will declare a stream with the given name for all flyers
    watch: set of watch groups, optional
        Additional groups to monitor while collecting from flyers.
        Will only be used if `collect_flush_period` is set.

    Yields
    ------
    msg : Msg
        'kickoff', 'wait', 'complete, 'wait', 'collect' messages

    See Also
    --------
    :func:`bluesky.preprocessors.fly_during_wrapper`
    :func:`bluesky.preprocessors.fly_during_decorator`
    """
    # Extract list of collectable detectors from flyers
    dets = [flyer for flyer in flyers if isinstance(flyer, (Collectable)) and isinstance(flyer, Readable)]

    # If provided, attempt to declare single stream for all collectable detectors
    # note that if set, all detectors must produce the same number of events.
    if stream_name is not None:
        yield from bps.declare_stream(*dets, name=stream_name)

    # Kickoff all flyers
    yield from bps.kickoff_all(*flyers, wait=True)

    # If flush period given, collect while completing.
    if collect_flush_period is not None:
        yield from bps.collect_while_completing(
            flyers, dets, flush_period=collect_flush_period, stream_name=stream_name, watch=watch
        )
    else:
        # Otherwise, wait for all flyers to complete before collecting.
        yield from bps.complete_all(*flyers, wait=True)
        yield from bps.collect_all(*dets, name=stream_name)


def _tomo_fly_stub(
    detectors: list[KinetixDetector | PhantomDetector],
    panda: HDFPanda,
    rot_motor: RotationMotor,
    fe_shutter: Shutter,
    photon_shutter: Shutter,
    single_axis_panda_flyer: StandardFlyable[SingleAxisFlyableLogic, None],
    num_images: int,
    exposure_time: float,
    acquire_period: float | None = None,
    images_to_average: int = 1,
    start: float = 0,
    stop: float = 180,
    stream_name: str = "primary",
    use_shutter: bool = True,
    time_based: bool = True,
):
    """Run a tomography flyscan with the specified parameters.

    Parameters
    ----------
    detectors : list[Flyable]
        list of detectors to be used in the scan
    panda : HDFPanda
        the panda device used for triggering and recording the rotation angle
    rot_motor : RotationMotor
        the rotation stage for tomography
    exposure_time : float
        exposure time to use on the camera(s), in seconds
    num_images : int
        total number of camera images to collect during the scan
    start : float (optional)
        starting point in degrees
    stop : float (optional)
        stopping point in degrees
    lead_angle : float (optional)
        the angle in degrees to be used to move motor to -lead_angle before
        'start_deg' and +lead_angle after 'stop_deg'
    reset_speed : float
        speed of the rotary motor during reset movements, in deg/s
    use_shutter : bool
        whether to use/check the shutter during the scan
    """

    if use_shutter:
        yield from ensure_shutter_open(fe_shutter)


    if acquire_period is None:
        acquire_period = exposure_time + FRAME_PERIOD_MARGIN

    all_detectors = [*detectors, panda]

    # Construct ephemeral flyer for the single axis flyscan
    all_devices: list[Flyable] = [*all_detectors, single_axis_panda_flyer, rot_motor]

    def _body():
        # Get the start position in encoder counts
        encoder_res = yield from bps.rd(rot_motor.encoder_resolution)
        max_velocity = yield from bps.rd(rot_motor.max_velocity)
        current_position = yield from bps.rd(rot_motor.user_readback)

        # TODO: Come up with better way to access panda calc block
        current_enc = yield from bps.rd(panda.calc[1].out)  # type: ignore

        det_trigger_info = TriggerInfo(
            number_of_events=num_images,
            livetime=exposure_time,
            deadtime=acquire_period - exposure_time,
            trigger=DetectorTrigger.EXTERNAL_EDGE,
            exposures_per_collection=images_to_average,
        )

        panda_trigger_info = TriggerInfo(
            number_of_events=num_images,
            livetime=exposure_time,
            trigger=DetectorTrigger.EXTERNAL_LEVEL,
            collections_per_event=images_to_average,
        )

        overhead = (
            acquire_period - exposure_time
            if acquire_period is not None and acquire_period > exposure_time
            else 0
        )

        flyer_info, motor_info = construct_fly_info_models(
            num_pulses=num_images * images_to_average,
            max_exposure_time=exposure_time,
            current_position=current_position,
            current_enc_position=current_enc,
            start_position=start,
            stop_position=stop,
            encoder_resolution=encoder_res,
            max_motor_velocity=max_velocity,
            acq_time_overhead=overhead,
            time_based=time_based,
        )

        if use_shutter:
            yield from ensure_shutter_open(
                photon_shutter, allow_actuation=True, group="prepare", wait=False
            )
        yield from bps.prepare(rot_motor, motor_info, group="prepare")
        yield from bps.prepare(single_axis_panda_flyer, flyer_info, group="prepare")

        for det in detectors:
            yield from bps.prepare(det, det_trigger_info, group="prepare")

        yield from bps.prepare(panda, panda_trigger_info, group="prepare")

        # TODO: Come up with a way to set a timeout automatically based on the
        # motor move to start position time.
        yield from bps.wait(group="prepare")

        yield from fly_stub(
            all_devices,
            collect_flush_period=max(1, exposure_time + overhead),
            stream_name=stream_name,
        )

    def _cleanup():
        """Perform post-scan cleanup, regardless of result."""
        yield from ensure_shutter_closed(photon_shutter, allow_actuation=True)
        yield from bps.abs_set(rot_motor.motor_stop, 1)

    yield from bpp.finalize_wrapper(_body(), _cleanup())

def tomo_flyscan(
    detectors: list[KinetixDetector | PhantomDetector],
    num_images: int,
    exposure_time: float,
    acquire_period: float | None = None,
    images_to_average: int = 1,
    start: float = 0,
    stop: float = 180,
    use_shutter: bool = True,
    sample_name: str | None = None,
    time_based: bool = True,
    stream_name: str = "primary",
    panda: HDFPanda | None = None,
    rot_motor: RotationMotor | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
):

    panda = ensure_available(HDFPanda, panda=panda)
    rot_motor = ensure_available(RotationMotor, rot_motor=rot_motor)
    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)
    single_axis_panda_flyer = SingleAxisFlyableLogic(panda).with_device()

    _md = {
        "detectors": [det.name for det in detectors],
        "num_points": num_images,
        "exposure_time": exposure_time,
        "acquire_period": acquire_period,
        "time_based": time_based,
        "start_position": start,
        "stop_position": stop,
        "images_to_average": images_to_average,
        "plan_name": "tomo_flyscan",
    }
    if sample_name is not None:
        _md["sample_name"] = sample_name


    @bpp.stage_decorator(detectors + [panda, single_axis_panda_flyer, rot_motor])
    @bpp.run_decorator(md=_md)
    def _body():
        yield from _tomo_fly_stub(
            detectors=detectors,
            single_axis_panda_flyer=single_axis_panda_flyer,
            num_images=num_images,
            exposure_time=exposure_time,
            acquire_period=acquire_period,
            images_to_average=images_to_average,
            start=start,
            stop=stop,
            use_shutter=use_shutter,
            time_based=time_based,
            stream_name=stream_name,
            panda=panda,
            rot_motor=rot_motor,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
        )

    def _cleanup():
        yield from ensure_shutter_closed(photon_shutter)
        yield from bps.abs_set(rot_motor.motor_stop, 1)

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))


def tomo_nd_step_scan(
    detectors: list[KinetixDetector | PhantomDetector],
    *args,
    num_images: int,
    exposure_time: float,
    acquire_period: float | None = None,
    images_to_average: int = 1,
    start: float = 0,
    stop: float = 180,
    alternate_flyscan_dir: bool = False,
    use_shutter: bool = True,
    sample_name: str | None = None,
    time_based: bool = True,
    stream_name: str = "primary",
    snake_axes: bool | None = False,
    panda: HDFPanda | None = None,
    rot_motor: RotationMotor | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
    md: dict | None = None,
):
    """Run a tomography flyscan at each point of an N-dimensional step grid.

    Parameters
    ----------
    detectors : list[Flyable]
        detectors used for the tomography flyscan at each grid point
    ``*args``
        step axes patterned like ``(motor1, start1, stop1, num1, motor2,
        start2, stop2, num2, ...)`` with the outer (slowest) axis first,
        matching :func:`bluesky.plans.grid_scan`
    num_images : int
        The number of images to acquire at each step.
    exposure_time : float
        The exposure time for each image.
    acquire_period : float | None, optional
        The period between consecutive image acquisitions. If None, it defaults to the exposure time.
    images_to_average : int, optional
        The number of images to average at each step.
    start : float, optional
        The starting angle for the rotation motor.
    stop : float, optional
        The stopping angle for the rotation motor.
    alternate_flyscan_dir : bool, optional
        Whether to alternate the flyscan direction at each step.
    use_shutter : bool, optional
        Whether to use the front-end shutter during the scan.
    sample_name : str | None, optional
        The name of the sample being scanned.
    time_based : bool, optional
        Whether the scan is time-based.
    stream_name : str, optional
        The name of the data stream.
    panda : HDFPanda | None, optional
        The HDFPanda instance for data storage.
    rot_motor : RotationMotor | None, optional
        The rotation motor for the scan.
    fe_shutter : Shutter | None, optional
        The front-end shutter for the scan.
    photon_shutter : Shutter | None, optional
        The photon shutter for the scan.
    snake_axes : bool | iterable | None
        which step axes to snake, forwarded to :func:`bluesky.plans.grid_scan`

    All remaining keyword arguments are forwarded to :func:`tomo_flyscan`.
    """

    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)
    panda = ensure_available(HDFPanda, panda=panda)
    rot_motor = ensure_available(RotationMotor, rot_motor=rot_motor)
    single_axis_panda_flyer = SingleAxisFlyableLogic(panda).with_device()

    if use_shutter:
        yield from ensure_shutter_open(fe_shutter)

    # Index each axis position in the order it is first visited; a snaked axis
    # revisits the same values, so indices stay stable across passes.
    axis_indices: defaultdict[Movable, dict[Any, int]] = defaultdict(dict)
    iteration_counter: int = 0

    # TODO: It would be worth seeing if we could loosen the requirements for
    # the per_step function signature to be more flexible upstream.
    def _per_step(
        detectors: Sequence[Readable],
        step: Mapping[Movable, Any],
        pos_cache: dict[Movable, Any],
        take_reading: bps.TakeReading | None = None,
    ):
        nonlocal iteration_counter

        # Move the step axes to this point
        yield from bps.move_per_step(step, pos_cache)

        # Record where we are so the BEC shows the outer grid as a table.
        yield from bps.trigger_and_read(
            [motor for motor in step if isinstance(motor, Readable)], name="positions"
        )

        point_stream_name = stream_name + "_" + "_".join(
            f"{axis_indices[motor].setdefault(position, len(axis_indices[motor]))}"
            for motor, position in step.items()
            if isinstance(motor, HasName)
        )

        actual_start = start if not alternate_flyscan_dir or iteration_counter % 2 == 0 else stop
        actual_stop = stop if not alternate_flyscan_dir or iteration_counter % 2 == 0 else start

        print(f"Starting flyscan from {actual_start} to {actual_stop}")

        # Then, run a tomo flyscan
        yield from _tomo_fly_stub(
            detectors=[
                det for det in detectors
                if isinstance(det, KinetixDetector) or isinstance(det, PhantomDetector)
            ],
            single_axis_panda_flyer=single_axis_panda_flyer,
            num_images=num_images,
            exposure_time=exposure_time,
            acquire_period=acquire_period,
            images_to_average=images_to_average,
            start=actual_start,
            stop=actual_stop,
            use_shutter=use_shutter,
            time_based=time_based,
            stream_name=point_stream_name,
            panda=panda,
            rot_motor=rot_motor,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
        )
    
        iteration_counter += 1

    # Stage the ephemeral flyer and rot_motor here,
    # since the stage inside the grid scan is not aware of them
    @bpp.stage_decorator([panda, single_axis_panda_flyer, rot_motor])
    def _body():
        _md: dict = {"plan_name": "tomo_nd_step_scan"}
        if sample_name is not None:
            _md["sample_name"] = sample_name
        _md.update(md or {})
        yield from bp.grid_scan(detectors, *args, snake_axes=snake_axes, per_step=_per_step, md=_md)

    def _cleanup():
        """Perform post-scan cleanup, regardless of result"""

        yield from ensure_shutter_closed(photon_shutter, allow_actuation=True)
        yield from bps.abs_set(rot_motor.motor_stop, 1)

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))


def tomo_1d_step_scan(
    detectors: list[KinetixDetector | PhantomDetector],
    step_motor: AsyncMovable[float],
    step_start: float,
    step_stop: float,
    step_num: int,
    num_images: int,
    exposure_time: float,
    acquire_period: float | None = None,
    images_to_average: int = 1,
    start: float = 0,
    stop: float = 180,
    use_shutter: bool = True,
    sample_name: str | None = None,
    alternate_flyscan_dir: bool = False,
    time_based: bool = True,
    stream_name: str = "primary",
    panda: HDFPanda | None = None,
    rot_motor: RotationMotor | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
):
    """1-D tomography step scan: a tomo flyscan at each point along one axis.
    
    Parameters
    ----------
    detectors : list[KinetixDetector | PhantomDetector]
        The list of detectors to use during the scan.
    step_motor : AsyncMovable[float]
        The motor that will be moved in steps during the scan.
    step_start : float
        The starting position for the step motor.
    step_stop : float
        The stopping position for the step motor.
    step_num : int
        The number of steps to take between the start and stop positions.
    num_images : int
        The number of images to acquire at each step.
    exposure_time : float
        The exposure time for each image.
    acquire_period : float | None, optional
        The period between consecutive acquisitions. If None, the period is determined by the exposure time.
    images_to_average : int, optional
        The number of images to average for each acquisition.
    start : float, optional
        The starting angle for the rotation motor.
    stop : float, optional
        The stopping angle for the rotation motor.
    use_shutter : bool, optional
        Whether to use the shutter during the scan.
    sample_name : str | None, optional
        The name of the sample being scanned.
    alternate_flyscan_dir : bool, optional
        Whether to alternate the flyscan direction between steps.
    time_based : bool, optional
        Whether the scan is time-based.
    stream_name : str, optional
        The name of the data stream.
    panda : HDFPanda | None, optional
        The HDFPanda instance for data storage.
    rot_motor : RotationMotor | None, optional
        The rotation motor for the scan.
    fe_shutter : Shutter | None, optional
        The front-end shutter for the scan.
    photon_shutter : Shutter | None, optional
        The photon shutter for the scan.
    """

    return (yield from tomo_nd_step_scan(
        detectors,
        step_motor,
        step_start,
        step_stop,
        step_num,
        num_images=num_images,
        exposure_time=exposure_time,
        acquire_period=acquire_period,
        images_to_average=images_to_average,
        start=start,
        stop=stop,
        alternate_flyscan_dir=alternate_flyscan_dir,
        use_shutter=use_shutter,
        sample_name=sample_name,
        time_based=time_based,
        stream_name=stream_name,
        panda=panda,
        rot_motor=rot_motor,
        fe_shutter=fe_shutter,
        photon_shutter=photon_shutter,
        md={"plan_name": "tomo_1d_step_scan"},
    ))


def tomo_2d_step_scan(
    detectors: list[KinetixDetector | PhantomDetector],
    outer_motor: AsyncMovable[float],
    outer_start: float,
    outer_stop: float,
    outer_num: int,
    inner_motor: AsyncMovable[float],
    inner_start: float,
    inner_stop: float,
    inner_num: int,
    num_images: int,
    exposure_time: float,
    acquire_period: float | None = None,
    images_to_average: int = 1,
    start: float = 0,
    stop: float = 180,
    use_shutter: bool = True,
    alternate_flyscan_dir: bool = False,
    sample_name: str | None = None,
    time_based: bool = True,
    stream_name: str = "primary",
    snake_axes: bool | None = False,
    panda: HDFPanda | None = None,
    rot_motor: RotationMotor | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
):
    """2-D tomography step scan: a tomo flyscan at each point of a 2-D grid.

    The ``outer_*`` axis is the slowest (outer) loop and the ``inner_*`` axis is
    the fastest (inner) loop, matching :func:`bluesky.plans.grid_scan`.

    Parameters
    ----------
    detectors : list[KinetixDetector | PhantomDetector]
        The list of detectors to use for the scan.
    outer_motor : AsyncMovable[float]
        The motor controlling the outer axis of the 2-D grid.
    outer_start : float
        The starting position of the outer motor.
    outer_stop : float
        The stopping position of the outer motor.
    outer_num : int
        The number of steps for the outer motor.
    inner_motor : AsyncMovable[float]
        The motor controlling the inner axis of the 2-D grid.
    inner_start : float
        The starting position of the inner motor.
    inner_stop : float
        The stopping position of the inner motor.
    inner_num : int
        The number of steps for the inner motor.
    num_images : int
        The number of images to acquire at each point of the 2-D grid.
    exposure_time : float
        The exposure time for each image.
    acquire_period : float | None, default None
        The period between acquisitions. If None, the acquisition will proceed as fast as possible.
    images_to_average : int, default 1
        The number of images to average for each acquisition.
    start : float, default 0
        The starting angle for the tomography scan.
    stop : float, default 180
        The stopping angle for the tomography scan.
    use_shutter : bool, default True
        Whether to use the shutter during the scan.
    alternate_flyscan_dir : bool, default False
        If True, alternate the direction of the flyscan for each iteration.
    sample_name : str | None, default None
        The name of the sample being scanned.
    time_based : bool, default True
        If True, the scan is time-based rather than angle-based.
    stream_name : str, default "primary"
        The name of the data stream.
    snake_axes : bool | None, default False
        If True, the scan will snake along the axes.
    panda : HDFPanda | None, default None
        The HDFPanda instance for data storage.
    rot_motor : RotationMotor | None, default None
        The rotation motor for the tomography scan.
    fe_shutter : Shutter | None, default None
        The front-end shutter for the scan.
    photon_shutter : Shutter | None, default None
        The photon shutter for the scan.

    Returns
    -------
    generator
        A generator that yields the results of the tomography 2-D step scan.
    """
    return (yield from tomo_nd_step_scan(
        detectors,
        outer_motor,
        outer_start,
        outer_stop,
        outer_num,
        inner_motor,
        inner_start,
        inner_stop,
        inner_num,
        num_images=num_images,
        exposure_time=exposure_time,
        acquire_period=acquire_period,
        images_to_average=images_to_average,
        start=start,
        stop=stop,
        alternate_flyscan_dir=alternate_flyscan_dir,
        use_shutter=use_shutter,
        sample_name=sample_name,
        time_based=time_based,
        stream_name=stream_name,
        snake_axes=snake_axes,
        panda=panda,
        rot_motor=rot_motor,
        fe_shutter=fe_shutter,
        photon_shutter=photon_shutter,
        md={"plan_name": "tomo_2d_step_scan"},
    ))

