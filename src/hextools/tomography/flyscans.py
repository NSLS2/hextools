"""Tomography plans for HEX beamline."""

from collections.abc import Mapping, Sequence

from bluesky import plan_stubs as bps
from bluesky.protocols import Collectable, Flyable, HasName, Movable, Readable
from bluesky.utils import CustomPlanMetadata, MsgGenerator, plan
from nslsii import detectors
from ophyd_async.core import AsyncMovable, DetectorTrigger, StandardFlyable, TriggerInfo
from ophyd_async.epics.adkinetix import KinetixDetector, KinetixTriggerMode
from ophyd_async.fastcs.panda import HDFPanda
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor
from typing import Any

from hextools.photon_delivery_system.shutter import (
    ensure_shutter_closed,
    ensure_shutter_open,
)
from hextools.utils import ensure_available, get_obj_from_ipython_ns
from hextools.photon_delivery_system import Shutter

from ..detectors.phantom import PhantomDetector
from ..flyers import SingleAxisFlyableLogic, construct_fly_info_models
from ..motors import RotationMotor
from bluesky import preprocessors as bpp, plans as bp

from hextools.detectors import FRAME_PERIOD_MARGIN

from hextools import flyers


# TODO: This is vendored from bluesky.plans, because we need to be able to fly
# either with or without creating a run.
def fly(
    flyers: list[Flyable],
    *,
    as_stub: bool = False,
    md: CustomPlanMetadata | None = None,
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
    uid = None
    if not as_stub:
        uid = yield from bps.open_run(md)

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

    if not as_stub:
        yield from bps.close_run()
        return uid


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
    single_axis_panda_flyer: StandardFlyable | None = None,
    as_stub: bool = False,
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

    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)

    if use_shutter:
        yield from ensure_shutter_open(fe_shutter)

    panda = ensure_available(HDFPanda, panda=panda)
    rot_motor = ensure_available(RotationMotor, rot_motor=rot_motor)

    if acquire_period is None:
        acquire_period = exposure_time + FRAME_PERIOD_MARGIN

    all_detectors = [*detectors, panda]

    # Construct ephemeral flyer for the single axis flyscan
    single_axis_panda_flyer = SingleAxisFlyableLogic(panda).with_device() if not single_axis_panda_flyer else single_axis_panda_flyer
    all_devices = [*all_detectors, single_axis_panda_flyer, rot_motor]

    @bpp.stage_decorator([] if as_stub else all_devices)
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

        _md = {
            "detectors": [det.name for det in detectors],
            "num_points": num_images,
            "images_to_average": images_to_average,
            "plan_name": "tomo_flyscan",
        }
        if sample_name is not None:
            _md["sample_name"] = sample_name

        yield from fly(
            all_devices,
            md=_md,
            collect_flush_period=max(1, exposure_time + overhead),
            stream_name=stream_name,
            as_stub=as_stub,
        )

    def _cleanup():
        """Perform post-scan cleanup, regardless of result"""

        yield from ensure_shutter_closed(photon_shutter, allow_actuation=True)
        yield from bps.abs_set(rot_motor.motor_stop, 1)

    yield from bpp.finalize_wrapper(_body(), _cleanup())


def tomo_nd_step_scan(
    detectors: list[KinetixDetector | PhantomDetector],
    *args,
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
    snake_axes : bool | iterable | None
        which step axes to snake, forwarded to :func:`bluesky.plans.grid_scan`

    All remaining keyword arguments are forwarded to :func:`tomo_flyscan`.
    """

    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)
    panda = ensure_available(HDFPanda, panda=panda)
    rot_motor = ensure_available(RotationMotor, rot_motor=rot_motor)

    if use_shutter:
        yield from ensure_shutter_open(fe_shutter)

    # TODO: It would be worth seeing if we could loosen the requirements for
    # the per_step function signature to be more flexible upstream.
    def _per_step(
        detectors: Sequence[Readable],
        step: Mapping[Movable, Any],
        pos_cache: dict[Movable, Any],
        take_reading: bps.TakeReading | None = None,
    ):

        # Move the step axes to this grid point
        yield from bps.move_per_step(step, pos_cache)

        # Generate a position tag based on the current step positions
        # to be used for the per-step tomography stream name.
        # "-"/"." are substituted so the stream name stays a valid identifier.
        position_tag = "_".join(
            f"{motor.name}_{position:g}".replace("-", "_").replace(".", "_")
            for motor, position in step.items() if isinstance(motor, HasName)
        )

        # Then, run a tomo flyscan
        yield from tomo_flyscan(
            detectors=[
                det for det in detectors
                if isinstance(det, KinetixDetector) or isinstance(det, PhantomDetector)
            ],
            num_images=num_images,
            exposure_time=exposure_time,
            acquire_period=acquire_period,
            images_to_average=images_to_average,
            start=start,
            stop=stop,
            use_shutter=use_shutter,
            sample_name=sample_name,
            time_based=time_based,
            stream_name=f"{stream_name}_{position_tag}" if position_tag else stream_name,
            panda=panda,
            rot_motor=rot_motor,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
            as_stub=True,
        )

    single_axis_panda_flyer = SingleAxisFlyableLogic(panda).with_device()

    # Stage the ephemeral flyer and rot_motor here,
    # since the stage inside the grid scan is not aware of them
    @bpp.stage_decorator(single_axis_panda_flyer, rot_motor)
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

    yield from bpp.finalize_wrapper(_body(), _cleanup())


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
    time_based: bool = True,
    stream_name: str = "primary",
    panda: HDFPanda | None = None,
    rot_motor: RotationMotor | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
    md: dict | None = None,
):
    """1-D tomography step scan: a tomo flyscan at each point along one axis."""
    yield from tomo_nd_step_scan(
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
        use_shutter=use_shutter,
        sample_name=sample_name,
        time_based=time_based,
        stream_name=stream_name,
        panda=panda,
        rot_motor=rot_motor,
        fe_shutter=fe_shutter,
        photon_shutter=photon_shutter,
        md=md,
    )


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
    """2-D tomography step scan: a tomo flyscan at each point of a 2-D grid.

    The ``outer_*`` axis is the slowest (outer) loop and the ``inner_*`` axis is
    the fastest (inner) loop, matching :func:`bluesky.plans.grid_scan`.
    """
    yield from tomo_nd_step_scan(
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
        use_shutter=use_shutter,
        sample_name=sample_name,
        time_based=time_based,
        stream_name=stream_name,
        snake_axes=snake_axes,
        panda=panda,
        rot_motor=rot_motor,
        fe_shutter=fe_shutter,
        photon_shutter=photon_shutter,
        md=md,
    )

