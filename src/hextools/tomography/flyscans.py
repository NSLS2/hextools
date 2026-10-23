"""Tomography plans for HEX beamline."""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, cast

from bluesky import plan_patterns
from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from bluesky.protocols import Collectable, Flyable
from bluesky.utils import (
    FailedStatus,
    MsgGenerator,
    PlanHalt,
    RunEngineControlException,
)
from ophyd_async.core import (
    DetectorTrigger,
    TriggerInfo,
)
from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.fastcs.panda import HDFPanda

from hextools.detectors import (
    FRAME_PERIOD_MARGIN,
    enable_preview_mode,
    set_plugin_queue_sizes,
)
from hextools.motors import SampleTower
from hextools.photon_delivery_system import Shutter
from hextools.photon_delivery_system.shutter import (
    ensure_shutter_closed,
    ensure_shutter_open,
)
from hextools.specs import TOMO_DARK_FLAT_V1, TOMO_FLYSCAN_V1
from hextools.utils import Steppable, ensure_available

from ..detectors.phantom import PhantomDetector
from ..flyers import SingleAxisFlyableLogic, construct_fly_info_models
from ..motors import RotationMotor

logger = logging.getLogger("hextools")

#: Most frames any detector may drop before a flyscan is treated as failed.
MAX_MISSING_FRAMES = 10


def tomo_dark_flat(
    detectors: list[KinetixDetector | PhantomDetector],
    exposure_time: float,
    num_dark_images: int = 20,
    num_flat_images: int = 50,
    x_offset: float = 0.0,
    use_shutter: bool = True,
    sample_tower: SampleTower | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
):
    """Take a series of dark and flat images for tomography.

    Parameters
    ----------
    detectors : list[KinetixDetector | PhantomDetector]
        The list of detectors to use for the dark and flat images.
    exposure_time : float
        The exposure time for each image in seconds.
    num_dark_images : int, optional
        The number of dark images to acquire. Default is 20.
    num_flat_images : int, optional
        The number of flat images to acquire. Default is 50.
    x_offset : float, optional
        The offset to apply to the sample tower's x1 position for flat images.
        Default is 0.0.
    use_shutter : bool, optional
        Whether to use the shutter for the acquisition. Default is True.
    sample_tower : SampleTower | None, optional
        The sample tower to use. If None, will attempt to ensure availability.
    fe_shutter : Shutter | None, optional
        The front-end shutter to use. If None, will attempt to ensure availability.
    photon_shutter : Shutter | None, optional
        The photon shutter to use. If None, will attempt to ensure availability.
    """
    sample_tower = ensure_available(SampleTower, sample_tower=sample_tower)
    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)
    if use_shutter:
        yield from ensure_shutter_open(fe_shutter)

    _md = {
        "plan_name": "tomo_dark_flat",
        "detectors": [det.name for det in detectors],
        "exposure_time": exposure_time,
        "num_dark_images": num_dark_images,
        "num_flat_images": num_flat_images,
        "x_offset": x_offset,
        "tiled_specs": [
            {"name": TOMO_DARK_FLAT_V1.name, "version": TOMO_DARK_FLAT_V1.version}
        ],
    }

    @bpp.reset_positions_decorator(
        [sample_tower.x1, *[det.driver.acquire_time for det in detectors]]
    )
    @bpp.run_decorator(md=_md)
    @bpp.stage_decorator([*detectors])
    def _body():
        yield from ensure_shutter_closed(photon_shutter)

        for dset in ("dark", "flat"):
            if dset == "flat":
                yield from bps.mvr(sample_tower.x1, x_offset)
                if use_shutter:
                    yield from ensure_shutter_open(photon_shutter, allow_actuation=True)

            for det in detectors:
                yield from bps.prepare(
                    det,
                    TriggerInfo(
                        livetime=exposure_time,
                        collections_per_event=num_dark_images
                        if dset == "dark"
                        else num_flat_images,
                    ),
                    group=f"prepare-{dset}",
                )
            yield from bps.wait(group=f"prepare-{dset}")
            yield from bps.trigger_and_read(list(detectors), name=dset)

    def _cleanup():
        if use_shutter:
            yield from ensure_shutter_closed(photon_shutter)
        for det in detectors:
            if isinstance(det, KinetixDetector):
                yield from enable_preview_mode(det)

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))


def tomo_flyscan(
    detectors: list[KinetixDetector | PhantomDetector],
    num_projections: int,
    exposure_time: float,
    acquire_period: float | None = None,
    images_to_average: int = 1,
    start: float = 0,
    stop: float = 180,
    use_shutter: bool = True,
    sample_name: str | None = None,
    time_based: bool = True,
    panda: HDFPanda | None = None,
    rot_motor: RotationMotor | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
    md: dict[str, Any] | None = None,
):
    """Take a single tomography fly scan.

    Parameters
    ----------
    detectors : list[KinetixDetector | PhantomDetector]
        The list of detectors to use for the tomography fly scans.
    num_projections : int
        The number of projections to acquire during the tomography fly scan.
    exposure_time : float
        The exposure time for each projection.
    acquire_period : float | None
        The period between acquisitions. If None, it will be set to
        exposure_time + some margin.
    images_to_average : int
        The number of images to average for each projection.
    start : float
        The starting angle for the tomography fly scan.
    stop : float
        The stopping angle for the tomography fly scan.
    use_shutter : bool
        Whether to use the shutter during the scan.
    sample_name : str | None
        The name of the sample being scanned.
    time_based : bool
        Whether the triggers are issued equally spaced in time, as opposed to
        being position-based.
    panda : HDFPanda | None
        The HDF Panda device to use.
    rot_motor : RotationMotor | None
        The rotation motor to use.
    fe_shutter : Shutter | None
        The front-end shutter to use.
    photon_shutter : Shutter | None
        The photon shutter to use.
    md : dict[str, Any] | None
        Extra metadata for the run's start document.
    """
    panda = ensure_available(HDFPanda, panda=panda)
    rot_motor = ensure_available(RotationMotor, rot_motor=rot_motor)
    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)
    single_axis_panda_flyer = SingleAxisFlyableLogic(panda).with_device()

    _md = {
        "detectors": [det.name for det in detectors],
        "num_points": num_projections,
        "exposure_time": exposure_time,
        "acquire_period": acquire_period,
        "time_based": time_based,
        "start_position": start,
        "stop_position": stop,
        "images_to_average": images_to_average,
        "plan_name": "tomo_flyscan",
        # TODO: Allow Spec objects directly in tiled_specs
        "tiled_specs": [
            {"name": TOMO_FLYSCAN_V1.name, "version": TOMO_FLYSCAN_V1.version}
        ],
    }
    if sample_name is not None:
        _md["sample_name"] = sample_name
    _md.update(md or {})

    if use_shutter:
        yield from ensure_shutter_open(fe_shutter)

    if acquire_period is None:
        acquire_period = exposure_time + FRAME_PERIOD_MARGIN

    all_detectors = [*detectors, panda]

    # Construct ephemeral flyer for the single axis flyscan
    all_devices = [*all_detectors, single_axis_panda_flyer, rot_motor]

    @bpp.stage_decorator(all_devices)
    @bpp.run_decorator(md=_md)
    def _body():

        # Get the start position in encoder counts
        encoder_res = yield from bps.rd(rot_motor.encoder_resolution)
        max_velocity = yield from bps.rd(rot_motor.max_velocity)
        current_position = yield from bps.rd(rot_motor.user_readback)

        # TODO: Come up with better way to access panda calc block
        current_enc = yield from bps.rd(panda.calc[1].out)  # type: ignore

        det_trigger_info = TriggerInfo(
            number_of_events=num_projections,
            livetime=exposure_time,
            deadtime=acquire_period - exposure_time,
            trigger=DetectorTrigger.EXTERNAL_EDGE,
            exposures_per_collection=images_to_average,
        )

        panda_trigger_info = TriggerInfo(
            number_of_events=num_projections,
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
            num_pulses=num_projections * images_to_average,
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
        yield from bps.wait(group="prepare")

        # Arm the PandA and detectors only once the motor is at its run-up position,
        # so no frames are captured before kickoff.
        yield from bps.prepare(single_axis_panda_flyer, flyer_info, group="arm")
        for det in detectors:
            yield from bps.prepare(det, det_trigger_info, group="arm")

        # Set plugin queues to fit the entire run in memory
        yield from set_plugin_queue_sizes(
            detectors, queue_size=det_trigger_info.number_of_exposures
        )

        yield from bps.prepare(panda, panda_trigger_info, group="arm")
        yield from bps.wait(group="arm")

        try:
            yield from bps.fly_stub(
                all_devices,  # type: ignore
                collect_flush_period=max(1, exposure_time + overhead),
                stream_name="primary",
            )
        except FailedStatus as err:
            # bluesky annotates this as Status, but the RunEngine returns futures.
            indices = cast(
                list[asyncio.Future[int]],
                (yield from bps.wait_for([det.get_index for det in detectors])),
            )
            missing = {
                det.name: num_projections - index.result()
                for det, index in zip(detectors, indices, strict=True)
            }
            most_missing = max(missing.values(), default=0)
            if most_missing == 0:
                # The detectors got every frame, so something else failed.
                raise
            if most_missing > MAX_MISSING_FRAMES:
                # Losing half or more points to a failure other than dropped frames.
                # Re-raise the raw exception for more helpful debugging.
                if most_missing >= num_projections / 2:
                    raise
                raise RuntimeError(
                    f"Scan failed! Detector(s) missed up to {most_missing} frame(s). "
                    f"Failure threshold: {MAX_MISSING_FRAMES}"
                ) from err
            for det, count in missing.items():
                logger.warning(f"Detector {det} missed {count} frame(s)")
            logger.warning(
                f"Max missed frames {most_missing} is below failure threshold "
                f"{MAX_MISSING_FRAMES}"
            )
            # Record the frames that every device did manage to write.
            yield from bps.collect(
                *[dev for dev in all_devices if isinstance(dev, Collectable)],
                name="primary",
            )

    def _cleanup():
        yield from ensure_shutter_closed(photon_shutter)
        yield from bps.abs_set(rot_motor.motor_stop, 1)
        for det in detectors:
            if isinstance(det, KinetixDetector):
                yield from enable_preview_mode(det)

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))


def tomo_nd_step_scan(
    detectors: list[KinetixDetector | PhantomDetector],
    *args,
    num_projections: int,
    exposure_time: float,
    acquire_period: float | None = None,
    images_to_average: int = 1,
    start: float = 0,
    stop: float = 180,
    num_repeats: int = 1,
    delay_between_points: float = 0.0,
    use_shutter: bool = True,
    alternate_flyscan_dir: bool = False,
    dark_flat_offset: float = 0.0,
    dark_flat_exposure_time: float | None = None,
    num_dark_images: int = 20,
    num_flat_images: int = 50,
    dark_flat_frequency: int = 0,
    sample_name: str | None = None,
    time_based: bool = True,
    snake_axes: bool | Sequence[Steppable] = False,
    num_retries: int = 0,
    sample_tower: SampleTower | None = None,
    panda: HDFPanda | None = None,
    rot_motor: RotationMotor | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
):
    """N-D tomography step scan: a tomo flyscan at each point of an N-D grid.

    Step motors are returned to their starting positions when the scan ends.

    Parameters
    ----------
    detectors : list[KinetixDetector | PhantomDetector]
        The list of detectors to use for each tomo flyscan.
    ``*args``
        Step axes patterned like ``(motor1, start1, stop1, num1, motor2, start2,
        stop2, num2, ...)``, outermost (slowest) axis first, as in
        :func:`bluesky.plans.grid_scan`. With no axes, the flyscan is repeated
        in place.
    num_projections : int
        The number of images to acquire at each grid point.
    exposure_time : float
        The exposure time for each image.
    acquire_period : float | None, optional
        The period between consecutive acquisitions. If None, it is derived from
        the exposure time.
    images_to_average : int, optional
        The number of images to average for each acquisition.
    start : float, optional
        The starting angle for the rotation motor.
    stop : float, optional
        The stopping angle for the rotation motor.
    num_repeats : int, optional
        The number of times to run the whole trajectory.
    delay_between_points : float, optional
        Seconds to wait between consecutive flyscans.
    use_shutter : bool, optional
        Whether to use the shutters during the scan.
    alternate_flyscan_dir : bool, optional
        Whether to reverse the flyscan direction at every other grid point.
    sample_name : str | None, optional
        The name of the sample being scanned.
    dark_flat_offset : float, optional
        Sample x offset for flats. If non-zero, a dark/flat is taken before and
        after the scan; if 0, no darks/flats are taken.
    dark_flat_exposure_time : float | None, optional
        Exposure time for darks/flats. If None, ``exposure_time`` is used.
    num_dark_images : int, optional
        Number of dark images per dark/flat.
    num_flat_images : int, optional
        Number of flat images per dark/flat.
    dark_flat_frequency : int, optional
        If greater than 0, also take a dark/flat after every this many grid points.
    time_based : bool, optional
        Whether the scan is time-based.
    snake_axes : bool | Sequence[Steppable], optional
        If True, snake every axis but the outermost; otherwise, the motors to snake.
    num_retries : int, optional
        How many times to retry a grid point's flyscan if it fails; 0 never retries.
    panda : HDFPanda | None, optional
        The HDFPanda instance for data storage.
    rot_motor : RotationMotor | None, optional
        The rotation motor for the scan.
    fe_shutter : Shutter | None, optional
        The front-end shutter for the scan.
    photon_shutter : Shutter | None, optional
        The photon shutter for the scan.
    sample_tower : SampleTower | None, optional
        The SampleTower instance to use for positioning the sample. Default is None.
    """

    # Extract the step axes from the positional arguments.
    if len(args) % 4:
        raise ValueError("Step axes must be given as (motor, start, stop, num) groups.")
    axes = [args[i : i + 4] for i in range(0, len(args), 4)]
    motors = [axis[0] for axis in axes]

    # Determine which axes are snaked based on the snake_axes argument.
    if isinstance(snake_axes, bool):
        snaked = motors[1:] if snake_axes else []
    else:
        snaked = list(snake_axes)

    # Make sure we didn't ask to snake the outermost axis.
    if motors and motors[0] in snaked:
        raise ValueError("The outermost axis cannot be snaked.")

    # Construct the grid of positions for the scan.
    if axes:
        pattern_args: list[Any] = [*axes[0]]
        for axis in axes[1:]:
            pattern_args += [*axis, axis[0] in snaked]
        grid = plan_patterns.outer_product(pattern_args)
    else:
        grid = [{}]  # When no axes are specified, create a single-point grid.

    # Repeat the trajectory the specified number of times.
    trajectory = grid * num_repeats

    @bpp.reset_positions_decorator(motors)
    def _body():
        # Flats need the sample moved out of the beam, so a zero offset means no
        # darks/flats.
        take_dark_flats = dark_flat_offset != 0

        # Define a helper function to acquire dark and flat images.
        def _dark_flat():
            yield from bps.checkpoint()
            yield from tomo_dark_flat(
                detectors=detectors,
                exposure_time=dark_flat_exposure_time or exposure_time,
                num_dark_images=num_dark_images,
                num_flat_images=num_flat_images,
                x_offset=dark_flat_offset,
                use_shutter=use_shutter,
                sample_tower=sample_tower,
                fe_shutter=fe_shutter,
                photon_shutter=photon_shutter,
            )

        if take_dark_flats:
            yield from _dark_flat()

        num_points = len(trajectory)
        pos_cache: defaultdict[Any, Any] = defaultdict(lambda: None)
        previous_uid: str | None = None
        for i, point in enumerate(trajectory):
            # A deferred pause stops here; resuming starts this point afresh.
            yield from bps.checkpoint()
            if i > 0 and delay_between_points > 0:
                yield from bps.sleep(delay_between_points)

            # Only moves the axes whose position changed since the last point.
            yield from bps.move_per_step(point, pos_cache)

            reverse = alternate_flyscan_dir and i % 2 == 1
            md: dict[str, Any] = {
                "grid_index": i,
                "num_grid_points": num_points,
                "grid_positions": {motor.name: pos for motor, pos in point.items()},
            }
            if previous_uid is not None:
                md["previous_grid_point_uid"] = previous_uid
            for attempt in range(1, num_retries + 2):
                try:
                    previous_uid = yield from tomo_flyscan(
                        detectors=detectors,
                        num_projections=num_projections,
                        exposure_time=exposure_time,
                        acquire_period=acquire_period,
                        images_to_average=images_to_average,
                        start=stop if reverse else start,
                        stop=start if reverse else stop,
                        use_shutter=use_shutter,
                        sample_name=sample_name,
                        time_based=time_based,
                        panda=panda,
                        rot_motor=rot_motor,
                        fe_shutter=fe_shutter,
                        photon_shutter=photon_shutter,
                        md=md,
                    )
                    break
                except (RunEngineControlException, PlanHalt):
                    # Never retry a user's abort, stop or halt request.
                    raise
                except Exception as err:
                    if attempt > num_retries:
                        raise RuntimeError(
                            f"Flyscan at point {i} failed after {attempt} "
                            f"attempts: {err}"
                        ) from err
                    logger.error(
                        f"Flyscan at point {i} failed ({err}), retrying "
                        f"(attempt {attempt}/{num_retries})."
                    )

            # The final dark/flat below covers the last point.
            is_last = i == num_points - 1
            if (
                take_dark_flats
                and dark_flat_frequency > 0
                and not is_last
                and (i + 1) % dark_flat_frequency == 0
            ):
                yield from _dark_flat()

        if take_dark_flats:
            yield from _dark_flat()

    return (yield from _body())


def tomo_flyscan_loop(
    detectors: list[KinetixDetector | PhantomDetector],
    num_projections: int,
    exposure_time: float,
    acquire_period: float | None = None,
    images_to_average: int = 1,
    start: float = 0,
    stop: float = 180,
    num_iterations: int = 1,
    delay_between_tomograms: float = 0.0,
    dark_flat_offset: float = 0.0,
    num_dark_images: int = 20,
    num_flat_images: int = 50,
    dark_flat_frequency: int = 0,
    dark_flat_exposure_time: float | None = None,
    use_shutter: bool = True,
    sample_name: str | None = None,
    time_based: bool = True,
    panda: HDFPanda | None = None,
    sample_tower: SampleTower | None = None,
    rot_motor: RotationMotor | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
):
    """Take a series of tomography fly scans, with interspersed dark/flat acquisitions.

    Parameters
    ----------
    detectors : list[KinetixDetector | PhantomDetector]
        The list of detectors to use for the tomography fly scans.
    num_projections : int
        The number of projections to acquire for each tomography fly scan.
    exposure_time : float
        The exposure time for each projection.
    acquire_period : float | None, optional
        The period between successive acquisitions. If None, it will be
        determined automatically.
    images_to_average : int, optional
        The number of images to average for each projection. Default is 1.
    start : float, optional
        The starting angle for the tomography scan. Default is 0.
    stop : float, optional
        The stopping angle for the tomography scan. Default is 180.
    num_iterations : int, optional
        The number of times to repeat the tomography fly scan. Default is 1.
    delay_between_tomograms : float, optional
        The delay between successive tomography fly scans. Default is 0.0.
    dark_flat_offset : float, optional
        The offset to apply for dark/flat acquisitions. A value of 0.0 means no
        darks/flats. Default is 0.0.
    num_dark_images : int, optional
        The number of dark images to acquire for each dark/flat acquisition.
        Default is 20.
    num_flat_images : int, optional
        The number of flat images to acquire for each dark/flat acquisition.
        Default is 50.
    dark_flat_frequency : int, optional
        The frequency at which to take dark/flat acquisitions. Default is 0
        (no dark/flats).
    dark_flat_exposure_time : float | None, optional
        The exposure time for dark/flat acquisitions. If None, it will use the
        same exposure time as the projections.
    use_shutter : bool, optional
        Whether to use the shutter during acquisitions. Default is True.
    sample_name : str | None, optional
        The name of the sample being scanned. Default is None.
    time_based : bool, optional
        Whether the fly scan is time-based. Default is True.
    panda : HDFPanda | None, optional
        The HDFPanda instance to use for data storage. Default is None.
    sample_tower : SampleTower | None, optional
        The SampleTower instance to use for positioning the sample. Default is None.
    rot_motor : RotationMotor | None, optional
        The rotation motor to use for the tomography scan. Default is None.
    fe_shutter : Shutter | None, optional
        The front-end shutter to use during acquisitions. Default is None.
    photon_shutter : Shutter | None, optional
        The photon shutter to use during acquisitions. Default is None.

    """
    return (
        yield from tomo_nd_step_scan(
            detectors,
            num_projections=num_projections,
            exposure_time=exposure_time,
            acquire_period=acquire_period,
            images_to_average=images_to_average,
            start=start,
            stop=stop,
            num_repeats=num_iterations,
            delay_between_points=delay_between_tomograms,
            dark_flat_offset=dark_flat_offset,
            dark_flat_exposure_time=dark_flat_exposure_time,
            num_dark_images=num_dark_images,
            num_flat_images=num_flat_images,
            dark_flat_frequency=dark_flat_frequency,
            use_shutter=use_shutter,
            sample_name=sample_name,
            time_based=time_based,
            panda=panda,
            sample_tower=sample_tower,
            rot_motor=rot_motor,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
        )
    )


def tomo_1d_step_scan(
    detectors: list[KinetixDetector | PhantomDetector],
    step_motor: Steppable,
    step_start: float,
    step_stop: float,
    num_tomograms: int,
    num_projections: int,
    exposure_time: float,
    acquire_period: float | None = None,
    images_to_average: int = 1,
    start: float = 0,
    stop: float = 180,
    use_shutter: bool = True,
    sample_name: str | None = None,
    alternate_flyscan_dir: bool = False,
    num_retries: int = 3,
    dark_flat_offset: float = 0.0,
    num_dark_images: int = 20,
    num_flat_images: int = 50,
    dark_flat_frequency: int = 0,
    time_based: bool = True,
    panda: HDFPanda | None = None,
    rot_motor: RotationMotor | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
    sample_tower: SampleTower | None = None,
):
    """1-D tomography step scan: a tomo flyscan at each point along one axis.

    Parameters
    ----------
    detectors : list[KinetixDetector | PhantomDetector]
        The list of detectors to use during the scan.
    step_motor : Steppable
        The motor that will be moved in steps during the scan.
    step_start : float
        The starting position for the step motor.
    step_stop : float
        The stopping position for the step motor.
    num_tomograms : int
        The number of tomograms to acquire along the step axis.
    num_projections : int
        The number of images to acquire at each step.
    exposure_time : float
        The exposure time for each image.
    acquire_period : float | None, optional
        The period between consecutive acquisitions. If None, the period is
        determined by the exposure time.
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
    num_retries : int, optional
        The number of times to retry the scan if it fails.
    time_based : bool, optional
        Whether the scan is time-based.
    dark_flat_offset : float, optional
        The offset for dark and flat images.
    num_dark_images : int, optional
        The number of dark images to acquire.
    num_flat_images : int, optional
        The number of flat images to acquire.
    dark_flat_frequency : int, optional
        The frequency at which to acquire dark and flat images.
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
    sample_tower : SampleTower | None, optional
        The SampleTower instance to use for positioning the sample. Default is None.
    """
    return (
        yield from tomo_nd_step_scan(
            detectors,
            step_motor,
            step_start,
            step_stop,
            num_tomograms,
            num_projections=num_projections,
            exposure_time=exposure_time,
            acquire_period=acquire_period,
            images_to_average=images_to_average,
            start=start,
            stop=stop,
            use_shutter=use_shutter,
            num_retries=num_retries,
            alternate_flyscan_dir=alternate_flyscan_dir,
            sample_name=sample_name,
            time_based=time_based,
            dark_flat_offset=dark_flat_offset,
            num_dark_images=num_dark_images,
            num_flat_images=num_flat_images,
            dark_flat_frequency=dark_flat_frequency,
            panda=panda,
            rot_motor=rot_motor,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
            sample_tower=sample_tower,
        )
    )


def tomo_2d_step_scan(
    detectors: list[KinetixDetector | PhantomDetector],
    outer_motor: Steppable,
    outer_start: float,
    outer_stop: float,
    outer_num: int,
    inner_motor: Steppable,
    inner_start: float,
    inner_stop: float,
    inner_num: int,
    num_projections: int,
    exposure_time: float,
    acquire_period: float | None = None,
    images_to_average: int = 1,
    start: float = 0,
    stop: float = 180,
    use_shutter: bool = True,
    alternate_flyscan_dir: bool = False,
    num_retries: int = 0,
    sample_name: str | None = None,
    time_based: bool = True,
    snake_axes: bool | None = False,
    panda: HDFPanda | None = None,
    rot_motor: RotationMotor | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
    sample_tower: SampleTower | None = None,
):
    """2-D tomography step scan: a tomo flyscan at each point of a 2-D grid.

    The ``outer_*`` axis is the slowest (outer) loop and the ``inner_*`` axis is
    the fastest (inner) loop, matching :func:`bluesky.plans.grid_scan`.

    Parameters
    ----------
    detectors : list[KinetixDetector | PhantomDetector]
        The list of detectors to use for the scan.
    outer_motor : Steppable
        The motor controlling the outer axis of the 2-D grid.
    outer_start : float
        The starting position of the outer motor.
    outer_stop : float
        The stopping position of the outer motor.
    outer_num : int
        The number of steps for the outer motor.
    inner_motor : Steppable
        The motor controlling the inner axis of the 2-D grid.
    inner_start : float
        The starting position of the inner motor.
    inner_stop : float
        The stopping position of the inner motor.
    inner_num : int
        The number of steps for the inner motor.
    num_projections : int
        The number of images to acquire at each point of the 2-D grid.
    exposure_time : float
        The exposure time for each image.
    acquire_period : float | None, default None
        The period between acquisitions. If None, the acquisition will proceed
        as fast as possible.
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
    sample_tower : SampleTower | None, default None
        The SampleTower instance to use for positioning the sample. Default is None.
    num_retries : int, default 3
        The number of times to retry the scan if it fails.
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
    return (
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
            num_projections=num_projections,
            exposure_time=exposure_time,
            acquire_period=acquire_period,
            images_to_average=images_to_average,
            start=start,
            stop=stop,
            use_shutter=use_shutter,
            alternate_flyscan_dir=alternate_flyscan_dir,
            sample_name=sample_name,
            time_based=time_based,
            snake_axes=bool(snake_axes),
            num_retries=num_retries,
            panda=panda,
            rot_motor=rot_motor,
            sample_tower=sample_tower,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
        )
    )
