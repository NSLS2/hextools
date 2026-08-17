from bluesky import plan_stubs as bps
from ophyd_async.core import DetectorTrigger, StandardFlyable, TriggerInfo
from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.fastcs.panda import HDFPanda

from hextools.photon_delivery_system import Shutter

from ..detectors.phantom import PhantomDetector
from ..flyers import SingleAxisFlyableLogic, construct_fly_info_models
from ..motors import RotationMotor


def tomo_flyscan(
    detectors: list[KinetixDetector | PhantomDetector],
    exposure_time: float,
    panda: HDFPanda,
    motor: RotationMotor,
    photon_shutter: Shutter,
    num_images: int,
    start_position: float = 0,
    stop_position: float = 180,
    use_shutter: bool = True,
    sample_name: str | None = None,
    acquire_period: float | None = None,
    time_based: bool = True,
    stream_name: str = "primary",
):
    """Simple hardware triggered flyscan tomography

    Parameters
    ----------
    detectors : list[Flyable]
        list of detectors to be used in the scan
    panda : HDFPanda
        the panda device used for triggering and recording the rotation angle
    motor : RotationMotor
        the rotation stage for tomography
    exposure_time : float
        exposure time to use on the camera(s), in seconds
    num_images : int
        total number of camera images to collect during the scan
    start_position : float (optional)`
        starting point in degrees
    stop_position : float (optional)
        stopping point in degrees
    lead_angle : float (optional)
        the angle in degrees to be used to move motor to -lead_angle before 'start_deg' and +lead_angle after 'stop_deg'
    reset_speed : float
        speed of the rotary motor during reset movements, in deg/s
    use_shutter : bool
        whether to use/check the shutter during the scan
    """
    all_detectors = [*detectors, panda]

    # Construct ephemeral flyer for the single axis flyscan
    single_axis_panda_flyer = StandardFlyable(SingleAxisFlyableLogic(panda))
    all_devices = [*all_detectors, single_axis_panda_flyer, motor]

    # Get the start position in encoder counts
    encoder_res = yield from bps.rd(motor.encoder_resolution)
    max_velocity = yield from bps.rd(motor.max_velocity)

    det_trigger_info = TriggerInfo(
        number_of_events=num_images,
        trigger=DetectorTrigger.EXTERNAL_EDGE,
    )

    panda_trigger_info = TriggerInfo(
        number_of_events=num_images,
        trigger=DetectorTrigger.EXTERNAL_LEVEL,
    )

    overhead = (
        acquire_period - exposure_time
        if acquire_period is not None and acquire_period > exposure_time
        else 0
    )

    flyer_info, motor_info = construct_fly_info_models(
        num_pulses=num_images,
        max_exposure_time=exposure_time,
        start_position=start_position,
        stop_position=stop_position,
        encoder_resolution=encoder_res,
        max_motor_velocity=max_velocity,
        encoder_pos_at_zero=motor.encoder_pos_at_zero,
        acq_time_overhead=overhead,
        time_based=time_based,
    )

    _md = {
        "detectors": [det.name for det in detectors],
        "num_points": num_images,
        "plan_name": "single_axis_flyscan",
        "hints": {},
    }
    if sample_name is not None:
        _md["sample_name"] = sample_name
    yield from bps.open_run(md=_md)

    yield from bps.stage_all(*all_detectors)

    yield from bps.prepare(motor, motor_info, group="prepare")
    yield from bps.prepare(single_axis_panda_flyer, flyer_info, group="prepare")

    for det in detectors:
        yield from bps.prepare(det, det_trigger_info, group="prepare")

    yield from bps.prepare(panda, panda_trigger_info, group="prepare")

    # TODO: Come up with a way to set a timeout automatically based on the
    # motor move to start position time.
    yield from bps.wait(group="prepare")

    yield from bps.declare_stream(*all_detectors, name=stream_name)

    yield from bps.kickoff_all(*all_devices, wait=True)

    flush_period = max(1, exposure_time + overhead)
    yield from bps.collect_while_completing(
        all_devices,
        all_detectors,
        flush_period=flush_period,
        stream_name=stream_name,
    )
    yield from bps.unstage_all(*all_detectors)

    yield from bps.close_run()
