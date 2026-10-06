

from bluesky.preprocessors import finalize_wrapper
import numpy as np

from hextools.detectors.germ import GeRMDetector
from hextools.photon_delivery_system import ensure_shutter_open, Shutter
from hextools.photon_delivery_system.shutter import ensure_shutter_closed
from hextools.utils import ensure_available, Steppable
from hextools.motors import CollimatorTable, EDXDTable, SampleTower
from bluesky import plan_stubs as bps, plans as bp, preprocessors as bpp
from tests.photon_delivery_system.test_dclm import photon_shutter

def configure_test_pulses(
    use_test_pulses: bool,
    test_pulse_amplitude: int = 100,
    test_pulse_freq: int = 1000,
    test_pulse_count: int = 99999,
    germ: GeRMDetector | None = None,
):
    """Configure test pulses for the GeRM detector.

    Parameters
    ----------
    use_test_pulses : bool
        Whether to enable or disable test pulses.
    test_pulse_amplitude : int, default 100
        The amplitude of the test pulses.
    test_pulse_freq : int, default 1000
        The frequency of the test pulses.
    test_pulse_count : int, default 99999
        The number of test pulses to generate.
    germ : GeRMDetector, optional
        The GeRM detector device. If None, will pull from global namespace.
    """

    germ = ensure_available(GeRMDetector, germ=germ)
    
    test_pulses_enabled = yield from bps.rd(germ.driver.test_pulse_enable)

    # If already in the given state, must switch it off first
    if test_pulses_enabled == use_test_pulses:
        yield from bps.mv(germ.driver.test_pulse_enable, not use_test_pulses)

    # Enable/disable test pulses for all channels
    yield from bps.abs_set(germ.driver.enable_all_tsen, 1 if use_test_pulses else 0, wait=True)

    # Set the test pulse parameters and enable/disable test pulses
    yield from bps.mv(
        germ.driver.test_pulse_amplitude, test_pulse_amplitude if use_test_pulses else 0,
        germ.driver.test_pulse_frequency, test_pulse_freq if use_test_pulses else 0,
        germ.driver.test_pulse_count, test_pulse_count if use_test_pulses else 0,
        germ.driver.test_pulse_enable, use_test_pulses
    )


def edxd_count(
    count_time: float,
    num: int = 1,
    description: str | None = None,
    germ: GeRMDetector | None = None,
):
    """Take one or more counts with the GeRM detector.

    Parameters
    ----------
    count_time : float
        The acquisition time for each count.
    num : int, default 1
        The number of counts to take.
    description : str | None, optional
        A description for the count, if any.
    germ : GeRMDetector | None, optional
        The GeRM detector device. If None, will pull from global namespace.
    """
    
    germ = ensure_available(GeRMDetector, germ=germ)

    yield from bps.mv(germ.driver.acquire_time, count_time)

    _md = {"plan_name": "edxd_count"}
    if description is not None:
        _md["description"] = description
    yield from bp.count([germ], num=num, md=_md)


def edxd_scan(
    motor: Steppable,
    start: float,
    stop: float,
    num_points: int,
    count_time: float,
    num_iterations: int = 1,
    time_between_iterations: float = 0.0,
    description: str | None = None,
    save_as_tiff: bool = False,
    reset_position: bool = True,
    use_shutter: bool = True,
    germ: GeRMDetector | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
):
    """Perform a 1D scan with the EDXD detector.
    
    Parameters
    ----------
    motor : Steppable,
        The motor to scan.
    start : float
        The starting position for the motor.
    stop : float
        The stopping position for the motor.
    num_points : int
        The number of points in the scan.
    count_time : float
        The acquisition time at each point.
    num_iterations : int, default 1
        The number of times to repeat the scan.
    time_between_iterations : float, default 0.0
        The time to wait between iterations.
    description : str | None, optional
        A description for the scan, if any.
    save_as_tiff : bool, default False
        Whether to save the data as a TIFF file.
    reset_position : bool, default True
        Whether to reset the motor to its starting position after the scan.
    use_shutter : bool, default True
        Whether to use the shutters during the scan.
    germ : GeRMDetector | None, optional
        The GeRM detector device. If None, will pull from global namespace.
    fe_shutter : Shutter | None, optional
        The front-end shutter device. If None, will pull from global namespace.
    photon_shutter : Shutter | None, optional
        The photon shutter device. If None, will pull from global namespace.
    """

    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)
    germ = ensure_available(GeRMDetector, germ=germ)

    def _body():
        if use_shutter:
            yield from ensure_shutter_open(fe_shutter, allow_actuation=True)
            yield from ensure_shutter_open(photon_shutter, allow_actuation=True)

        yield from bps.mv(germ.driver.acquire_time, count_time)

        if save_as_tiff:
            germ.save_as_tiff()
        else:
            germ.save_as_hdf()

        _md = {
            "plan_name": "edxd_scan",
        }
        if description is not None:
            _md["description"] = description
        yield from bp.scan([germ], motor, start, stop, num_points, md=_md)

    def _cleanup():
        if use_shutter:
            yield from ensure_shutter_closed(fe_shutter)
            yield from ensure_shutter_closed(photon_shutter)
        if reset_position:
            yield from bps.mv(motor, start)

    return (
        yield from bps.repeat(
            lambda: finalize_wrapper(_body(), _cleanup()),
            num=num_iterations,
            delay=time_between_iterations
        )
    )



def edxd_grid_scan(
    outer_motor: Steppable,
    outer_start: float,
    outer_stop: float,
    outer_num_steps: int,
    inner_motor: Steppable,
    inner_start: float,
    inner_stop: float,
    inner_num_steps: int,
    count_time: float,
    snake: bool = False,
    use_shutter: bool = True,
    reset_position: bool = True,
    description: str | None = None,
    germ: GeRMDetector | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
):
    """Perform a 2D grid scan, taking an EDXD measurement at each grid point.

    Parameters
    ----------
    outer_motor : Steppable
        The outer (slower) motor for the grid scan.
    outer_start : float
        The starting position for the outer motor.
    outer_stop : float
        The stopping position for the outer motor.
    outer_num_steps : int
        The number of steps for the outer motor.
    inner_motor : Steppable
        The inner (faster) motor for the grid scan.
    inner_start : float
        The starting position for the inner motor.
    inner_stop : float
        The stopping position for the inner motor.
    inner_num_steps : int
        The number of steps for the inner motor.
    count_time : float
        The acquisition time at each grid point.
    snake : bool, default False
        Whether to perform the scan in a snake pattern.
    use_shutter : bool, default True
        Whether to use the shutters during the scan.
    reset_position : bool, default True
        Whether to reset the motors to their starting positions after the scan.
    description : str | None, optional
        A description for the scan, if any.
    germ : GeRMDetector | None, optional
        The GeRM detector device. If None, will pull from global namespace.
    fe_shutter : Shutter | None, optional
        The front-end shutter device. If None, will pull from global namespace.
    photon_shutter : Shutter | None, optional
        The photon shutter device. If None, will pull from global namespace.
    """

    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)
    germ = ensure_available(GeRMDetector, germ=germ)
    germ.save_as_hdf()

    def _body():
        if use_shutter:
            yield from ensure_shutter_open(fe_shutter, allow_actuation=True)
            yield from ensure_shutter_open(photon_shutter, allow_actuation=True)

        yield from bps.mv(germ.driver.acquire_time, count_time)

        _md = {"plan_name": "edxd_grid_scan"}
        if description is not None:
            _md["description"] = description
        yield from bp.grid_scan(
            [germ],
            outer_motor, outer_start, outer_stop, outer_num_steps,
            inner_motor, inner_start, inner_stop, inner_num_steps,
            md=_md,
            snake_axes=snake
        )

    def _cleanup():
        if use_shutter:
            yield from ensure_shutter_closed(fe_shutter)
            yield from ensure_shutter_closed(photon_shutter)
        if reset_position:
            yield from bps.mv(outer_motor, outer_start)
            yield from bps.mv(inner_motor, inner_start)

    return (yield from finalize_wrapper(_body(), _cleanup()))


def edxd_custom_pos_list_grid(
    outer_motor: Steppable,
    outer_positions_list: list[float],
    inner_motor: Steppable,
    inner_start: float,
    inner_stop: float,
    inner_num_steps: int,
    count_time: float,
    snake: bool = False,
    use_shutter: bool = True,
    reset_position: bool = True,
    description: str | None = None,
    germ: GeRMDetector | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
):
    """Plan stub for a custom position list grid scan for the EDXD detector."""

    germ = ensure_available(GeRMDetector, germ=germ)
    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)
    germ.save_as_hdf()

    def _body():
        if use_shutter:
            yield from ensure_shutter_open(fe_shutter, allow_actuation=True)
            yield from ensure_shutter_open(photon_shutter, allow_actuation=True)

        yield from bps.mv(germ.driver.acquire_time, count_time)

        _md = {"plan_name": "edxd_custom_pos_list_grid"}
        if description is not None:
            _md["description"] = description
        yield from bp.list_grid_scan(
            [germ],
            outer_motor, outer_positions_list,
            inner_motor, np.linspace(inner_start, inner_stop, inner_num_steps),
            md=_md,
            snake_axes=snake
        )

    def _cleanup():
        if use_shutter:
            yield from ensure_shutter_closed(fe_shutter)
            yield from ensure_shutter_closed(photon_shutter)
        if reset_position:
            yield from bps.mv(inner_motor, inner_start)

    return (yield from finalize_wrapper(_body(), _cleanup()))


def edxd_2theta_tilt(
    theta: float, 
    move_out: bool = False,
    edxd_table: EDXDTable | None = None,
    collimator_table: CollimatorTable | None = None,
):
    """Plan stub to tilt the EDXD detector to a specified 2-theta angle.

    Parameters
    ----------
    theta : float
        The desired 2-theta angle for the EDXD detector.
    move_out : bool, default False
        Whether to move the EDXD table out before tilting.
    edxd_table : EDXDTable | None, optional
        The EDXD table device. If None, will pull from global namespace.
    collimator_table : CollimatorTable | None, optional
        The collimator table device. If None, will pull from global namespace.
    """

    edxd_table = ensure_available(EDXDTable, edxd_table=edxd_table)
    collimator_table = ensure_available(CollimatorTable, collimator_table=collimator_table)

    # TODO: Pull out into human readable file
    x_out = -270.0
    x0 = 0.0
    coll_x0 = -54
    d1 = 425
    ds = 263
    R_value = 70.5
    beta_rad = np.deg2rad(theta)

    if move_out:
        yield from bps.mv(edxd_table.x, x_out)
        return

    yield from bps.mv(
        edxd_table.x, x0,
        edxd_table.rx, theta
    )
    d2: float = yield from bps.rd(edxd_table.z.user_readback)
    delta_theta = np.arctan(24/(d2 - ds))

    yield from bps.mv(
        edxd_table.y, -d2 * np.tan(beta_rad),
        collimator_table.slits_pitch, (theta - np.rad2deg(delta_theta)),
        collimator_table.y_coarse, -((d1 + R_value * np.sin(beta_rad - delta_theta) - (d1 - ds) * np.cos(beta_rad - delta_theta)) * np.tan(beta_rad) + (d1 - ds) * np.tan(beta_rad - delta_theta) - R_value * (1 - np.cos(beta_rad - delta_theta))),
        collimator_table.x, coll_x0
    )


def edxd_calib_scan(
    start: float,
    stop: float,
    count_time: float = 3000,
    description: str | None = None,
    sample_tower: SampleTower | None = None,
    germ: GeRMDetector | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
):
    germ = ensure_available(GeRMDetector, germ=germ)
    sample_tower = ensure_available(SampleTower, sample_tower=sample_tower)

    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)

    _md = {"plan_name": "edxd_calib_scan"}
    if description is not None:
        _md["description"] = description

    initial_z1 = yield from bps.rd(sample_tower.z1.user_readback)

    @bpp.stage_decorator([germ])
    @bpp.run_decorator(md=_md)
    def _body():
        yield from ensure_shutter_open(fe_shutter, allow_actuation=True)
        yield from ensure_shutter_open(photon_shutter, allow_actuation=True)
        yield from bps.mv(germ.driver.acquire_time, count_time)
        count_status = yield from bps.trigger(germ, wait=False, group="germ")

        sweep_to_stop = True
        while not count_status.done:
            move_sts = yield from bps.abs_set(sample_tower.z1, stop if sweep_to_stop else start, wait=False)
            while move_sts.done is False:
                yield from bps.sleep(0.1)
                if count_status.done:
                    break
            sweep_to_stop = False

        yield from bps.mv(sample_tower.z1.motor_stop, 1)

        yield from bps.create(name="primary")
        reading = yield from bps.read(germ)
        yield from bps.save()
        return reading

    def _cleanup():
        yield from ensure_shutter_closed(fe_shutter, allow_actuation=True)
        yield from ensure_shutter_closed(photon_shutter, allow_actuation=True)
        yield from bps.mv(sample_tower.z1, initial_z1)


    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))