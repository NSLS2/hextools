

from bluesky.preprocessors import finalize_wrapper, reset_positions_decorator
import numpy as np
from ophyd_async.core import StandardMovable

from hextools.detectors.germ import GeRMDetector
from hextools.photon_delivery_system import ensure_shutter_open, Shutter
from hextools.photon_delivery_system.shutter import ensure_shutter_closed
from hextools.utils import ensure_available
from hextools.motors import AsyncMovable, CollimatorTable, EDXDTable
from bluesky import plan_stubs as bps, plans as bp
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor


def configure_test_pulses(
    germ: GeRMDetector,
    use_test_pulses: bool,
    test_pulse_amplitude: int = 100,
    test_pulse_freq: int = 1000,
    test_pulse_count: int = 99999
):
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
    description: str | None = None,
    germ: GeRMDetector | None = None,
):
    germ = ensure_available(GeRMDetector, germ=germ)

    yield from bps.mv(germ.driver.acquire_time, count_time)

    _md = {"plan_name": "edxd_count"}
    if description is not None:
        _md["description"] = description
    yield from bp.count([germ], md=_md)


def edxd_scan(
    motor: StandardMovable[float],
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
            finalize_wrapper(_body(), _cleanup()),
            num=num_iterations,
            delay=time_between_iterations
        )
    )


def edxd_grid_scan(
    outer_motor: StandardMovable[float],
    outer_start: float,
    outer_stop: float,
    outer_num_steps: int,
    inner_motor: StandardMovable[float],
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

def edxd_2theta_tilt(
    theta: float, 
    move_out: bool = False,
    edxd_table: EDXDTable | None = None,
    collimator_table: CollimatorTable | None = None,
):

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



