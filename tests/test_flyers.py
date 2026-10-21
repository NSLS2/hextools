"""Where a single-axis flyscan's triggers land, for both trigger modes."""

import numpy as np
import pytest
from ophyd_async.fastcs.panda import PandaPcompDirection

from hextools.flyers import construct_fly_info_models

RES = 0.001  # deg per encoder count


def _trigger_angles(flyer_info, motor_info, start):
    """Angles the PandA records at each trigger, relative to ``start``."""
    k = np.arange(flyer_info.num_pulses)
    if not flyer_info.time_based:
        sign = 1 if flyer_info.direction == PandaPcompDirection.POSITIVE else -1
        counts = flyer_info.start + sign * k * flyer_info.pulse_step
        return counts * flyer_info.position_scale + flyer_info.position_offset
    # Constant velocity between start_position and end_position over time_for_move.
    velocity = (
        motor_info.end_position - motor_info.start_position
    ) / motor_info.time_for_move
    positions = motor_info.start_position + velocity * k * flyer_info.pulse_step
    return positions - start


def _models(num_projections, images_to_average, time_based, start=0.0, stop=180.0):
    extra = (
        {"pulses_per_projection": images_to_average} if images_to_average > 1 else {}
    )
    return construct_fly_info_models(
        num_pulses=num_projections * images_to_average,
        max_exposure_time=0.01,
        start_position=start,
        stop_position=stop,
        encoder_resolution=RES,
        max_motor_velocity=10.0,
        current_position=0.0,
        current_enc_position=0,
        time_based=time_based,
        **extra,
    )


@pytest.mark.parametrize("time_based", [False, True])
@pytest.mark.parametrize(("start", "stop"), [(0.0, 180.0), (180.0, 0.0)])
def test_triggers_span_start_to_stop(time_based, start, stop):
    flyer_info, motor_info = _models(19, 1, time_based, start, stop)
    angles = _trigger_angles(flyer_info, motor_info, start)
    np.testing.assert_allclose(angles, np.linspace(0, stop - start, 19), atol=1e-9)


@pytest.mark.parametrize("time_based", [False, True])
def test_averaged_projections_centre_on_requested_angles(time_based):
    flyer_info, motor_info = _models(19, 5, time_based)
    angles = _trigger_angles(flyer_info, motor_info, 0.0)
    np.testing.assert_allclose(
        angles.reshape(19, 5).mean(axis=1), np.linspace(0, 180, 19), atol=1e-9
    )
    # Every trigger is evenly spaced, groups included.
    np.testing.assert_allclose(np.diff(angles), 2.0, atol=1e-9)


def test_pulses_must_split_into_projections():
    with pytest.raises(ValueError, match="pulses_per_projection"):
        construct_fly_info_models(
            num_pulses=96,
            max_exposure_time=0.01,
            start_position=0.0,
            stop_position=180.0,
            encoder_resolution=RES,
            max_motor_velocity=10.0,
            current_position=0.0,
            current_enc_position=0,
            pulses_per_projection=5,
        )


def test_single_time_based_pulse_fires_at_start():
    flyer_info, motor_info = _models(1, 1, time_based=True)
    assert flyer_info.num_pulses == 1
    np.testing.assert_allclose(_trigger_angles(flyer_info, motor_info, 0.0), [0.0])
