"""Where a single-axis flyscan's triggers land, for both trigger modes."""

import numpy as np
import pytest
from ophyd_async.fastcs.panda import PandaPcompDirection

from hextools.exporters.utils import average_per_projection
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
    )


@pytest.mark.parametrize("time_based", [False, True])
@pytest.mark.parametrize(("start", "stop"), [(0.0, 180.0), (180.0, 0.0)])
def test_triggers_span_start_to_stop(time_based, start, stop):
    flyer_info, motor_info = _models(19, 1, time_based, start, stop)
    angles = _trigger_angles(flyer_info, motor_info, start)
    np.testing.assert_allclose(angles, np.linspace(0, stop - start, 19), atol=1e-9)


# 19 x 5 triggers over 0-188 deg: 188000 counts split evenly into 94 steps.
@pytest.mark.parametrize("time_based", [False, True])
def test_averaged_triggers_span_start_to_stop(time_based):
    flyer_info, motor_info = _models(19, 5, time_based, stop=188.0)
    angles = _trigger_angles(flyer_info, motor_info, 0.0)
    np.testing.assert_allclose(angles, np.linspace(0, 188, 95), atol=1e-9)


def test_averaged_projection_angle_is_mean_of_its_images():
    flyer_info, motor_info = _models(19, 5, time_based=True)
    angles = _trigger_angles(flyer_info, motor_info, 0.0)
    groups = angles.reshape(19, 5)
    np.testing.assert_allclose(
        average_per_projection(groups, 19, 5), groups.sum(axis=1) / 5
    )
    assert angles[0] == pytest.approx(0.0)
    assert angles[-1] == pytest.approx(180.0)


def test_single_time_based_pulse_fires_at_start():
    flyer_info, motor_info = _models(1, 1, time_based=True)
    assert flyer_info.num_pulses == 1
    np.testing.assert_allclose(_trigger_angles(flyer_info, motor_info, 0.0), [0.0])
