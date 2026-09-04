"""Flyer classes for XPD beamline at NSLS-II."""

import asyncio

from ophyd_async.core import (
    ConfinedModel,
    FlyableLogic,
    FlyMotorInfo,
    wait_for_value,
)
from ophyd_async.fastcs.panda import CommonPandaBlocks, PandaPcompDirection

from .motors import get_encoder_value_from_pos


class SingleAxisFlyscanInfo(ConfinedModel):
    """Information for a single axis flyscan.

    Attributes
    ----------
    start : int
        The start position for the flyscan, in encoder counts
    num_pulses : int
        The number of pulses to send during the flyscan
    direction : PandaPcompDirection
        The direction of the flyscan, either positive or negative
    pulse_width : float | int
        The width of each pulse, in counts for position based scans, s for time based
    pulse_step : float | int
        The step between pulses, in counts for position based scans, s for time based
    time_based : bool
        If true, equally spaced in time triggers. Otherwise, equally spaced in position
    """

    start: int
    num_pulses: int
    direction: PandaPcompDirection
    pulse_width: float | int
    pulse_step: float | int
    time_based: bool


class SingleAxisFlyableLogic(FlyableLogic[SingleAxisFlyscanInfo, None]):
    """Logic class for setting up a PandABox for a single axis flyscan."""

    def __init__(self, panda: CommonPandaBlocks) -> None:
        self.panda = panda

    async def prepare(self, value: SingleAxisFlyscanInfo):
        pcomp = self.panda.pcomp[1]
        pulse = self.panda.pulse[1]
        coros = [
            pcomp.dir.set(value.direction),
            pcomp.start.set(value.start),
        ]
        if not value.time_based:
            coros.extend(
                [
                    pcomp.pulses.set(value.num_pulses),
                    pcomp.width.set(int(value.pulse_width)),
                    pcomp.step.set(int(value.pulse_step)),
                    pulse.pulses.set(1),
                    # TODO: Come up with how we can get always valid values
                    # for these. Must be shorter than the pcomp pulses.
                    pulse.width.set(0.000001),
                    pulse.step.set(0.000002),
                ]
            )
        else:
            coros.extend(
                [
                    pcomp.pulses.set(1),
                    pcomp.width.set(1),
                    pcomp.step.set(2),
                    pulse.pulses.set(value.num_pulses),
                    pulse.width.set(value.pulse_width),
                    pulse.step.set(value.pulse_step),
                ]
            )
        await asyncio.gather(*coros)

    async def kickoff(self) -> None:
        await wait_for_value(self.panda.pcomp[1].active, True, timeout=1)

    async def complete(self, timeout: float | None = None) -> None:
        await wait_for_value(self.panda.pcomp[1].active, False, timeout=timeout)

    async def stop(self):
        await wait_for_value(self.panda.pcomp[1].active, False, timeout=1)


def calculate_move_time_for_flyscan(
    travel_distance: float,
    max_motor_velocity: float,
    num_images: int,
    exposure_time: float,
    acq_time_overhead: float = 0.001,
) -> float:
    """Calculate the time for a motor move during a flyscan.

    The motor travel and acquisition happen concurrently. The total time is
    whichever takes longer: the motor travel time or the total acquisition time.

    Parameters
    ----------
    travel_distance : float
        The distance the motor will travel during the flyscan.
    max_motor_velocity : float
        The maximum velocity of the motor.
    num_images : int
        The number of images to acquire during the flyscan.
    exposure_time : float
        The maximum acquisition time for a single image.
    acq_time_overhead : float, default 0.001
        An overhead time per image to add to each acquisition.

    Returns
    -------
    float
        The time for the motor move during the flyscan.
    """
    fastest_possible_move_time = travel_distance / max_motor_velocity
    total_acq_time = num_images * (exposure_time + acq_time_overhead)

    return max(fastest_possible_move_time, total_acq_time)


def construct_fly_info_models(
    num_pulses: int,
    max_exposure_time: float,
    start_position: float,
    stop_position: float,
    encoder_resolution: float,
    max_motor_velocity: float,
    encoder_pos_at_zero: int = 0,
    acq_time_overhead: float = 0.001,
    time_based: bool = False,
) -> tuple[SingleAxisFlyscanInfo, FlyMotorInfo]:
    """Construct the fly info models for a single axis flyscan.

    Returns
    -------
    tuple[SingleAxisFlyscanInfo, FlyMotorInfo]
        The fly info models for a single axis flyscan.
    """
    start_in_counts = get_encoder_value_from_pos(
        start_position, encoder_resolution, encoder_pos_at_zero
    )
    stop_in_counts = get_encoder_value_from_pos(
        stop_position, encoder_resolution, encoder_pos_at_zero
    )
    travel_counts = abs(stop_in_counts - start_in_counts)
    move_time = calculate_move_time_for_flyscan(
        abs(stop_position - start_position),
        max_motor_velocity,
        num_pulses,
        max_exposure_time,
        acq_time_overhead=acq_time_overhead,
    )

    if not time_based:
        if travel_counts % (num_pulses - 1) != 0:
            # Subtract one from pulses because the num of steps is one less than
            # the number of pulses
            raise ValueError(
                f"Travel distance in counts ({travel_counts}) is not evenly divisible "
                f"by the number of pulses ({num_pulses - 1})."
            )
        elif travel_counts < (num_pulses - 1) * 2:
            raise ValueError(
                f"Travel distance in counts ({travel_counts}) is less than the minimum"
                f" required for the number of pulses ({num_pulses}). At least two "
                f"counts are required between each pulse, one for livetime, one "
                f"for deadtime({2 * (num_pulses - 1)})."
            )
        pulse_width = 1
        pulse_step = travel_counts // (num_pulses - 1)
    else:
        pulse_width = max_exposure_time + acq_time_overhead
        pulse_step = move_time / num_pulses

    flyer_info = SingleAxisFlyscanInfo(
        start=start_in_counts,
        num_pulses=num_pulses,
        direction=PandaPcompDirection.POSITIVE
        if stop_position > start_position
        else PandaPcompDirection.NEGATIVE,
        pulse_width=pulse_width,
        pulse_step=pulse_step,
        time_based=time_based,
    )

    motor_info = FlyMotorInfo(
        start_position=start_position,
        end_position=stop_position,
        time_for_move=move_time,
    )
    return flyer_info, motor_info
