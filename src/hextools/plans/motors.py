"""Plans for moving motors and building step trajectories."""

from typing import (
    Any,
    NamedTuple,
)

import bluesky.plan_stubs as bps
from bluesky import (
    plan_patterns,
)
from bluesky.protocols import (
    Movable,
)
from ophyd_async.epics.motor import (
    Motor as AsyncEpicsMotor,
)
from ophyd_async.sim import (
    SimMotor,
)


def move_motor(
    motor: AsyncEpicsMotor | SimMotor, value: float, timeout: float | None = None
):
    """Move the specified motor to the given position.

    Wraps the `bps.mv` plan stub to create a plan signature more conducive to creating
    a user-friendly GUI input.

    Parameters
    ----------
    motor : AsyncEpicsMotor | SimMotor
        The motor to move.
    value : float
        The target position for the motor.
    timeout : float, optional
        Maximum time to wait for the motor to reach the target position.
        Defaults to None.

    Raises
    ------
    RuntimeError
        If the motor fails to reach the target position within the specified timeout.
    """
    yield from bps.mv(motor, value, timeout=timeout)


class NDAxis(NamedTuple):
    """One axis of an N-dimensional step trajectory."""

    axis: Movable[float]
    start: float
    stop: float
    num_points: int


def construct_nd_trajectory(
    *axes: NDAxis, snake_axes: bool | list = False, num_iterations: int = 1
):
    """Construct an ND trajectory for the specified axes."""
    # Extract the step axes from the positional arguments.
    motors = [axis.axis for axis in axes]

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
            pattern_args += [*axis, axis.axis in snaked]
        grid = plan_patterns.outer_product(pattern_args)
    else:
        grid = [{}]  # When no axes are specified, create a single-point grid.

    # Repeat the trajectory the specified number of times.
    trajectory = grid * num_iterations
    return trajectory
