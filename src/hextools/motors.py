"""Motion stages and related utility functions for the HEX beamline."""

import asyncio

from ophyd_async.core import (
    DeviceMock,
    callback_on_mock_put,
    default_mock_class,
    derived_signal_r,
    set_mock_put_proceeds,
    set_mock_value,
)
from ophyd_async.epics.core import EpicsDevice
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor


def get_encoder_value_from_pos(
    current_position: float, encoder_resolution: float, encoder_pos_at_zero: int
) -> int:
    """Calculate the encoder value from a motor position.

    Parameters
    ----------
    current_position : float
        The current position of the motor.
    encoder_resolution : float
        The resolution of the encoder in counts per degree.
    encoder_pos_at_zero : int
        The encoder position corresponding to 0 degrees.

    Returns
    -------
    int
        The encoder value corresponding to the given motor position.
    """
    return int(current_position / encoder_resolution + encoder_pos_at_zero)


class OpticsTable(EpicsDevice):
    """HEX optics table."""

    def __init__(self):
        super().__init__("XF:27ID1A-OP:1{OPT:1-Ax:", name="optics_table")
        self.x2 = AsyncEpicsMotor("X2}Mtr", name="x2")
        self.y2 = AsyncEpicsMotor("Y2}Mtr", name="y2")
        self.rx3 = AsyncEpicsMotor("RX3}Mtr", name="rx3")
        self.ry3 = AsyncEpicsMotor("RY3}Mtr", name="ry3")
        self.x3 = AsyncEpicsMotor("X3}Mtr", name="x3")
        self.y3 = AsyncEpicsMotor("Y3}Mtr", name="y3")
        self.ry4 = AsyncEpicsMotor("RY4}Mtr", name="ry4")
        self.x4 = AsyncEpicsMotor("X4}Mtr", name="x4")


# TODO: Get this upstreamed to ophyd_async and remove it from here.
# It is a general utility that is not specific to HEX.
class VelocityRespectingMotorMock(DeviceMock[AsyncEpicsMotor]):
    """Mock behaviour that respects motor velocity and acceleration time."""

    async def connect(self, device: AsyncEpicsMotor) -> None:
        """Mock signals to simulate a move respecting velocity and acceleration."""
        set_mock_value(device.velocity, 10)
        set_mock_value(device.max_velocity, 100)
        set_mock_value(device.acceleration_time, 0.01)

        # Motor starts in "done" state (not moving)
        set_mock_value(device.motor_done_move, 1)

        async def _do_move(target: float):
            current = await device.user_readback.get_value()
            velocity = await device.velocity.get_value()
            acceleration_time = await device.acceleration_time.get_value()
            move_time = abs(target - current) / velocity + 2 * acceleration_time
            set_mock_value(device.motor_done_move, 0)
            elapsed = 0.0
            while elapsed < move_time:
                await asyncio.sleep(min(1.0, move_time - elapsed))
                elapsed += 1.0
                fraction = min(elapsed / move_time, 1.0)
                position = current + (target - current) * fraction
                set_mock_value(device.user_readback, position)
            set_mock_value(device.user_readback, target)
            set_mock_value(device.motor_done_move, 1)
            set_mock_put_proceeds(device.user_setpoint, True)

        def _on_setpoint_write(value):
            set_mock_put_proceeds(device.user_setpoint, False)
            asyncio.ensure_future(_do_move(value))

        callback_on_mock_put(device.user_setpoint, _on_setpoint_write)


@default_mock_class(VelocityRespectingMotorMock)
class RotationMotor(AsyncEpicsMotor):
    """A motor that can be used for rotation scans.

    This class is a subclass of the AsyncEpicsMotor class and is used to represent
    a motor that can be used for rotation scans. It has additional attributes and
    methods that are specific to rotation scans.
    """

    def __init__(self, prefix: str, name: str = ""):
        super().__init__(prefix, name=name)
        self.encoder_counts_per_rev = derived_signal_r(
            self.get_encoder_counts_per_rev,
            derived_units="counts",
            derived_precision=0,
            encoder_resolution=self.encoder_resolution,
        )

    def get_encoder_counts_per_rev(self, encoder_resolution: float) -> int:
        """Calculate the number of encoder counts per revolution.

        Parameters
        ----------
        encoder_resolution : float
            The resolution of the encoder in counts per degree.

        Returns
        -------
        int
            The number of encoder counts per revolution.
        """
        return int(360.0 * encoder_resolution)


class SampleTower(EpicsDevice):
    """HEX sample tower."""

    def __init__(self):
        super().__init__("XF:27ID1A-OP:1{SMPL:1-Ax:", name="sample_tower")
        self.y = AsyncEpicsMotor("Y}Mtr", name="y")
        self.pitch = AsyncEpicsMotor("Rx}Mtr", name="pitch")
        self.roll = AsyncEpicsMotor("Rz}Mtr", name="roll")

        # Real motors that combine to give y, pitch, and roll.
        self.x1 = AsyncEpicsMotor("X1}Mtr", name="x1")
        self.x2 = AsyncEpicsMotor("X2}Mtr", name="x2")
        self.z1 = AsyncEpicsMotor("Z1}Mtr", name="z1")
        self.z2 = AsyncEpicsMotor("Z2}Mtr", name="z2")
        self.inboard_y = AsyncEpicsMotor("Y1}Mtr", name="inboard_y")
        self.outboard_y = AsyncEpicsMotor("Y2}Mtr", name="outboard_y")
        self.downstream_y = AsyncEpicsMotor("Y3}Mtr", name="downstream_y")
