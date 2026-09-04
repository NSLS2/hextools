"""Motion stages and related utility functions for the HEX beamline."""

import asyncio

from ophyd_async.core import (
    AsyncMovable,
    AsyncStatus,
    DeviceMock,
    StandardReadable,
    StrictEnum,
    callback_on_mock_put,
    default_mock_class,
    derived_signal_r,
    set_mock_put_proceeds,
    set_mock_value,
    wait_for_value,
)
from ophyd_async.core import (
    StandardReadableFormat as Format,
)
from ophyd_async.epics.core import (
    EpicsDevice,
    epics_signal_r,
    epics_triggerable_command,
)
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


class OpticsTable(StandardReadable, EpicsDevice):
    """HEX optics table."""

    def __init__(self, prefix: str, name="optics_table"):
        super().__init__(prefix, name=name)
        with self.add_children_as_readables(Format.CHILD):
            self.x2 = AsyncEpicsMotor(prefix + "X2}Mtr", name="x2")
            self.y2 = AsyncEpicsMotor(prefix + "Y2}Mtr", name="y2")
            self.rx3 = AsyncEpicsMotor(prefix + "RX3}Mtr", name="rx3")
            self.ry3 = AsyncEpicsMotor(prefix + "RY3}Mtr", name="ry3")
            self.x3 = AsyncEpicsMotor(prefix + "X3}Mtr", name="x3")
            self.y3 = AsyncEpicsMotor(prefix + "Y3}Mtr", name="y3")
            self.ry4 = AsyncEpicsMotor(prefix + "RY4}Mtr", name="ry4")
            self.x4 = AsyncEpicsMotor(prefix + "X4}Mtr", name="x4")
            self.a1 = AsyncEpicsMotor(prefix + "A1}Mtr", name="a1")
            self.z0 = AsyncEpicsMotor(prefix + "Z0}Mtr", name="z0")


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


class SampleTower(StandardReadable, EpicsDevice):
    """HEX sample tower."""

    def __init__(self, prefix: str, name: str = "sample_tower"):
        super().__init__(prefix, name=name)
        with self.add_children_as_readables(Format.CHILD):
            self.y = AsyncEpicsMotor(prefix + "Y}Mtr", name="y")
            self.pitch = AsyncEpicsMotor(prefix + "Rx}Mtr", name="pitch")
            self.roll = AsyncEpicsMotor(prefix + "Rz}Mtr", name="roll")

        # Real motors that combine to give y, pitch, and roll.
        self.x1 = AsyncEpicsMotor(prefix + "X1}Mtr", name="x1")
        self.x2 = AsyncEpicsMotor(prefix + "X2}Mtr", name="x2")
        self.z1 = AsyncEpicsMotor(prefix + "Z1}Mtr", name="z1")
        self.z2 = AsyncEpicsMotor(prefix + "Z2}Mtr", name="z2")
        self.inboard_y = AsyncEpicsMotor(prefix + "Y1}Mtr", name="inboard_y")
        self.outboard_y = AsyncEpicsMotor(prefix + "Y2}Mtr", name="outboard_y")
        self.downstream_y = AsyncEpicsMotor(prefix + "Y3}Mtr", name="downstream_y")


class CameraObjective(StrictEnum):
    """Represents the camera objective in use."""

    LEFT_4MM = "left_4mm"
    RIGHT_2MM = "right_2mm"


class HomeStatus(StrictEnum):
    """Represents the home status of a motor."""

    NOT_HOMED = "Not homed"
    HOMED = "Homed"


class DoubleObjCamera(StandardReadable, EpicsDevice, AsyncMovable[CameraObjective]):
    """HEX double objective camera."""

    def __init__(self, prefix: str, name: str = "double_obj_camera"):
        super().__init__(prefix, name=name)
        self.rotation = AsyncEpicsMotor(prefix + "CamRot}Mtr", name="rotation")
        self.left_focus = AsyncEpicsMotor(prefix + "Focus1}Mtr", name="left_focus")
        self.right_focus = AsyncEpicsMotor(prefix + "Focus2}Mtr", name="right_focus")
        self.obj_selector = AsyncEpicsMotor(prefix + "ObjSel}Mtr", name="obj_selector")
        self.home_obj_selector = epics_triggerable_command(
            prefix + "ObjSel}Start:Home-Cmd", name="home_obj_selector"
        )
        self._obj_selector_home_sts = epics_signal_r(
            HomeStatus,
            prefix + "ObjSel}Sts:HomeCmplt-Sts",
            name="obj_selector_home_sts",
        )
        self._at_right_objective = epics_signal_r(
            bool, prefix + "ObjSel}AtRightObj", name="at_right_objective"
        )
        self._at_left_objective = epics_signal_r(
            bool, prefix + "ObjSel}AtLeftObj", name="at_left_objective"
        )
        self._goto_right_objective = epics_triggerable_command(
            prefix + "ObjSel}Cmd:GotoRight-Cmd", name="goto_right_objective"
        )
        self._goto_left_objective = epics_triggerable_command(
            prefix + "ObjSel}Cmd:GotoLeft-Cmd", name="goto_left_objective"
        )

    @AsyncStatus.wrap
    async def set(self, value: CameraObjective):
        """Move the camera to the specified objective.

        Parameters
        ----------
        value : CameraObjective
            The objective to move the camera to.

        Raises
        ------
        RuntimeError
            If the camera objective selector is not homed.
        """
        if await self._obj_selector_home_sts.get_value() != HomeStatus.HOMED:
            raise RuntimeError(
                "Camera objective selector is not homed. "
                "Please home it before moving to a specific objective."
            )

        if value == CameraObjective.LEFT_4MM:
            await self._goto_left_objective.execute()
            rb_check = self._at_left_objective
        else:
            await self._goto_right_objective.execute()
            rb_check = self._at_right_objective

        await wait_for_value(rb_check, True, timeout=None)


class WideFOVCamera(StandardReadable, EpicsDevice):
    """HEX wide field of view camera."""

    def __init__(self, prefix: str, name: str = "wide_fov_camera"):
        super().__init__(prefix, name=name)
        with self.add_children_as_readables(Format.CHILD):
            self.focus = AsyncEpicsMotor(prefix + "Focus}Mtr", name="focus")
            self.rotation = AsyncEpicsMotor(prefix + "CamRot}Mtr", name="rotation")
