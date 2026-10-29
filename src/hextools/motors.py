"""Motion stages and related utility functions for the HEX beamline."""

import asyncio
from typing import Any

import bluesky.plan_stubs as bps
from ophyd_async.core import (
    AsyncMovable,
    AsyncStatus,
    SignalR,
    StandardReadable,
    StrictEnum,
    default_mock_class,
    derived_signal_r,
    observe_signals_value,
    wait_for_value,
)
from ophyd_async.core import (
    StandardReadableFormat as Format,
)
from ophyd_async.epics.core import (
    EpicsDevice,
    epics_signal_r,
    epics_signal_rw,
    epics_triggerable_command,
)
from ophyd_async.epics.motor import (
    Motor as AsyncEpicsMotor,
)
from ophyd_async.epics.motor import (
    VeloAndAcclRespectingMotorMock,
)

# Make every Motor respect velocity/acceleration in mock mode by default.
default_mock_class(VeloAndAcclRespectingMotorMock)(AsyncEpicsMotor)


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


class RotationMotor(AsyncEpicsMotor):
    """A motor that can be used for rotation scans.

    Adds derived signal to find the number of encoder counts per revolution.
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
        return int(360.0 / encoder_resolution)


class SampleSmall(StandardReadable, EpicsDevice):
    """HEX small sample stage with independent X and Y motors."""

    def __init__(self, prefix: str, name: str = "sample_small"):
        super().__init__(prefix, name=name)
        with self.add_children_as_readables(Format.CHILD):
            self.x = AsyncEpicsMotor(prefix + "X}Mtr", name="x")
            self.y = AsyncEpicsMotor(prefix + "Y}Mtr", name="y")


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

        # TODO: Get this prefix adjusted so it doesn't need to be ah
        self.ry2 = RotationMotor("XF:27IDF-OP:1{MC:5-Ax:4}Mtr", name="ry2")


class OpticsTable(StandardReadable, EpicsDevice):
    """HEX optics table."""

    def __init__(self, prefix: str, name="optics_table"):
        super().__init__(prefix, name=name)
        with self.add_children_as_readables(Format.CHILD):
            self.x2 = AsyncEpicsMotor(prefix + "X2}Mtr", name="x2")
            self.y2 = AsyncEpicsMotor(prefix + "Y2}Mtr", name="y2")
            self.rx3 = AsyncEpicsMotor(prefix + "Rx3}Mtr", name="rx3")
            self.ry3 = AsyncEpicsMotor(prefix + "Ry3}Mtr", name="ry3")
            self.x3 = AsyncEpicsMotor(prefix + "X3}Mtr", name="x3")
            self.y3 = AsyncEpicsMotor(prefix + "Y3}Mtr", name="y3")
            self.ry4 = AsyncEpicsMotor(prefix + "Ry4}Mtr", name="ry4")
            self.x4 = AsyncEpicsMotor(prefix + "X4}Mtr", name="x4")
            self.a1 = AsyncEpicsMotor(prefix + "A1}Mtr", name="a1")
            self.z0 = AsyncEpicsMotor(prefix + "Z0}Mtr", name="z0")


class CameraObjective(StrictEnum):
    """Represents the camera objective in use."""

    RIGHT_2MM = "Right Objective"
    LEFT_4MM = "Left Objective"


class HomeStatus(StrictEnum):
    """Represents the home status of a motor."""

    NOT_HOMED = "Not homed"
    HOMED = "Homed"


class FOV_2_4_mm_Camera(  # noqa: N801
    StandardReadable, EpicsDevice, AsyncMovable[CameraObjective | str]
):
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
        self.objective = epics_signal_rw(
            CameraObjective, prefix + "ObjSel}Objective", name="objective"
        )

        self._at_right_objective = epics_signal_r(
            bool, prefix + "ObjSel}AtRightObj", name="at_right_objective"
        )
        self._at_left_objective = epics_signal_r(
            bool, prefix + "ObjSel}AtLeftObj", name="at_left_objective"
        )
        self._stop_requested = asyncio.Event()
        self._arrival_grace = 2.0
        self._start_timeout = 5.0

    def set(self, value: CameraObjective | str) -> AsyncStatus:
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
        # Reset on the call, not in the task, so a stop() at any later point is honored.
        self._stop_requested = asyncio.Event()
        return AsyncStatus(self._set(value))

    async def _set(self, value: CameraObjective | str):
        if await self._obj_selector_home_sts.get_value() != HomeStatus.HOMED:
            raise RuntimeError(
                "Camera objective selector is not homed. "
                "Please home it before moving to a specific objective."
            )

        # Attempt to auto-deduce the CameraObjective from a string input.
        if isinstance(value, str):
            for possible_value in CameraObjective:
                if value.upper() in possible_value.name:
                    value = possible_value
                    break

        if not isinstance(value, CameraObjective):
            raise ValueError(
                f"Invalid objective value: {value}. "
                "Must be a CameraObjective or a string matching"
                "one of its names."
            )

        if value == CameraObjective.RIGHT_2MM:
            at_objective = self._at_right_objective
        else:
            at_objective = self._at_left_objective
        stopped = AsyncStatus(self._stop_requested.wait())
        watch = asyncio.create_task(self._reached(at_objective, stopped))
        try:
            # Let the watch subscribe first, so a move that ends during the put is seen.
            await asyncio.sleep(0)
            if not self._stop_requested.is_set():
                await self.objective.set(value)
                if self._stop_requested.is_set():
                    # stop() came while the command was in flight, maybe before motion.
                    await self.obj_selector.motor_stop.set(1)
            reached = await watch
        finally:
            watch.cancel()
            stopped.task.cancel()
        if not reached:
            raise RuntimeError(
                f"{self.name} objective selector stopped before reaching {value.value}"
            )

    async def _reached(self, at_objective: SignalR[bool], stopped: AsyncStatus) -> bool:
        """Whether the selector reaches the objective before it stops moving."""
        dmov = self.obj_selector.motor_done_move
        signals: list[SignalR[Any]] = [at_objective, dmov]
        moved = False
        try:
            async with asyncio.timeout(self._start_timeout) as start_deadline:
                async for signal, reading in observe_signals_value(
                    *signals, done_status=stopped
                ):
                    if signal is at_objective and reading:
                        return True
                    if signal is dmov:
                        if not reading:
                            moved = True
                            start_deadline.reschedule(None)
                        elif moved:
                            break
        except TimeoutError:
            return bool(await at_objective.get_value())
        if stopped.done:
            return bool(await at_objective.get_value())
        # AtLeftObj/AtRightObj can update just after the motor reports done.
        try:
            await wait_for_value(at_objective, True, timeout=self._arrival_grace)
        except TimeoutError:
            return False
        return True

    async def stop(self, success: bool = False):
        """Stop the objective selector, which the IOC moves to the chosen objective."""
        self._stop_requested.set()
        await self.obj_selector.stop()


class FOV_20_40_mm_Camera(StandardReadable, EpicsDevice):  # noqa: N801
    """HEX wide field of view camera."""

    def __init__(self, prefix: str, name: str = "fov_20_40_mm_camera"):
        super().__init__(prefix, name=name)
        with self.add_children_as_readables(Format.CHILD):
            self.focus = AsyncEpicsMotor(prefix + "Focus}Mtr", name="focus")
            self.rotation = AsyncEpicsMotor(prefix + "CamRot}Mtr", name="rotation")


class EDXDTable(StandardReadable, EpicsDevice):
    """HEX EDXD table."""

    def __init__(self, prefix: str, name: str = "edxd_table"):
        super().__init__(prefix, name=name)
        with self.add_children_as_readables(Format.CHILD):
            self.x = AsyncEpicsMotor(prefix + "X}Mtr", name="x")
            self.y = AsyncEpicsMotor(prefix + "Y}Mtr", name="y")
            self.z = AsyncEpicsMotor(prefix + "Z}Mtr", name="z")
            self.rx = AsyncEpicsMotor(prefix + "Rx}Mtr", name="rx")


class CollimatorTable(StandardReadable, EpicsDevice):
    """HEX collimator table."""

    def __init__(self, prefix: str, name: str = "collimator"):
        super().__init__(prefix, name=name)
        with self.add_children_as_readables(Format.CHILD):
            self.x = AsyncEpicsMotor(prefix + "CMT:1-Ax:X}Mtr", name="x")
            self.y_coarse = AsyncEpicsMotor(prefix + "CMT:1-Ax:Yc}Mtr", name="y_coarse")
            self.y_fine = AsyncEpicsMotor(prefix + "CMT:1-Ax:Yf}Mtr", name="y_fine")
            # TODO: Consider renaming PV to follow format of above.
            self.slits_pitch = AsyncEpicsMotor(
                prefix + "Slt:CMT-Ax:Pitch}Mtr", name="slits_pitch"
            )


def move_motor(motor: AsyncEpicsMotor, value: float, timeout: float | None = None):
    """Move the specified motor to the given position.

    Wraps the `bps.mv` plan stub to create a plan signature more conducive to creating
    a user-friendly GUI input.

    Parameters
    ----------
    motor : AsyncEpicsMotor
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
