"""Filter device with multiple positions."""

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, TypeAlias

from bluesky.protocols import Callback, Reading, Subscribable
from ophyd_async.core import (
    AsyncMovable,
    DeviceMock,
    StandardReadable,
    WatchableAsyncStatus,
    default_mock_class,
    derived_signal_r,
    forward_watcher_updates,
    set_mock_value,
    wait_for_value,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice, epics_signal_r
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor

from hextools.bl_constants import (
    build_enums_from_position_lists,
    fetch_beamline_constants,
)
from hextools.devices.shutter import (
    PPSShutter,
    ShutterStatus,
)
from hextools.utils import get_obj_from_ipython_ns


@dataclass
class FilterSetting:
    """Description and motor position for a filter setting."""

    position: float
    description: str | None = None


class FilterPosition(IntEnum):
    """Filter positions."""

    UPPER_LIMIT = 0
    PASS_THROUGH = 1
    POS_1 = 2  # Thickest filter
    POS_2 = 3
    POS_3 = 4
    POS_4 = 5  # Thinnest filter
    LOWER_LIMIT = 6


class FilterMock(DeviceMock["Filter"]):
    """Starts at pass through; in position while the motor rests on a setting.

    Like the beamline, moving the filter trips the front-end shutter's interlock. The
    shutter is ``fe_shutter`` from the session namespace, unless :meth:`interlock`
    gave one.
    """

    #: How close (motor units) the readback must be to a setting's position.
    tolerance = 0.01

    def __init__(self, name: str = "", parent: DeviceMock | None = None) -> None:
        super().__init__(name, parent)
        self._fe_shutter: PPSShutter | None = None

    def interlock(self, shutter: PPSShutter) -> None:
        """Trip ``shutter``'s interlock on every move, instead of ``fe_shutter``'s."""
        self._fe_shutter = shutter

    async def connect(self, device: "Filter") -> None:
        motor = device.filter_motor
        start = device.positions[FilterPosition.PASS_THROUGH].position
        set_mock_value(motor.user_setpoint, start)
        set_mock_value(motor.user_readback, start)

        def _update_in_position(reading: dict[str, Reading[float]]) -> None:
            readback = reading[motor.user_readback.name]["value"]
            set_mock_value(
                device.in_position,
                any(
                    abs(readback - setting.position) <= self.tolerance
                    for setting in device.positions.values()
                ),
            )

        def _trip_interlock(reading: dict[str, Reading[int]]) -> None:
            if reading[motor.motor_done_move.name]["value"]:
                return
            # Looked up per move: the profile defines fe_shutter after the filters.
            shutter = self._fe_shutter or get_obj_from_ipython_ns(
                "fe_shutter", PPSShutter
            )
            if shutter is not None:
                set_mock_value(shutter.status, ShutterStatus.CLOSED)
                set_mock_value(shutter.interlock_ack.acknowledged, False)

        motor.user_readback.subscribe_reading(_update_in_position)
        motor.motor_done_move.subscribe_reading(_trip_interlock)


@default_mock_class(FilterMock)
class Filter(
    StandardReadable, EpicsDevice, AsyncMovable[FilterPosition], Subscribable[str]
):
    """Filter device with multiple positions.

    This class represents a filter device with multiple positions, each corresponding
    to a specific filter setting. It provides methods to set the filter position and
    retrieve the description of the current filter setting. Subscribing follows the
    description.

    Attributes
    ----------
    in_position : epics_signal_r
        Signal indicating whether the filter is in position.
    positions : dict
        Dictionary mapping FilterPosition to FilterSetting.
    filter_motor : AsyncEpicsMotor
        Motor controlling the filter position.
    description : derived_signal_r
        Derived signal providing the description of the current filter setting.
    """

    def __init__(
        self,
        motor_pv: str,
        in_pos_switch_pv: str,
        name: str = "",
    ):
        try:
            self.positions = {
                FilterPosition[pos.upper()]: (
                    FilterSetting(**setting)
                    if isinstance(setting, dict)
                    else FilterSetting(setting)
                )
                for pos, setting in fetch_beamline_constants("filters")[name].items()
            }
        except KeyError as e:
            raise KeyError(
                f"Filter settings for '{name}' not found in beamline constants."
            ) from e
        self.in_position = epics_signal_r(bool, in_pos_switch_pv)
        self.filter_motor = AsyncEpicsMotor(motor_pv)
        self._stopped = False
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.description = derived_signal_r(
                self._get_description,
                in_pos=self.in_position,
                motor_pos=self.filter_motor.user_readback,
            )
        super().__init__(name=name)

    def subscribe_reading(self, function: Callback[str]) -> None:
        """Subscribe to the description of the current setting."""
        self.description.subscribe_reading(function)

    def clear_sub(self, function: Callback[str]) -> None:
        """Remove a description subscription."""
        self.description.clear_sub(function)

    def label(self, position: FilterPosition) -> str:
        """Return the label of ``position``, as used by this filter's position enum."""
        return (
            self.positions[position].description
            or position.name.replace("_", " ").title()
        )

    def position_for(self, setting: FilterPosition | str) -> FilterPosition:
        """Return the position for ``setting``: a position, or its label.

        Raises
        ------
        ValueError
            If ``setting`` is not one of this filter's positions or labels.
        """
        if isinstance(setting, FilterPosition):
            if setting in self.positions:
                return setting
        else:
            for position in self.positions:
                if self.label(position) == setting:
                    return position
        raise ValueError(
            f"{setting!r} is not a setting of {self.name}; choose from "
            f"{[self.label(p) for p in self.positions]}"
        )

    def _get_description(self, in_pos: bool, motor_pos: float) -> str:
        """Get the description of the current filter setting based on motor position.

        Parameters
        ----------
        in_pos : bool
            Indicates whether the filter is in position.
        motor_pos : float
            Current position of the filter motor.

        Returns
        -------
        str
            Label of the closest setting, matching this filter's position enum,
            or ``"out of position"``.
        """
        if not in_pos:
            return "out of position"
        closest = min(
            self.positions.items(), key=lambda item: abs(item[1].position - motor_pos)
        )
        return self.label(closest[0])

    @WatchableAsyncStatus.wrap
    async def set(self, value: FilterPosition):
        """Set the filter to the specified position.

        Parameters
        ----------
        value : FilterPosition
            The desired filter position to set.
        """
        if value not in self.positions:
            raise ValueError(
                f"Invalid filter position: {value}. "
                f"Defined positions are: {list(self.positions.keys())}"
            )

        self._stopped = False
        move = self.filter_motor.set(self.positions[value].position)
        async for update in forward_watcher_updates([move], self.name, combine=False):
            yield update
        if self._stopped:
            # Stopped short of the position, e.g. by a pause; it never arrives.
            return
        await wait_for_value(self.in_position, True, timeout=10)

    async def stop(self, success: bool = False):
        """Stop the filter motor, so pausing or aborting the RunEngine stops a move.

        With ``success``, as when the RunEngine pauses, the move ends without error.
        """
        self._stopped = True
        await self.filter_motor.stop(success=success)


# Dynamically build enums for filter positions from beamline constants
if TYPE_CHECKING:
    Filter1UpstreamPosition: TypeAlias = StrEnum
    Filter1DownstreamPosition: TypeAlias = StrEnum
    Filter2Position: TypeAlias = StrEnum
    Filter3Position: TypeAlias = StrEnum
    Filter4Position: TypeAlias = StrEnum
else:
    (
        Filter1UpstreamPosition,
        Filter1DownstreamPosition,
        Filter2Position,
        Filter3Position,
        Filter4Position,
    ) = build_enums_from_position_lists(
        "filters", always_included_keys=["pass_through"]
    )
