"""Filter device with multiple positions."""

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, TypeAlias

import bluesky.plan_stubs as bps
from bluesky.protocols import Reading
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

from ..utils import ensure_available
from .shutter import PPSShutter, ShutterStatus, ensure_shutter_closed


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
    """Starts at pass through; in position while the motor rests on a setting."""

    #: How close (motor units) the readback must be to a setting's position.
    tolerance = 0.01

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

        motor.user_readback.subscribe_reading(_update_in_position)


@default_mock_class(FilterMock)
class Filter(StandardReadable, EpicsDevice, AsyncMovable[FilterPosition]):
    """Filter device with multiple positions.

    This class represents a filter device with multiple positions, each corresponding
    to a specific filter setting. It provides methods to set the filter position and
    retrieve the description of the current filter setting.

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
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.description = derived_signal_r(
                self._get_description,
                in_pos=self.in_position,
                motor_pos=self.filter_motor.user_readback,
            )
        super().__init__(name=name)

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
            Description of the current filter setting.
        """
        if not in_pos:
            return "out of position"
        closest = min(
            self.positions.items(), key=lambda item: abs(item[1].position - motor_pos)
        )
        return closest[1].description or closest[0].name.lower().replace("_", " ")

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

        move = self.filter_motor.set(self.positions[value].position)
        async for update in forward_watcher_updates([move], self.name, combine=False):
            yield update
        await wait_for_value(self.in_position, True, timeout=10)

    async def stop(self, success: bool = False):
        """Stop the filter motor, so pausing or aborting the RunEngine stops a move."""
        await self.filter_motor.stop()


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


def set_filters(
    filter1_upstream_setting: Filter1UpstreamPosition | None = None,
    filter1_downstream_setting: Filter1DownstreamPosition | None = None,
    filter2_setting: Filter2Position | None = None,
    filter1_upstream: Filter | None = None,
    filter1_downstream: Filter | None = None,
    filter2: Filter | None = None,
    fe_shutter: PPSShutter | None = None,
):
    """Set filter 1 upstream, filter 1 downstream and filter 2 by material.

    Each setting is one of that filter's materials from ``bl_constants.yml`` (for
    example ``"6 mm Cu"``) or ``"Pass Through"``. A filter with no setting does
    not move; the others move together.

    Moving a filter closes the front-end shutter through its interlock. Once the
    filters are in position the interlock is acknowledged, and the front-end
    shutter is reopened if it was open before.

    Parameters
    ----------
    filter1_upstream_setting : Filter1UpstreamPosition, optional
        Material for filter 1 upstream.
    filter1_downstream_setting : Filter1DownstreamPosition, optional
        Material for filter 1 downstream.
    filter2_setting : Filter2Position, optional
        Material for filter 2.
    filter1_upstream, filter1_downstream, filter2 : Filter, optional
        The filter devices. If None, taken from the session namespace.
    fe_shutter : PPSShutter, optional
        The front-end shutter. If None, taken from the session namespace.

    Raises
    ------
    ValueError
        If no setting is given, or a setting is not one of that filter's choices.
    """
    filter1_upstream = ensure_available(Filter, filter1_upstream=filter1_upstream)
    filter1_downstream = ensure_available(Filter, filter1_downstream=filter1_downstream)
    filter2 = ensure_available(Filter, filter2=filter2)
    fe_shutter = ensure_available(PPSShutter, fe_shutter=fe_shutter)

    args = []
    for device, setting, choices in (
        (filter1_upstream, filter1_upstream_setting, Filter1UpstreamPosition),
        (filter1_downstream, filter1_downstream_setting, Filter1DownstreamPosition),
        (filter2, filter2_setting, Filter2Position),
    ):
        if setting is not None:
            # Lookup by value accepts the member or its material string, and raises
            # ValueError for anything that isn't one of this filter's choices.
            args += [device, FilterPosition[choices(setting).name]]
    if not args:
        raise ValueError("Choose a setting for at least one filter.")

    fe_was_open = (yield from bps.rd(fe_shutter.status)) == ShutterStatus.OPEN
    yield from ensure_shutter_closed(fe_shutter)
    yield from bps.mv(*args)
    yield from bps.mv(fe_shutter.interlock_ack, True)
    if fe_was_open:
        yield from bps.mv(fe_shutter, True)
