"""Filter device with multiple positions."""

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from importlib import resources
from typing import TYPE_CHECKING

import bluesky.plan_stubs as bps
import yaml
from ophyd_async.core import (
    AsyncMovable,
    StandardReadable,
    WatchableAsyncStatus,
    derived_signal_r,
    forward_watcher_updates,
    wait_for_value,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice, epics_signal_r
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor

from ..utils import ensure_available
from .shutter import PPSShutter, ShutterStatus


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
        positions: dict[FilterPosition, FilterSetting],
        in_pos_switch_pv: str,
        name: str = "",
    ):
        self.in_position = epics_signal_r(bool, in_pos_switch_pv)
        self.positions = positions
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
        in_pos : int
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


FILTERS_CONFIG_RESOURCE: str = "filters.yml"


def _filters_config() -> dict:
    """Read the packaged ``filters.yml``."""
    source = resources.files(__package__ or "hextools").joinpath(
        FILTERS_CONFIG_RESOURCE
    )
    return yaml.safe_load(source.read_text(encoding="utf-8"))


def load_filters() -> list[Filter]:
    """Create Filter devices from the packaged YAML configuration file.

    The ``filters.yml`` file bundled with the ``hextools`` package is used.
    Each top-level key defines a filter with its ``motor_pv``,
    ``in_position_switch`` PV, and a ``positions`` mapping. The ``positions``
    mapping must contain exactly one entry per :class:`FilterPosition`, keyed by
    the lowercased enum name (``upper_limit``, ``pass_through``, ``pos_1`` ..
    ``pos_4``, ``lower_limit``). Each value is either a number (the motor
    position) or a mapping with ``position`` and optional ``description`` keys.

    Returns
    -------
    list[Filter]
        One configured Filter device per top-level entry, named after its key.
    """
    config = _filters_config()

    filters: list[Filter] = []
    for name, spec in config["filters"].items():
        positions_spec = spec["positions"]
        if len(positions_spec) != len(FilterPosition):
            raise ValueError(
                f"Filter {name!r} must define exactly {len(FilterPosition)} "
                f"positions (one per FilterPosition), got {len(positions_spec)}."
            )
        positions: dict[FilterPosition, FilterSetting] = {}
        for key, value in positions_spec.items():
            filter_position = FilterPosition[key.upper()]
            if isinstance(value, dict):
                setting = FilterSetting(value["position"], value.get("description"))
            else:
                setting = FilterSetting(value)
            positions[filter_position] = setting
        filters.append(
            Filter(
                motor_pv=spec["motor_pv"],
                positions=positions,
                in_pos_switch_pv=spec["in_position_switch"],
                name=name,
            )
        )
    return filters


SET_FILTERS_NAMES = ("filter1_upstream", "filter1_downstream", "filter2")


def _setting_positions(filter_name: str) -> dict[str, FilterPosition]:
    """Map a filter's choices (pass through and its materials) to positions."""
    positions = _filters_config()["filters"][filter_name]["positions"]
    choices = {"pass through": FilterPosition.PASS_THROUGH}
    for position in (
        FilterPosition.POS_1,
        FilterPosition.POS_2,
        FilterPosition.POS_3,
        FilterPosition.POS_4,
    ):
        value = positions[position.name.lower()]
        # A slot with no description holds no filter, so it is not offered
        if isinstance(value, dict) and value.get("description"):
            choices[value["description"]] = position
    return choices


SETTING_POSITIONS: dict[str, dict[str, FilterPosition]] = {
    name: _setting_positions(name) for name in SET_FILTERS_NAMES
}

if TYPE_CHECKING:
    Filter1UpstreamSetting = Filter1DownstreamSetting = Filter2Setting = str
else:
    # Enums built from filters.yml so the GUI offers each filter's own materials
    Filter1UpstreamSetting, Filter1DownstreamSetting, Filter2Setting = (
        StrEnum(
            name.title().replace("_", "") + "Setting",
            {label: label for label in SETTING_POSITIONS[name]},
        )
        for name in SET_FILTERS_NAMES
    )


def set_filters(
    filter1_upstream_setting: Filter1UpstreamSetting | None = None,
    filter1_downstream_setting: Filter1DownstreamSetting | None = None,
    filter2_setting: Filter2Setting | None = None,
    filter1_upstream: Filter | None = None,
    filter1_downstream: Filter | None = None,
    filter2: Filter | None = None,
    fe_shutter: PPSShutter | None = None,
):
    """Set filter 1 upstream, filter 1 downstream and filter 2 by material.

    Each setting is one of that filter's materials from ``filters.yml`` (for
    example ``"6 mm Cu"``) or ``"pass through"``. A filter with no setting does
    not move; the others move together.

    Moving a filter closes the front-end shutter through its interlock. Once the
    filters are in position the interlock is acknowledged, and the front-end
    shutter is reopened if it was open before.

    Parameters
    ----------
    filter1_upstream_setting : str, optional
        Material for filter 1 upstream.
    filter1_downstream_setting : str, optional
        Material for filter 1 downstream.
    filter2_setting : str, optional
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
    requests = (
        ("filter1_upstream", filter1_upstream_setting, filter1_upstream),
        ("filter1_downstream", filter1_downstream_setting, filter1_downstream),
        ("filter2", filter2_setting, filter2),
    )
    args = []
    for name, setting, device in requests:
        if setting is None:
            continue
        choices = SETTING_POSITIONS[name]
        if str(setting) not in choices:
            raise ValueError(
                f"{setting!r} is not a setting of {name}; choose one of {list(choices)}"
            )
        args += [ensure_available(Filter, **{name: device}), choices[str(setting)]]
    if not args:
        raise ValueError("Choose a setting for at least one filter.")
    fe_shutter = ensure_available(PPSShutter, fe_shutter=fe_shutter)
    fe_was_open = (yield from bps.rd(fe_shutter.status)) == ShutterStatus.OPEN
    yield from bps.mv(*args)
    yield from bps.mv(fe_shutter.interlock, True)
    if fe_was_open:
        yield from bps.mv(fe_shutter, True)
