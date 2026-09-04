"""Filter device with multiple positions."""

from dataclasses import dataclass
from enum import IntEnum
from importlib import resources

import yaml
from ophyd_async.core import (
    AsyncMovable,
    AsyncStatus,
    StandardReadable,
    derived_signal_r,
    wait_for_value,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice, epics_signal_r
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor


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

    @AsyncStatus.wrap
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

        await self.filter_motor.set(self.positions[value].position)
        await wait_for_value(self.in_position, True, timeout=10)


FILTERS_CONFIG_RESOURCE: str = "filters.yml"


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
    source = resources.files(__package__ or "hextools").joinpath(
        FILTERS_CONFIG_RESOURCE
    )
    config = yaml.safe_load(source.read_text(encoding="utf-8"))

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
