"""Utility functions for dealing with beamline constants."""

from dataclasses import dataclass
from enum import StrEnum
from importlib import resources
from typing import cast

import yaml

HEX_BEAMLINE_CONSTANTS_RESOURCE: str = "bl_constants.yml"


def fetch_beamline_constants(key: str | None = None) -> dict:
    """Fetch the beamline constants associated with a given key.

    Parameters
    ----------
    key : str, optional
        The key for the specific beamline constants to fetch.
        If None, all constants are returned.

    Returns
    -------
    dict
        The beamline constants corresponding to the given key.
    """
    source = resources.files(__package__ or "hextools").joinpath(
        HEX_BEAMLINE_CONSTANTS_RESOURCE
    )
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if key is None:
        return data
    return data.get(key, {})


@dataclass
class Position:
    """Represents a motor position with an optional description."""

    position: float
    description: str | None = None


def build_enum_from_position_list(
    key: str,
    positions: dict[str, dict | float],
    always_included_keys: list[str] | None = None,
    include_keys_without_desc: bool = False,
) -> type[StrEnum]:
    """Build a StrEnum of the selectable positions in a beamline constants mapping.

    Parameters
    ----------
    key : str
        Name of the positions mapping; the enum is named ``<Key>Position``.
    positions : dict[str, dict | float]
        Position name to either a bare motor position or a ``Position`` mapping.
    always_included_keys : list[str], optional
        Position names to include even if they have no description.
    include_keys_without_desc : bool, optional
        Whether to include every position without a description.

    Returns
    -------
    type[StrEnum]
        Members are named after the uppercased position name, and valued by its
        description, or by the title-cased position name if it has none.
    """
    always_included_keys = always_included_keys or []
    members = {}
    for name, value in positions.items():
        position = Position(**value) if isinstance(value, dict) else Position(value)
        if (
            position.description
            or include_keys_without_desc
            or name in always_included_keys
        ):
            members[name.upper()] = (
                position.description or name.replace("_", " ").title()
            )
    return cast(
        type[StrEnum], StrEnum(key.title().replace("_", "") + "Position", members)
    )


def build_enums_from_position_lists(
    key: str,
    always_included_keys: list[str] | None = None,
    include_keys_without_desc: bool = False,
) -> tuple[type[StrEnum], ...]:
    """Build one enum per positions mapping nested under ``key``.

    See :func:`build_enum_from_position_list` for the parameters and members.
    """
    return tuple(
        build_enum_from_position_list(
            name, positions, always_included_keys, include_keys_without_desc
        )
        for name, positions in fetch_beamline_constants(key).items()
    )
