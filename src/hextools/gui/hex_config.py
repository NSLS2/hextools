"""HEX-specific policy adapters for generic GUI extension points."""

import asyncio
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

from hextools.gui.device_status import QtDeviceStatus, QtStatusGroup
from hextools.machine import NSLS2StorageRing
from hextools.motors import (
    CollimatorTable,
    EDXDTable,
    OpticsTable,
    SampleSmall,
    SampleTower,
)
from hextools.photon_delivery_system.dclm import DCLM
from hextools.photon_delivery_system.filters import (
    Filter,
    Filter1DownstreamPosition,
    Filter1UpstreamPosition,
    Filter2Position,
    Filter3Position,
    Filter4Position,
)
from hextools.photon_delivery_system.shutter import (
    Shutter,
    ensure_shutter_closed,
    ensure_shutter_open,
)
from hextools.photon_delivery_system.slits import FrontEndSlit, Slits

# Upstream to downstream, with their section titles.
_TOWER_TITLES: dict[type, str] = {
    OpticsTable: "Optics Table",
    SampleTower: "Sample Tower",
    SampleSmall: "Sample Small",
    CollimatorTable: "Collimator Table",
    EDXDTable: "EDXD Table",
}

# Position enum of each filter, by device name.
FILTER_POSITION_ENUMS: dict[str, type] = {
    "filter1_upstream": Filter1UpstreamPosition,
    "filter1_downstream": Filter1DownstreamPosition,
    "filter2": Filter2Position,
    "filter3": Filter3Position,
    "filter4": Filter4Position,
}

# Filters offered in the filter controls, by device name.
CONTROLLED_FILTERS = ("filter1_upstream", "filter1_downstream", "filter2")


def _find(namespace: Mapping[str, Any], types: type | tuple[type, ...]):
    """``(variable name, object)`` for each distinct instance of ``types``."""
    seen: set[int] = set()
    found = []
    for name, obj in sorted(namespace.items()):
        if not name.startswith("_") and isinstance(obj, types) and id(obj) not in seen:
            seen.add(id(obj))
            found.append((name, obj))
    return found


def _slits_title(name: str) -> str:
    """``a_slits`` -> ``A Slits``, ``edxd_slits`` -> ``EDXD Slits``."""
    first, *rest = name.split("_")
    return " ".join([first.upper(), *(word.title() for word in rest)])


def slits_entries(namespace: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(variable name, display title)`` for each set of slits in ``namespace``."""
    return [(name, _slits_title(name)) for name, _ in _find(namespace, Slits)]


def _filter_label(name: str) -> str:
    """``filter1_upstream`` -> ``1 Upstream``, ``filter2`` -> ``2``."""
    return name.removeprefix("filter").replace("_", " ").strip().title() or name


def _filters(namespace: Mapping[str, Any]) -> list[tuple[str, Filter]]:
    """``(variable name, filter)`` for each filter, upstream before downstream."""
    return sorted(
        _find(namespace, Filter),
        key=lambda item: item[0].replace("upstream", "0").replace("downstream", "1"),
    )


def filter_rows(
    namespace: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, type]]:
    """Row label to path, and path to position enum, for each controlled filter.

    Only :data:`CONTROLLED_FILTERS` with a position enum are included.
    """
    rows: dict[str, str] = {}
    value_types: dict[str, type] = {}
    for path, device in _filters(namespace):
        enum = FILTER_POSITION_ENUMS.get(device.name)
        if enum is not None and device.name in CONTROLLED_FILTERS:
            rows[f"Filter {_filter_label(path)}"] = path
            value_types[path] = enum
    return rows, value_types


def status_sections(
    namespace: Mapping[str, Any], loop: asyncio.AbstractEventLoop
) -> list[QtStatusGroup]:
    """Status sections for the HEX storage ring, apertures, optics, and towers.

    Devices are found by type in ``namespace``; any that are missing are skipped.
    """
    sections: list[QtStatusGroup] = []
    for _, ring in _find(namespace, NSLS2StorageRing):
        sections.append(
            QtStatusGroup(
                "Storage Ring",
                {"Current": ring.beam_current, "Mode": ring.operating_mode},
                loop,
            )
        )
    fe_slits = _find(namespace, FrontEndSlit)
    for name, fe_slit in fe_slits:
        title = f"Front-End Slit ({name})" if len(fe_slits) > 1 else "Front-End Slit"
        sections.append(QtStatusGroup(title, {"Aperture": fe_slit.aperture}, loop))
    filters = _filters(namespace)
    if filters:
        sections.append(
            QtStatusGroup(
                "Filters",
                {_filter_label(name): f.description for name, f in filters},
                loop,
            )
        )
    for _, mono in _find(namespace, DCLM):
        sections.append(
            QtStatusGroup(
                "Monochromator",
                {"Beam Mode": mono.beam_mode, "Energy": mono.energy},
                loop,
            )
        )
    for name, slits in _find(namespace, Slits):
        sections.append(
            QtStatusGroup(
                _slits_title(name),
                {
                    "H Gap": slits.horizontal_gap,
                    "H Center": slits.horizontal_center,
                    "V Gap": slits.vertical_gap,
                    "V Center": slits.vertical_center,
                },
                loop,
            )
        )
    for tower_type, title in _TOWER_TITLES.items():
        towers = _find(namespace, tower_type)
        for name, tower in towers:
            display_name = f"{title} ({name})" if len(towers) > 1 else title
            sections.append(QtDeviceStatus(tower, loop, display_name=display_name))
    return sections


def shutter_controls(
    namespace: MutableMapping[str, Any],
) -> tuple[dict[str, Shutter], Callable[[str, bool], str]]:
    """Select HEX shutters and supply the manual, zero-settling command policy."""
    shutters = {}
    seen = set()
    for name, obj in sorted(namespace.items()):
        if (
            not name.startswith("_")
            and isinstance(obj, Shutter)
            and id(obj) not in seen
        ):
            shutters[name] = obj
            seen.add(id(obj))

    def command(name: str, open_shutter: bool) -> str:
        plan = ensure_shutter_open if open_shutter else ensure_shutter_closed
        namespace.setdefault(plan.__name__, plan)
        return f"RE({plan.__name__}({name}, allow_actuation=True, sleep_time=0.0))"

    return shutters, command
