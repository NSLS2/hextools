"""HEX-specific policy adapters for generic GUI extension points."""

from collections.abc import Callable
from typing import Any

from hextools.catalog.generate import type_expr
from hextools.catalog.model import DeviceRegistry, TypeExpr
from hextools.gui.run_engine_clients import RunEngineClient
from hextools.gui.signal_status_group import QtDeviceStatus, QtStatusGroup
from hextools.photon_delivery_system.filters import (
    Filter1DownstreamPosition,
    Filter1UpstreamPosition,
    Filter2Position,
    Filter3Position,
    Filter4Position,
)

# Qualified names of the device classes, as in the device registry.
_STORAGE_RING = "hextools.machine.NSLS2StorageRing"
_DCLM = "hextools.photon_delivery_system.dclm.DCLM"
_FILTER = "hextools.photon_delivery_system.filters.Filter"
_SHUTTER = "hextools.photon_delivery_system.shutter.Shutter"
_FRONT_END_SLIT = "hextools.photon_delivery_system.slits.FrontEndSlit"
_SLITS = "hextools.photon_delivery_system.slits.Slits"

# Upstream to downstream, with their section titles.
_TOWER_TITLES: dict[str, str] = {
    "hextools.motors.OpticsTable": "Optics Table",
    "hextools.motors.SampleTower": "Sample Tower",
    "hextools.motors.SampleSmall": "Sample Small",
    "hextools.motors.CollimatorTable": "Collimator Table",
    "hextools.motors.EDXDTable": "EDXD Table",
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


def _find(registry: DeviceRegistry, class_name: str) -> list[str]:
    """Sorted names of the top-level devices of qualified class ``class_name``."""
    return sorted(key for key, node in registry.items() if class_name in node.classes)


def _slits_title(name: str) -> str:
    """``a_slits`` -> ``A Slits``, ``edxd_slits`` -> ``EDXD Slits``."""
    first, *rest = name.split("_")
    return " ".join([first.upper(), *(word.title() for word in rest)])


def slits_entries(registry: DeviceRegistry) -> list[tuple[str, str]]:
    """``(device name, display title)`` for each set of slits in ``registry``."""
    return [(name, _slits_title(name)) for name in _find(registry, _SLITS)]


def _filter_label(name: str) -> str:
    """``filter1_upstream`` -> ``1 Upstream``, ``filter2`` -> ``2``."""
    return name.removeprefix("filter").replace("_", " ").strip().title() or name


def _filters(registry: DeviceRegistry) -> list[str]:
    """Name of each filter, upstream before downstream."""
    return sorted(
        _find(registry, _FILTER),
        key=lambda name: name.replace("upstream", "0").replace("downstream", "1"),
    )


def filter_rows(
    registry: DeviceRegistry,
) -> tuple[dict[str, str], dict[str, TypeExpr]]:
    """Row label to path, and path to position type, for each controlled filter.

    Only :data:`CONTROLLED_FILTERS` with a position enum are included.
    """
    rows: dict[str, str] = {}
    value_types: dict[str, TypeExpr] = {}
    for path in _filters(registry):
        name = registry[path].name or path
        enum = FILTER_POSITION_ENUMS.get(name)
        if enum is not None and name in CONTROLLED_FILTERS:
            rows[f"Filter {_filter_label(path)}"] = path
            value_types[path] = type_expr(enum)
    return rows, value_types


def status_sections(re_client: RunEngineClient) -> list[QtStatusGroup]:
    """Status sections for the HEX storage ring, apertures, optics, and towers.

    Devices are found by class in the client's device registry and read through
    the client; any that are missing are skipped.
    """
    registry = re_client.device_registry
    sections: list[QtStatusGroup] = []
    for name in _find(registry, _STORAGE_RING):
        sections.append(
            QtStatusGroup(
                "Storage Ring",
                {"Current": f"{name}.beam_current", "Mode": f"{name}.operating_mode"},
                re_client,
            )
        )
    fe_slits = _find(registry, _FRONT_END_SLIT)
    for name in fe_slits:
        title = f"Front-End Slit ({name})" if len(fe_slits) > 1 else "Front-End Slit"
        sections.append(
            QtStatusGroup(title, {"Aperture": f"{name}.aperture"}, re_client)
        )
    filters = _filters(registry)
    if filters:
        sections.append(
            QtStatusGroup(
                "Filters",
                {_filter_label(name): f"{name}.description" for name in filters},
                re_client,
            )
        )
    for name in _find(registry, _DCLM):
        sections.append(
            QtStatusGroup(
                "Monochromator",
                {"Beam Mode": f"{name}.beam_mode", "Energy": f"{name}.energy"},
                re_client,
            )
        )
    for name in _find(registry, _SLITS):
        sections.append(
            QtStatusGroup(
                _slits_title(name),
                {
                    "H Gap": f"{name}.horizontal_gap",
                    "H Center": f"{name}.horizontal_center",
                    "V Gap": f"{name}.vertical_gap",
                    "V Center": f"{name}.vertical_center",
                },
                re_client,
            )
        )
    for tower_class, title in _TOWER_TITLES.items():
        towers = _find(registry, tower_class)
        for name in towers:
            display_name = f"{title} ({name})" if len(towers) > 1 else title
            sections.append(QtDeviceStatus(name, re_client, display_name=display_name))
    return sections


def shutter_controls(
    registry: DeviceRegistry,
) -> tuple[list[str], Callable[[str, bool], tuple[str, dict[str, Any]]]]:
    """Select HEX shutters and supply the manual, zero-settling plan policy."""

    def shutter_plan(name: str, open_shutter: bool) -> tuple[str, dict[str, Any]]:
        plan = "ensure_shutter_open" if open_shutter else "ensure_shutter_closed"
        return plan, {"allow_actuation": True, "sleep_time": 0.0}

    return _find(registry, _SHUTTER), shutter_plan
