import enum
from collections.abc import Sequence
from typing import Any

import bluesky.plan_stubs as bps
import pytest
from bluesky.protocols import Movable
from ophyd_async.core import Device, DeviceVector, init_devices, soft_signal_rw
from ophyd_async.sim import SimMotor

from hextools.catalog import (
    NO_DEFAULT,
    DeviceRegistry,
    PlanCatalog,
    load_catalog,
    write_catalog,
)


class Mode(enum.Enum):
    FAST = "Fast"
    SLOW = "Slow"


def scan(
    points: Sequence[Movable],
    mode: Mode = Mode.FAST,
    stage: SimMotor | None = None,
):
    """Scan some points."""
    yield from bps.null()


def opaque(thing: Any):
    """Take something no form can build."""
    yield from bps.null()


class _Axes(Device):
    def __init__(self, name=""):
        self.axis = DeviceVector({1: SimMotor()})
        super().__init__(name=name)


@pytest.fixture
def namespace(RE):
    with init_devices(mock=True):
        motor = SimMotor(name="motor")
        axes = _Axes(name="axes")
    return {
        "motor": motor,
        "m": motor,
        "axes": axes,
        "temperature": soft_signal_rw(float, 1.5, units="degC", name="temperature"),
        "scan": scan,
        "opaque": opaque,
    }


@pytest.fixture
def registry(namespace):
    return DeviceRegistry.from_namespace(namespace, revision=4)


@pytest.fixture
def catalog(namespace):
    return PlanCatalog.from_namespace(namespace, revision=2)


def test_documents_round_trip(catalog, registry):
    assert PlanCatalog.from_dict(catalog.to_dict()) == catalog
    assert DeviceRegistry.from_dict(registry.to_dict()) == registry


def test_registry_resolves_paths_aliases_and_indexes(registry):
    assert registry.canonical("m") == "motor"
    assert registry.node("m.velocity") is registry["motor"].children["velocity"]
    assert registry.node("axes.axis[1]").name == "axes-axis-1"
    for bad in ("nope", "motor.nope", "axes.axis.1", "axes.axis[2]", "motor;x"):
        with pytest.raises(KeyError):
            registry.node(bad)


def test_registry_finds_devices_by_type_and_text(registry):
    movable = {"kind": "device", "protocols": ["Movable"]}
    assert registry.find(movable) == ["axes.axis[1]", "motor"]
    assert "temperature" in registry.find({**movable, "signals": True})
    sequence = {"kind": "sequence", "items": movable}
    assert registry.choices({"kind": "optional", "of": sequence}) == [
        "axes.axis[1]",
        "motor",
    ]
    assert registry.search("AXIS") == ["axes.axis", "axes.axis[1]"]
    assert "motor.velocity" in registry.search("velocity", signals=True)


def test_catalog_searches_and_finds_plans_for_a_device(catalog, registry):
    assert [p.name for p in catalog.search("points")] == ["scan"]
    assert [p.name for p in catalog.gui_plans()] == ["scan"]
    scan_plan = catalog["scan"]
    assert scan_plan.parameter("mode").default == Mode.FAST.value
    assert scan_plan.parameter("points").default is NO_DEFAULT
    assert [p.name for p in scan_plan.device_parameters()] == ["points", "stage"]
    taking = [
        (plan.name, p.name) for plan, p in catalog.plans_taking("motor", registry)
    ]
    assert taking == [("scan", "points"), ("scan", "stage")]


def test_registry_applies_deltas_on_its_revision_only(registry):
    delta = {
        "schema_version": 1,
        "base_revision": 4,
        "revision": 5,
        "removed": ["axes"],
        "added": {"slow": registry["motor"].to_dict()},
    }
    updated = registry.apply(delta)
    assert updated.revision == 5
    assert set(updated) == {"motor", "temperature", "slow"}
    assert set(registry) == {"axes", "motor", "temperature"}
    with pytest.raises(ValueError, match="revision"):
        updated.apply(delta)


def test_load_catalog_reads_written_files(namespace, tmp_path, catalog, registry):
    path = tmp_path / "catalog.yml"
    write_catalog(namespace, path, revision=2)
    loaded_catalog, loaded_registry = load_catalog(path)
    assert loaded_catalog == catalog
    assert set(loaded_registry) == set(registry)
