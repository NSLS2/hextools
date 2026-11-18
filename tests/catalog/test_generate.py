import dataclasses
import enum
import functools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import bluesky.plan_stubs as bps
import pytest
import yaml
from annotated_types import Ge, Gt
from bluesky.protocols import Movable, Readable
from ophyd_async.core import Device, DeviceVector, init_devices, soft_signal_rw
from ophyd_async.sim import SimMotor

from hextools.catalog import (
    Units,
    device_registry,
    plan_catalog,
    to_value,
    type_expr,
    write_catalog,
)
from hextools.photon_delivery_system.slits import Slits

jsonschema = pytest.importorskip("jsonschema")

_SCHEMA = json.loads(
    (
        Path(__file__).parents[2]
        / "docs/_static/schemas/plan_device_catalog.schema.json"
    ).read_text()
)


def _assert_valid(document: dict[str, Any]):
    validator = jsonschema.Draft202012Validator(_SCHEMA)
    errors = [
        f"{list(e.absolute_path)}: {e.message}" for e in validator.iter_errors(document)
    ]
    assert not errors, errors[:5]


class Mode(enum.Enum):
    FAST = "Fast"
    SLOW = "Slow"


@dataclasses.dataclass(frozen=True)
class ScanPoint:
    """Where to measure, and with what.

    Attributes
    ----------
    stage : Movable
        The stage to move.
    """

    stage: Movable
    detectors: Sequence[Readable]
    exposure: Annotated[float, Units("s"), Gt(0)] = 0.1


def scan(
    points: list[ScanPoint],
    mode: Mode = Mode.FAST,
    step: Annotated[float, Units("mm"), Ge(0)] = 1.0,
    repeats: int = 1,
    note: str | None = None,
    stage: SimMotor | Slits | None = None,
):
    """Scan some points.

    Parameters
    ----------
    points : list[ScanPoint]
        The points, in order.
    """
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
        slits = Slits("SLT:", name="a_slits")
        axes = _Axes(name="axes")
    return {
        "motor": motor,
        "m": motor,
        "a_slits": slits,
        "axes": axes,
        "temperature": soft_signal_rw(float, 1.5, units="degC", name="temperature"),
        "scan": scan,
        "opaque": opaque,
        "not_a_plan": print,
        "_hidden": scan,
    }


def test_plan_catalog_describes_types_defaults_and_records(namespace):
    catalog = plan_catalog(namespace, revision=3)
    _assert_valid(catalog)
    assert list(catalog["plans"]) == ["opaque", "scan"]
    plan = catalog["plans"]["scan"]
    assert plan["description"] == "Scan some points."
    params = {p["name"]: p for p in plan["parameters"]}
    points = params["points"]
    assert points["required"] is True
    assert points["description"] == "The points, in order."
    record = points["type"]["items"]
    assert points["type"]["kind"] == "sequence"
    assert record["kind"] == "record"
    assert record["name"].endswith("test_generate.ScanPoint")
    fields = {f["name"]: f for f in record["fields"]}
    assert fields["stage"]["type"] == {"kind": "device", "protocols": ["Movable"]}
    assert fields["stage"]["description"] == "The stage to move."
    assert fields["detectors"]["type"]["items"] == {
        "kind": "device",
        "protocols": ["Readable"],
    }
    assert fields["exposure"] == {
        "name": "exposure",
        "type": {"kind": "number", "units": "s", "exclusive_minimum": 0},
        "required": False,
        "default": 0.1,
    }
    assert params["mode"]["type"]["choices"] == [{"value": "Fast"}, {"value": "Slow"}]
    assert params["mode"]["default"] == "Fast"
    assert params["step"]["type"] == {"kind": "number", "units": "mm", "minimum": 0}
    assert params["note"]["type"] == {"kind": "optional", "of": {"kind": "string"}}
    stage = params["stage"]["type"]
    assert stage["kind"] == "optional"
    assert stage["of"]["kind"] == "device"
    assert [c.rsplit(".", 1)[1] for c in stage["of"]["classes"]] == [
        "SimMotor",
        "Slits",
    ]
    assert "gui_supported" not in plan
    assert catalog["plans"]["opaque"]["gui_supported"] is False


def test_device_registry_describes_the_tree(namespace):
    registry = device_registry(namespace, revision=7)
    _assert_valid(registry)
    devices = registry["devices"]
    assert list(devices) == ["a_slits", "axes", "motor", "temperature"]
    motor = devices["motor"]
    assert motor["aliases"] == ["m"]
    assert motor["connected"] is True
    assert motor["classes"][0].endswith(".SimMotor")
    assert {"Movable", "Readable", "Locatable", "Stoppable"} <= set(motor["protocols"])
    assert motor["set_type"] == {"kind": "number"}
    readback = motor["children"]["user_readback"]
    assert readback["signal"] is True
    assert readback["access"] == "r"
    assert readback["read_format"] == "hinted"
    children = motor["children"]
    assert children["velocity"]["read_format"] == "config"
    assert children["user_setpoint"]["read_format"] == "none"
    assert readback["datakey"]["dtype"] == "number"
    assert devices["axes"]["children"]["axis"]["index"] == "int"
    assert "1" in devices["axes"]["children"]["axis"]["children"]
    temperature = devices["temperature"]
    assert temperature["access"] == "rw"
    assert temperature["datakey"]["units"] == "degC"


def test_partial_plans_are_described_by_their_function_without_bound_args(namespace):
    namespace["fast_scan"] = functools.partial(scan, [], mode=Mode.SLOW)
    catalog = plan_catalog(namespace)
    _assert_valid(catalog)
    plan = catalog["plans"]["fast_scan"]
    assert plan["module"] == __name__
    assert plan["description"] == "Scan some points."
    names = [p["name"] for p in plan["parameters"]]
    assert names == ["step", "repeats", "note", "stage"]
    step = plan["parameters"][0]
    assert step["type"] == {"kind": "number", "units": "mm", "minimum": 0}


def test_values_tag_devices_and_records(namespace):
    paths = {id(namespace["motor"]): "motor"}
    point = ScanPoint(namespace["motor"], [namespace["motor"]])
    assert to_value(point, paths) == {
        "$record": f"{__name__}.ScanPoint",
        "stage": {"$device": "motor"},
        "detectors": [{"$device": "motor"}],
        "exposure": 0.1,
    }
    for bad in (object(), float("nan"), {"$x": 1}, namespace["a_slits"]):
        with pytest.raises(ValueError):
            to_value(bad, paths)


@pytest.mark.parametrize(
    "hint, expected",
    [
        (
            tuple[int, str],
            {"kind": "tuple", "items": [{"kind": "integer"}, {"kind": "string"}]},
        ),
        (tuple[int, ...], {"kind": "sequence", "items": {"kind": "integer"}}),
        (set[str], {"kind": "sequence", "items": {"kind": "string"}, "unique": True}),
        (dict[str, bool], {"kind": "mapping", "values": {"kind": "boolean"}}),
        (Annotated[int, Gt(0)], {"kind": "integer", "minimum": 1}),
        (
            int | str,
            {"kind": "union", "options": [{"kind": "integer"}, {"kind": "string"}]},
        ),
    ],
)
def test_type_expressions(hint, expected):
    assert type_expr(hint) == expected


def test_write_catalog_writes_both_documents(namespace, tmp_path):
    path = tmp_path / "catalog.yml"
    write_catalog(namespace, path, revision=1)
    catalog, registry = yaml.safe_load_all(path.read_text())
    _assert_valid(catalog)
    _assert_valid(registry)
    assert "scan" in catalog["plans"]
    assert "motor" in registry["devices"]
