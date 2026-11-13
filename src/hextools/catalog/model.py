"""Typed views of the plan catalog and device registry documents, with lookups.

:class:`PlanCatalog` and :class:`DeviceRegistry` wrap the ``PlanCatalog`` and
``DeviceRegistry`` documents of ``docs/plan_device_schema.md``. They need no
devices or control-system packages, so a remote client can use them on documents
it receives; :meth:`PlanCatalog.from_namespace` and
:meth:`DeviceRegistry.from_namespace` build them from a local namespace.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

#: A type expression; see ``docs/plan_device_schema.md``.
TypeExpr = dict[str, Any]

SCHEMA_VERSION = 1

_TOP = re.compile(r"[A-Za-z_]\w*")
_STEP = re.compile(r"\.([A-Za-z_]\w*)|\[(\d+)\]|\['(\w+)'\]")


class _NoDefault:
    def __repr__(self) -> str:
        return "NO_DEFAULT"


#: The default of a parameter or field that has none.
NO_DEFAULT: Any = _NoDefault()


def _optional(source: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {k: source[k] for k in keys if k in source}


def _drop_none(entries: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in entries.items() if v is not None}


def device_types(expr: TypeExpr) -> list[TypeExpr]:
    """Return every ``device`` type in ``expr``, at any depth (including records)."""
    kind = expr.get("kind")
    if kind == "device":
        return [expr]
    found = []
    for key in ("of", "values"):
        if isinstance(expr.get(key), dict):
            found += device_types(expr[key])
    items = expr.get("items")
    for item in items if isinstance(items, list) else [items] if items else []:
        found += device_types(item)
    for option in expr.get("options", []):
        found += device_types(option)
    for f in expr.get("fields", []):
        found += device_types(f["type"])
    return found


# -- Plans --------------------------------------------------------------------


@dataclass(frozen=True)
class Parameter:
    """A plan parameter."""

    name: str
    kind: str
    type: TypeExpr
    required: bool
    default: Any = NO_DEFAULT
    description: str | None = None
    ui: dict[str, Any] | None = None

    @property
    def has_default(self) -> bool:
        """Whether the catalog gives a default; ``None`` is a valid default."""
        return self.default is not NO_DEFAULT

    @classmethod
    def from_dict(cls, entry: Mapping[str, Any]) -> Parameter:
        """Return the parameter of a catalog ``Parameter`` entry."""
        return cls(
            entry["name"],
            entry["kind"],
            entry["type"],
            entry["required"],
            entry.get("default", NO_DEFAULT),
            entry.get("description"),
            entry.get("ui"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the catalog ``Parameter`` entry."""
        entry = {
            "name": self.name,
            "kind": self.kind,
            "type": self.type,
            "required": self.required,
        }
        if self.has_default:
            entry["default"] = self.default
        return entry | _drop_none({"description": self.description, "ui": self.ui})


@dataclass(frozen=True)
class PlanDescription:
    """A plan in a :class:`PlanCatalog`."""

    name: str
    parameters: tuple[Parameter, ...] = ()
    module: str | None = None
    description: str | None = None
    gui_supported: bool = True
    ui: dict[str, Any] | None = None

    def parameter(self, name: str) -> Parameter:
        """Return the parameter called ``name``.

        Raises
        ------
        KeyError
            If there is no such parameter.
        """
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(name)

    def device_parameters(self) -> list[Parameter]:
        """Return the parameters that take a device anywhere in their type."""
        return [p for p in self.parameters if device_types(p.type)]

    def matches(self, text: str) -> bool:
        """Return True if ``text`` is in the name, description or a parameter name.

        Case is ignored.
        """
        needle = text.lower()
        haystacks = [
            self.name,
            self.description or "",
            *(p.name for p in self.parameters),
        ]
        return any(needle in h.lower() for h in haystacks)

    @classmethod
    def from_dict(cls, entry: Mapping[str, Any]) -> PlanDescription:
        """Return the plan of a catalog ``Plan`` entry."""
        return cls(
            entry["name"],
            tuple(Parameter.from_dict(p) for p in entry["parameters"]),
            entry.get("module"),
            entry.get("description"),
            entry.get("gui_supported", True),
            entry.get("ui"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the catalog ``Plan`` entry."""
        entry: dict[str, Any] = {"name": self.name}
        entry |= _drop_none({"module": self.module, "description": self.description})
        entry["parameters"] = [p.to_dict() for p in self.parameters]
        if not self.gui_supported:
            entry["gui_supported"] = False
        return entry | _drop_none({"ui": self.ui})


@dataclass
class PlanCatalog(Mapping[str, PlanDescription]):
    """The plans that may be run, by name."""

    plans: dict[str, PlanDescription] = field(default_factory=dict)
    revision: int = 0

    def __getitem__(self, name: str) -> PlanDescription:
        """Return the plan called ``name``."""
        return self.plans[name]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the plan names."""
        return iter(self.plans)

    def __len__(self) -> int:
        """Return the number of plans."""
        return len(self.plans)

    def search(self, text: str) -> list[PlanDescription]:
        """Return the plans with ``text`` in their name, description or parameters."""
        return [p for p in self.plans.values() if p.matches(text)]

    def gui_plans(self) -> list[PlanDescription]:
        """Return the plans a form can build every required parameter of."""
        return [p for p in self.plans.values() if p.gui_supported]

    def plans_taking(
        self, path: str, registry: DeviceRegistry
    ) -> list[tuple[PlanDescription, Parameter]]:
        """Return ``(plan, parameter)`` for each parameter the device at ``path`` fits.

        Raises
        ------
        KeyError
            If ``path`` is not in ``registry``.
        """
        node = registry.node(path)
        return [
            (plan, p)
            for plan in self.plans.values()
            for p in plan.parameters
            if any(node.satisfies(t) for t in device_types(p.type))
        ]

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> PlanCatalog:
        """Return the catalog of a ``PlanCatalog`` document."""
        _check_version(document)
        return cls(
            {n: PlanDescription.from_dict(p) for n, p in document["plans"].items()},
            document["revision"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the ``PlanCatalog`` document."""
        return {
            "schema_version": SCHEMA_VERSION,
            "revision": self.revision,
            "plans": {n: p.to_dict() for n, p in self.plans.items()},
        }

    @classmethod
    def from_namespace(
        cls, namespace: Mapping[str, Any], revision: int = 0
    ) -> PlanCatalog:
        """Describe the plans in ``namespace``.

        See :func:`~hextools.catalog.plan_catalog`.
        """
        from hextools.catalog.generate import plan_catalog

        return cls.from_dict(plan_catalog(namespace, revision))


# -- Devices ------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceNode:
    """A device or signal in a :class:`DeviceRegistry`."""

    classes: tuple[str, ...]
    protocols: frozenset[str]
    signal: bool = False
    name: str | None = None
    description: str | None = None
    connected: bool | None = None
    access: str | None = None
    read_format: str | None = None
    hinted_fields: tuple[str, ...] = ()
    set_type: TypeExpr | None = None
    datakey: dict[str, Any] | None = None
    index: str | None = None
    children: dict[str, DeviceNode] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    ui: dict[str, Any] | None = None

    def satisfies(self, expr: TypeExpr) -> bool:
        """Return True if this node can be passed as a ``device`` type ``expr``.

        It must have every protocol listed, be an instance of one of the classes
        listed (if any), and be a device rather than a signal unless ``expr``
        accepts signals. Any other kind of type is not satisfied.
        """
        if expr.get("kind") != "device":
            return False
        if self.signal and not expr.get("signals", False):
            return False
        if not set(expr.get("protocols", ())) <= self.protocols:
            return False
        classes = expr.get("classes")
        return not classes or any(c in self.classes for c in classes)

    def child_path(self, path: str, key: str) -> str:
        """Return the path of child ``key`` of this node, at ``path``."""
        if self.index == "int":
            return f"{path}[{key}]"
        if self.index == "str":
            return f"{path}['{key}']"
        return f"{path}.{key}"

    @classmethod
    def from_dict(cls, entry: Mapping[str, Any]) -> DeviceNode:
        """Return the node of a registry ``DeviceNode`` entry."""
        return cls(
            tuple(entry["classes"]),
            frozenset(entry["protocols"]),
            entry.get("signal", False),
            entry.get("name"),
            entry.get("description"),
            entry.get("connected"),
            entry.get("access"),
            entry.get("read_format"),
            tuple(entry.get("hinted_fields", ())),
            entry.get("set_type"),
            entry.get("datakey"),
            entry.get("index"),
            {k: cls.from_dict(c) for k, c in entry.get("children", {}).items()},
            tuple(entry.get("aliases", ())),
            entry.get("ui"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the registry ``DeviceNode`` entry."""
        entry: dict[str, Any] = {
            "classes": list(self.classes),
            "protocols": sorted(self.protocols),
        }
        if self.signal:
            entry["signal"] = True
        entry |= _drop_none(
            {
                "name": self.name,
                "description": self.description,
                "connected": self.connected,
                "access": self.access,
                "read_format": self.read_format,
                "hinted_fields": list(self.hinted_fields) or None,
                "set_type": self.set_type,
                "datakey": self.datakey,
                "index": self.index,
            }
        )
        if self.children:
            entry["children"] = {k: c.to_dict() for k, c in self.children.items()}
        if self.aliases:
            entry["aliases"] = list(self.aliases)
        return entry | _drop_none({"ui": self.ui})


@dataclass
class DeviceRegistry(Mapping[str, DeviceNode]):
    """The devices that exist, by top-level name, and what they can do."""

    devices: dict[str, DeviceNode] = field(default_factory=dict)
    revision: int = 0

    def __getitem__(self, name: str) -> DeviceNode:
        """Return the top-level device called ``name`` (not an alias)."""
        return self.devices[name]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the top-level device names."""
        return iter(self.devices)

    def __len__(self) -> int:
        """Return the number of top-level devices."""
        return len(self.devices)

    def canonical(self, name: str) -> str:
        """Return the registry key of top-level ``name``, which may be an alias.

        Raises
        ------
        KeyError
            If there is no such device.
        """
        if name in self.devices:
            return name
        for key, node in self.devices.items():
            if name in node.aliases:
                return key
        raise KeyError(name)

    def node(self, path: str) -> DeviceNode:
        """Return the node at ``path``; its top-level name may be an alias.

        Raises
        ------
        KeyError
            If ``path`` is malformed or names no node.
        """
        top = _TOP.match(path)
        if top is None:
            raise KeyError(path)
        node = self.devices[self.canonical(top.group())]
        position = top.end()
        while position < len(path):
            step = _STEP.match(path, position)
            if step is None:
                raise KeyError(path)
            attribute, index, key = step.groups()
            name = attribute if attribute is not None else index or key
            # The step must use the syntax the node's children are addressed by.
            expected = {"int": index, "str": key}.get(node.index or "", attribute)
            if expected is None or name not in node.children:
                raise KeyError(path)
            node = node.children[name]
            position = step.end()
        return node

    def walk(self, *, signals: bool = True) -> Iterator[tuple[str, DeviceNode]]:
        """Yield ``(path, node)`` for every node, parents before their children."""

        def visit(path: str, node: DeviceNode) -> Iterator[tuple[str, DeviceNode]]:
            if signals or not node.signal:
                yield path, node
            for key, child in node.children.items():
                yield from visit(node.child_path(path, key), child)

        for name, node in self.devices.items():
            yield from visit(name, node)

    def find(self, expr: TypeExpr) -> list[str]:
        """Return the paths of the nodes that satisfy ``device`` type ``expr``."""
        return [path for path, node in self.walk() if node.satisfies(expr)]

    def choices(self, expr: TypeExpr) -> list[str]:
        """Return the paths a form could offer anywhere in type ``expr``."""
        paths = dict.fromkeys(p for t in device_types(expr) for p in self.find(t))
        return list(paths)

    def search(self, text: str, *, signals: bool = False) -> list[str]:
        """Return paths with ``text`` in the path, name or description; case ignored."""
        needle = text.lower()
        return [
            path
            for path, node in self.walk(signals=signals)
            if any(
                needle in h.lower()
                for h in (path, node.name or "", node.description or "")
            )
        ]

    def apply(self, delta: Mapping[str, Any]) -> DeviceRegistry:
        """Return this registry with a ``RegistryDelta`` applied.

        Raises
        ------
        ValueError
            If the delta is not based on this registry's revision; fetch the
            whole registry again instead.
        """
        _check_version(delta)
        if delta["base_revision"] != self.revision:
            raise ValueError(
                f"Delta is based on revision {delta['base_revision']}, "
                f"not {self.revision}"
            )
        devices = dict(self.devices)
        for name in delta.get("removed", ()):
            devices.pop(name, None)
        for section in ("added", "changed"):
            for name, entry in delta.get(section, {}).items():
                devices[name] = DeviceNode.from_dict(entry)
        return replace(self, devices=devices, revision=delta["revision"])

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> DeviceRegistry:
        """Return the registry of a ``DeviceRegistry`` document."""
        _check_version(document)
        return cls(
            {n: DeviceNode.from_dict(d) for n, d in document["devices"].items()},
            document["revision"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the ``DeviceRegistry`` document."""
        return {
            "schema_version": SCHEMA_VERSION,
            "revision": self.revision,
            "devices": {n: d.to_dict() for n, d in self.devices.items()},
        }

    @classmethod
    def from_namespace(
        cls, namespace: Mapping[str, Any], revision: int = 0, *, describe: bool = True
    ) -> DeviceRegistry:
        """Describe the devices in ``namespace``.

        See :func:`~hextools.catalog.device_registry`.
        """
        from hextools.catalog.generate import device_registry

        return cls.from_dict(device_registry(namespace, revision, describe=describe))


def _check_version(document: Mapping[str, Any]):
    version = document.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version {version!r}")


def load_catalog(path: str | Path) -> tuple[PlanCatalog, DeviceRegistry]:
    """Read a file written by :func:`~hextools.catalog.write_catalog`."""
    with Path(path).open() as file:
        plans, devices = yaml.safe_load_all(file)
    return PlanCatalog.from_dict(plans), DeviceRegistry.from_dict(devices)
