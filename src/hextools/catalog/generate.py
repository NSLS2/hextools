"""Describe a namespace's plans and devices in the plan and device catalog format.

:func:`plan_catalog` and :func:`device_registry` build the ``PlanCatalog`` and
``DeviceRegistry`` documents of ``docs/_static/schemas/plan_device_catalog.schema.json``
from the functions and device objects in a namespace, typically the IPython
``user_ns``; :func:`write_catalog` writes both to a YAML file. See
``docs/plan_device_schema.md`` for the format.
"""

from __future__ import annotations

import asyncio
import collections.abc
import dataclasses
import enum
import functools
import inspect
import math
import re
import types
import typing
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import bluesky.protocols
import docstring_parser
import numpy as np
import yaml
from bluesky.protocols import Movable
from bluesky.run_engine import (
    call_in_bluesky_event_loop,
    get_bluesky_event_loop,
    in_bluesky_event_loop,
)
from ophyd_async.core import (
    DeviceMap,
    DeviceVector,
    Signal,
    SignalR,
    SignalRW,
    SignalW,
    SignalX,
)

from hextools.catalog.model import SCHEMA_VERSION
from hextools.catalog.namespace import SIGNAL_TYPES as _SIGNAL_TYPES
from hextools.catalog.namespace import is_connected, is_device_type
from hextools.catalog.namespace import iter_children as _iter_children
from hextools.catalog.namespace import looks_like_device as _looks_like_device
from hextools.catalog.namespace import walk_namespace as _walk_namespace

_QUALIFIED_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_PROTOCOLS = sorted(
    (name, proto)
    for name, proto in vars(bluesky.protocols).items()
    if isinstance(proto, type) and getattr(proto, "_is_runtime_protocol", False)
)
_SEQUENCE_ORIGINS = (
    list,
    set,
    frozenset,
    collections.abc.Sequence,
    collections.abc.MutableSequence,
    collections.abc.Iterable,
    collections.abc.Collection,
    collections.abc.Set,
    collections.abc.MutableSet,
)
_MAPPING_ORIGINS = (dict, collections.abc.Mapping, collections.abc.MutableMapping)


@dataclasses.dataclass(frozen=True)
class Units:
    """``Annotated`` metadata giving the units of a number, e.g. ``Units("mm")``."""

    units: str


def set_type(obj: Movable[Any]) -> type | None:
    """Return the value type ``obj.set`` takes, or None if it can't be determined."""
    if isinstance(obj, Signal) and obj.datatype is not None:
        return obj.datatype
    set_method = inspect.unwrap(type(obj).set)
    try:
        hints = typing.get_type_hints(set_method)
        params = [p for p in inspect.signature(set_method).parameters if p != "self"]
    except Exception:
        hints, params = {}, []
    hint = hints.get(params[0]) if params else None
    if isinstance(hint, type):
        return hint
    # Generic movables, e.g. StandardMovable[float], declare it on a base class.
    for cls in type(obj).__mro__:
        for base in getattr(cls, "__orig_bases__", ()):
            origin, args = typing.get_origin(base), typing.get_args(base)
            if hasattr(origin, "set") and args and isinstance(args[0], type):
                return args[0]
    return None


def qualified_name(obj: Any) -> str | None:
    """Return ``module.QualName`` of a class or function, or None if not importable."""
    name = f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', '')}"
    return name if _QUALIFIED_NAME.fullmatch(name) else None


# -- Type expressions ---------------------------------------------------------


def type_expr(hint: Any) -> dict[str, Any]:
    """Return the type expression for a Python type hint."""
    return _type_expr(hint, set())


def _type_expr(hint: Any, records: set[type]) -> dict[str, Any]:
    if typing.get_origin(hint) is typing.Annotated:
        base, *metadata = typing.get_args(hint)
        return _constrain(_type_expr(base, records), metadata)
    if hint is None or hint is type(None):
        return {"kind": "literal", "values": [None]}
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is typing.Literal:
        values = [v.value if isinstance(v, enum.Enum) else v for v in args]
        if all(v is None or isinstance(v, str | int | float | bool) for v in values):
            return {"kind": "literal", "values": values}
        return _any(hint)
    if origin in (typing.Union, types.UnionType):
        return _union(list(args), records)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return {"kind": "sequence", "items": _type_expr(args[0], records)}
        return {"kind": "tuple", "items": [_type_expr(a, records) for a in args]}
    if origin in _SEQUENCE_ORIGINS:
        expr: dict[str, Any] = {
            "kind": "sequence",
            "items": _type_expr(args[0], records) if args else _any(None),
        }
        if origin in (set, frozenset, collections.abc.Set, collections.abc.MutableSet):
            expr["unique"] = True
        return expr
    if origin in _MAPPING_ORIGINS:
        if len(args) == 2 and args[0] is str:
            return {"kind": "mapping", "values": _type_expr(args[1], records)}
        return _any(hint)
    if origin is not None and is_device_type(origin):
        # Generic protocols, e.g. Movable[float].
        hint = origin
    if hint is bool:
        return {"kind": "boolean"}
    if hint is int:
        return {"kind": "integer"}
    if hint is float:
        return {"kind": "number"}
    if hint is str:
        return {"kind": "string"}
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return _enum(hint)
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        return _record(hint, records)
    if is_device_type(hint):
        return _device_type(hint)
    return _any(hint)


def _any(hint: Any) -> dict[str, Any]:
    expr: dict[str, Any] = {"kind": "any"}
    if hint is not None:
        expr["python"] = getattr(hint, "__name__", None) or repr(hint)
    return expr


def _enum(cls: type[enum.Enum]) -> dict[str, Any]:
    choices = []
    for member in cls:
        value = member.value
        if not isinstance(value, str | int | float | bool):
            value = member.name
        choice: dict[str, Any] = {"value": value}
        if str(value) != member.name and not isinstance(member.value, str):
            choice["label"] = member.name
        choices.append(choice)
    return {
        "kind": "enum",
        "name": qualified_name(cls) or cls.__name__,
        "choices": choices,
    }


def _device_type(cls: type) -> dict[str, Any]:
    if getattr(cls, "_is_protocol", False):
        return {"kind": "device", "protocols": [cls.__name__]}
    expr: dict[str, Any] = {
        "kind": "device",
        "classes": [qualified_name(cls) or cls.__name__],
    }
    if issubclass(cls, _SIGNAL_TYPES):
        expr["signals"] = True
    return expr


def _union(options: list[Any], records: set[type]) -> dict[str, Any]:
    has_none = any(o is None or o is type(None) for o in options)
    exprs = [_type_expr(o, records) for o in options if o not in (None, type(None))]
    # A union of device classes is one device type that accepts any of them.
    if len(exprs) > 1 and all(
        e["kind"] == "device" and "classes" in e and "protocols" not in e for e in exprs
    ):
        merged: dict[str, Any] = {
            "kind": "device",
            "classes": list(dict.fromkeys(c for e in exprs for c in e["classes"])),
        }
        if any(e.get("signals") for e in exprs):
            merged["signals"] = True
        exprs = [merged]
    inner = exprs[0] if len(exprs) == 1 else {"kind": "union", "options": exprs}
    return {"kind": "optional", "of": inner} if has_none else inner


_CONSTRAINT_KEYS = {
    "number": {
        "ge": "minimum",
        "le": "maximum",
        "gt": "exclusive_minimum",
        "lt": "exclusive_maximum",
        "multiple_of": "step",
    },
    "integer": {"ge": "minimum", "le": "maximum", "multiple_of": "step"},
    "string": {"min_length": "min_length", "max_length": "max_length"},
    "sequence": {"min_length": "min_items", "max_length": "max_items"},
}


def _constrain(expr: dict[str, Any], metadata: list[Any]) -> dict[str, Any]:
    """Apply ``Annotated`` metadata (``Units`` and annotated-types constraints)."""
    expr = dict(expr)
    keys = _CONSTRAINT_KEYS.get(expr["kind"], {})
    for item in metadata:
        if isinstance(item, Units) and expr["kind"] in ("number", "integer"):
            expr["units"] = item.units
        for attr, key in keys.items():
            value = getattr(item, attr, None)
            if isinstance(value, int | float) and not isinstance(value, bool):
                expr[key] = value
        if expr["kind"] == "integer":
            # The schema's integer type has inclusive bounds only.
            if isinstance(gt := getattr(item, "gt", None), int):
                expr["minimum"] = gt + 1
            if isinstance(lt := getattr(item, "lt", None), int):
                expr["maximum"] = lt - 1
    return expr


def _record(cls: type, records: set[type]) -> dict[str, Any]:
    name = qualified_name(cls) or cls.__name__
    if cls in records:
        # Recursive records can't be expanded; a form couldn't build them anyway.
        return {"kind": "any", "python": name}
    records = records | {cls}
    hints = _type_hints(cls)
    descriptions = _docstring_descriptions(cls, "attribute")
    fields = []
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        entry: dict[str, Any] = {
            "name": f.name,
            "type": _type_expr(hints.get(f.name, f.type), records),
            "required": f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING,
        }
        default = _field_default(f)
        if default is not dataclasses.MISSING:
            entry["default"] = default
        description = f.metadata.get("description") or descriptions.get(f.name)
        if description:
            entry["description"] = description
        fields.append(entry)
    expr: dict[str, Any] = {"kind": "record", "name": name, "fields": fields}
    summary = _summary(cls.__doc__ if "__doc__" in vars(cls) else None)
    if summary:
        expr["description"] = summary
    return expr


def _field_default(f: dataclasses.Field) -> Any:
    try:
        if f.default is not dataclasses.MISSING:
            return to_value(f.default)
        if f.default_factory is not dataclasses.MISSING:
            return to_value(f.default_factory())
    except Exception:
        pass
    return dataclasses.MISSING


def _gui_buildable(expr: dict[str, Any]) -> bool:
    """Return True if a form could build a value of type ``expr``."""
    kind = expr["kind"]
    if kind == "any":
        return False
    if kind == "optional":
        return True
    if kind == "union":
        return any(_gui_buildable(o) for o in expr["options"])
    if kind == "sequence":
        return _gui_buildable(expr["items"])
    if kind == "tuple":
        return all(_gui_buildable(i) for i in expr["items"])
    if kind == "mapping":
        return _gui_buildable(expr["values"])
    if kind == "record":
        return all(_gui_buildable(f["type"]) for f in expr["fields"] if f["required"])
    return True


# -- Values -------------------------------------------------------------------


def to_value(obj: Any, device_paths: Mapping[int, str] | None = None) -> Any:
    """Return the catalog value of a Python value.

    Parameters
    ----------
    obj : Any
        The value.
    device_paths : Mapping[int, str], optional
        ``id(device)`` to its path, for device values.

    Raises
    ------
    ValueError
        If ``obj`` has no catalog value, e.g. an unknown device or an arbitrary
        object.
    """
    device_paths = device_paths or {}
    if isinstance(obj, enum.Enum):
        return to_value(obj.value, device_paths)
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, np.generic):
        return to_value(obj.item(), device_paths)
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"{obj!r} is not finite")
        return obj
    if id(obj) in device_paths:
        return {"$device": device_paths[id(obj)]}
    if _looks_like_device(obj):
        raise ValueError(f"{obj!r} is not in the device registry")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        name = qualified_name(type(obj))
        if name is None:
            raise ValueError(f"{type(obj).__name__} is not importable")
        value: dict[str, Any] = {"$record": name}
        for f in dataclasses.fields(obj):
            if f.init:
                value[f.name] = to_value(getattr(obj, f.name), device_paths)
        return value
    if isinstance(obj, list | tuple | set | frozenset):
        return [to_value(v, device_paths) for v in obj]
    if isinstance(obj, Mapping):
        if not all(isinstance(k, str) and not k.startswith("$") for k in obj):
            raise ValueError("mapping keys must be strings not starting '$'")
        return {k: to_value(v, device_paths) for k, v in obj.items()}
    raise ValueError(f"{type(obj).__name__} values have no catalog form")


# -- Plans --------------------------------------------------------------------


def is_plan(obj: Any) -> bool:
    """Return True if ``obj`` is a plan: a generator function, or a partial of one."""
    return callable(obj) and inspect.isgeneratorfunction(_unwrap_partial(obj)[0])


def _unwrap_partial(function: Any) -> tuple[Any, set[str]]:
    """Return the function under any partials and decorators, and the bound keywords.

    Keywords bound by a partial are fixed, so they are not plan parameters.
    """
    bound: set[str] = set()
    while True:
        if isinstance(function, functools.partial):
            bound |= function.keywords.keys()
            function = function.func
        elif hasattr(function, "__wrapped__"):
            function = inspect.unwrap(function)
        else:
            return function, bound


def describe_plan(
    name: str, function: Callable[..., Any], device_paths: Mapping[int, str]
) -> dict[str, Any]:
    """Return the ``Plan`` entry for ``function``, run as ``name``.

    Parameters
    ----------
    name : str
        The plan's name in the namespace.
    function : Callable
        The plan.
    device_paths : Mapping[int, str]
        ``id(device)`` to its path, for device defaults.
    """
    base, bound = _unwrap_partial(function)
    hints = _type_hints(base)
    descriptions = _docstring_descriptions(base, "param")
    parameters = []
    for p in inspect.signature(function).parameters.values():
        if p.name in bound:
            continue
        hint = hints.get(p.name, p.annotation)
        expr = _any(None) if hint is inspect.Parameter.empty else type_expr(hint)
        variadic = p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        entry: dict[str, Any] = {
            "name": p.name,
            "kind": p.kind.name.lower(),
            "type": expr,
            "required": p.default is p.empty and not variadic,
        }
        if p.default is not p.empty:
            try:
                entry["default"] = to_value(p.default, device_paths)
            except ValueError:
                pass
        if description := descriptions.get(p.name):
            entry["description"] = description
        parameters.append(entry)
    plan: dict[str, Any] = {"name": name}
    module = getattr(base, "__module__", None)
    if isinstance(module, str) and _QUALIFIED_NAME.fullmatch(module):
        plan["module"] = module
    if summary := _summary(inspect.getdoc(base)):
        plan["description"] = summary
    plan["parameters"] = parameters
    if not all(_gui_buildable(p["type"]) for p in parameters if p["required"]):
        plan["gui_supported"] = False
    return plan


def plan_catalog(namespace: Mapping[str, Any], revision: int = 0) -> dict[str, Any]:
    """Return the ``PlanCatalog`` of the public plans in ``namespace``.

    Parameters
    ----------
    namespace : Mapping[str, object]
        Where the plans and devices are, typically the IPython ``user_ns``.
    revision : int, optional
        The catalog's revision.
    """
    device_paths = {id(obj): path for path, obj in _walk_namespace(namespace).items()}
    plans = {
        name: describe_plan(name, obj, device_paths)
        for name, obj in sorted(namespace.items())
        if not name.startswith("_") and is_plan(obj)
    }
    return {"schema_version": SCHEMA_VERSION, "revision": revision, "plans": plans}


# -- Devices ------------------------------------------------------------------


def device_registry(
    namespace: Mapping[str, Any],
    revision: int = 0,
    *,
    describe: bool = True,
    timeout: float = 1.0,
) -> dict[str, Any]:
    """Return the ``DeviceRegistry`` of the devices in ``namespace``.

    Parameters
    ----------
    namespace : Mapping[str, object]
        Where the devices are, typically the IPython ``user_ns``.
    revision : int, optional
        The registry's revision.
    describe : bool, optional
        Add each signal's data key from ``describe()``, and how its parent reads
        it from the parent's ``describe()`` and ``describe_configuration()``.
        ophyd-async devices are described on the bluesky event loop, and only if
        it is running. EPICS signals are read from the IOC to describe them.
    timeout : float, optional
        Seconds to wait for each description.
    """
    tops: dict[int, tuple[str, Any, list[str]]] = {}
    candidates = sorted(
        (
            (name, obj)
            for name, obj in namespace.items()
            if not name.startswith("_") and _looks_like_device(obj)
        ),
        key=lambda kv: (kv[0] != getattr(kv[1], "name", None), kv[0]),
    )
    for name, obj in candidates:
        if id(obj) in tops:
            tops[id(obj)][2].append(name)
        else:
            tops[id(obj)] = (name, obj, [])

    signals: list[tuple[dict[str, Any], Any]] = []
    parents: list[tuple[Any, dict[str, dict[str, Any]]]] = []
    devices: dict[str, Any] = {}
    for name, obj, aliases in sorted(tops.values(), key=lambda t: t[0]):
        node = _node(name, obj, None, signals, parents, set())
        if aliases:
            node["aliases"] = sorted(aliases)
        devices[name] = node
    if describe:
        _add_descriptions(signals, parents, timeout)
    return {"schema_version": SCHEMA_VERSION, "revision": revision, "devices": devices}


def _node(
    path: str,
    obj: Any,
    parent: Any,
    signals: list[tuple[dict[str, Any], Any]],
    parents: list[tuple[Any, dict[str, dict[str, Any]]]],
    ancestors: set[int],
) -> dict[str, Any]:
    classes = [n for c in type(obj).__mro__[:-1] if (n := qualified_name(c))]
    node: dict[str, Any] = {
        "classes": classes or [type(obj).__name__],
        "protocols": [n for n, p in _PROTOCOLS if _satisfies(obj, p)],
    }
    is_signal = isinstance(obj, _SIGNAL_TYPES)
    if is_signal:
        node["signal"] = True
    if isinstance(name := getattr(obj, "name", None), str) and name:
        node["name"] = name
    if summary := _summary(vars(type(obj)).get("__doc__")):
        node["description"] = summary
    node["connected"] = is_connected(obj)
    if access := _access(obj):
        node["access"] = access
    if parent is not None and is_signal and name in _hinted(parent):
        node["read_format"] = "hinted"
    if not is_signal and (hinted := _hinted(obj)):
        node["hinted_fields"] = hinted
    if isinstance(obj, Movable) and (value_type := _set_type(obj)) is not None:
        node["set_type"] = type_expr(value_type)
    if is_signal:
        signals.append((node, obj))
    if isinstance(obj, DeviceVector):
        node["index"] = "int"
    elif isinstance(obj, DeviceMap):
        node["index"] = "str"
    if not is_signal:
        children = {}
        signal_children = {}
        for child_path, child in _iter_children(path, obj):
            if id(child) in ancestors:
                continue
            key = _child_key(child_path[len(path) :])
            children[key] = _node(
                child_path, child, obj, signals, parents, ancestors | {id(obj)}
            )
            if isinstance(child, _SIGNAL_TYPES):
                signal_children[child.name] = children[key]
        if children:
            node["children"] = children
        if signal_children:
            parents.append((obj, signal_children))
    return node


def _satisfies(obj: Any, protocol: type) -> bool:
    try:
        return isinstance(obj, protocol)
    except Exception:
        return False


def _access(obj: Any) -> str | None:
    if isinstance(obj, SignalX):
        return "x"
    if isinstance(obj, SignalRW):
        return "rw"
    if isinstance(obj, SignalW):
        return "w"
    if isinstance(obj, SignalR):
        return "r"
    return None


def _hinted(obj: Any) -> list[str]:
    try:
        fields = (getattr(obj, "hints", None) or {}).get("fields", [])
    except Exception:
        return []
    return [f for f in fields if isinstance(f, str)]


def _set_type(obj: Any) -> type | None:
    try:
        return set_type(obj)
    except Exception:
        return None


def _child_key(step: str) -> str:
    """``.x`` -> ``x``, ``[1]`` -> ``1``, ``['x']`` -> ``x``."""
    if step.startswith("."):
        return step[1:]
    if step.startswith("['"):
        return step[2:-2]
    return step[1:-1]


def _add_descriptions(
    signals: list[tuple[dict[str, Any], Any]],
    parents: list[tuple[Any, dict[str, dict[str, Any]]]],
    timeout: float,
):
    """Add signals' data keys, and how their parents read them."""
    calls = [getattr(signal, "describe", None) for _, signal in signals]
    for parent, _ in parents:
        calls.append(getattr(parent, "describe", None))
        calls.append(getattr(parent, "describe_configuration", None))
    results = _describe_all(calls, timeout)

    for (node, signal), descriptions in zip(
        signals, results[: len(signals)], strict=True
    ):
        if descriptions is not None:
            _set_datakey(node, signal, descriptions)
    parent_results = results[len(signals) :]
    for i, (parent, children) in enumerate(parents):
        read, config = parent_results[2 * i : 2 * i + 2]
        # A failed call leaves its signals unknown rather than unread.
        complete = (read is not None or not hasattr(parent, "describe")) and (
            config is not None or not hasattr(parent, "describe_configuration")
        )
        for name, child in children.items():
            if "read_format" in child:
                continue
            if read is not None and name in read:
                child["read_format"] = "read"
            elif config is not None and name in config:
                child["read_format"] = "config"
            elif complete:
                child["read_format"] = "none"


def _describe_all(
    calls: list[Callable[[], Any] | None], timeout: float
) -> list[Mapping[str, Any] | None]:
    """Call each ``describe``-like method; None for missing or failed calls.

    Coroutine methods run concurrently on the bluesky event loop, and only if it
    is running.
    """
    results: list[Mapping[str, Any] | None] = [None] * len(calls)
    pending = []
    for i, call in enumerate(calls):
        if call is None:
            continue
        if inspect.iscoroutinefunction(call):
            pending.append(i)
            continue
        try:
            results[i] = call()
        except Exception:
            pass
    loop = get_bluesky_event_loop()
    if not pending or loop is None or not loop.is_running() or in_bluesky_event_loop():
        return results

    async def gather():
        async def one(call):
            try:
                return await asyncio.wait_for(call(), timeout)
            except Exception:
                return None

        return await asyncio.gather(*(one(calls[i]) for i in pending))

    for i, result in zip(pending, call_in_bluesky_event_loop(gather()), strict=True):
        results[i] = result
    return results


def _set_datakey(node: dict[str, Any], signal: Any, descriptions: Mapping[str, Any]):
    key = signal.name if signal.name in descriptions else next(iter(descriptions), None)
    if key is None:
        return
    datakey = _json_safe(descriptions[key])
    if isinstance(datakey, dict) and {"dtype", "shape", "source"} <= datakey.keys():
        node["datakey"] = datakey


def _json_safe(obj: Any) -> Any:
    """Return ``obj`` with tuples as lists and numpy scalars as Python ones."""
    if isinstance(obj, enum.Enum):
        return _json_safe(obj.value)
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# -- Shared -------------------------------------------------------------------


def _type_hints(obj: Any) -> dict[str, Any]:
    try:
        return typing.get_type_hints(obj, include_extras=True)
    except Exception:
        return {}


def _summary(doc: str | None) -> str | None:
    if not doc:
        return None
    return inspect.cleandoc(doc).split("\n\n", 1)[0].replace("\n", " ").strip() or None


def _docstring_descriptions(obj: Any, section: str) -> dict[str, str]:
    """Map names to descriptions in a numpydoc Parameters or Attributes section."""
    doc = inspect.getdoc(obj)
    if not doc:
        return {}
    try:
        parsed = docstring_parser.parse(doc, docstring_parser.DocstringStyle.NUMPYDOC)
    except docstring_parser.ParseError:
        return {}
    return {
        p.arg_name: " ".join(p.description.split())
        for p in parsed.params
        if p.description and p.args and p.args[0] == section
    }


def write_catalog(
    namespace: Mapping[str, Any],
    path: str | Path,
    revision: int = 0,
    *,
    describe: bool = True,
) -> None:
    """Write the plan catalog and device registry of ``namespace`` to a YAML file.

    The file holds two YAML documents: the ``PlanCatalog``, then the
    ``DeviceRegistry``.

    Parameters
    ----------
    namespace : Mapping[str, object]
        Where the plans and devices are, typically the IPython ``user_ns``.
    path : str or Path
        The file to write.
    revision : int, optional
        The revision of both documents.
    describe : bool, optional
        Add each signal's data key; see :func:`device_registry`.
    """
    documents = [
        plan_catalog(namespace, revision),
        device_registry(namespace, revision, describe=describe),
    ]
    with Path(path).open("w") as file:
        yaml.safe_dump_all(documents, file, sort_keys=False, allow_unicode=True)
