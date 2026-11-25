"""Find the devices in a namespace, and what kind of object a value or hint is."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping
from typing import Any

_BASES: list[type] = []
_SIGNAL_BASES: list[type] = []
try:  # ophyd_async devices (used throughout hextools).
    from ophyd_async.core import Device as _AsyncDevice
    from ophyd_async.core import DeviceVector as _DeviceVector
    from ophyd_async.core import Signal as _AsyncSignal

    _BASES.append(_AsyncDevice)
    _SIGNAL_BASES.append(_AsyncSignal)
except Exception:  # pragma: no cover - optional dependency layout
    _AsyncDevice = _DeviceVector = None  # type: ignore
try:
    from ophyd_async.core import DeviceMap as _DeviceMap
except Exception:  # pragma: no cover - older ophyd_async
    _DeviceMap = None  # type: ignore
try:  # classic ophyd devices, if present.
    from ophyd.ophydobj import OphydObject as _OphydObject
    from ophyd.signal import Signal as _OphydSignal

    _BASES.append(_OphydObject)
    _SIGNAL_BASES.append(_OphydSignal)
except Exception:  # pragma: no cover - optional dependency layout
    pass

#: ophyd-async and ophyd device base classes.
DEVICE_BASES: tuple[type, ...] = tuple(_BASES)
#: ophyd-async and ophyd signal base classes.
SIGNAL_TYPES: tuple[type, ...] = tuple(_SIGNAL_BASES)


def looks_like_device(obj: Any) -> bool:
    """Return True if ``obj`` is an ophyd or ophyd-async device (or signal)."""
    return bool(DEVICE_BASES) and isinstance(obj, DEVICE_BASES)


def iter_children(name: str, obj: Any) -> Iterator[tuple[str, Any]]:
    """Yield ``(path, child)`` for each direct child device of ``obj``.

    Paths are valid Python expressions relative to the namespace, so they can be
    rendered directly into ``RE(plan(...))`` calls.
    """
    if _AsyncDevice is not None and isinstance(obj, _AsyncDevice):
        for key, child in obj.children():
            if (
                _DeviceVector is not None
                and isinstance(obj, _DeviceVector)
                and key.isdigit()
            ):
                yield f"{name}[{key}]", child
            elif _DeviceMap is not None and isinstance(obj, _DeviceMap) and key in obj:
                yield f"{name}[{key!r}]", child
            else:
                yield f"{name}.{key}", child
        return
    for key in getattr(obj, "component_names", ()) or ():
        try:
            child = getattr(obj, key)
        except Exception:
            continue
        if looks_like_device(child):
            yield f"{name}.{key}", child


def walk_namespace(namespace: Mapping[str, Any]) -> dict[str, Any]:
    """Map path -> device for all devices in ``namespace`` and their children.

    Each device appears once: breadth-first order means a top-level name wins
    over any dotted path to the same object. Among top-level aliases, the one
    matching the device's ``name`` is preferred, then alphabetical order.
    """
    queue: deque[tuple[str, Any]] = deque(
        sorted(
            (
                (name, obj)
                for name, obj in namespace.items()
                if not name.startswith("_") and looks_like_device(obj)
            ),
            key=lambda kv: (kv[0] != getattr(kv[1], "name", None), kv[0]),
        )
    )
    seen: set[int] = set()
    found: dict[str, Any] = {}
    while queue:
        path, obj = queue.popleft()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        found[path] = obj
        queue.extend(iter_children(path, obj))
    return found


def is_connected(obj: Any) -> bool:
    """Return True only if ``obj`` has completed a successful connection."""
    if _AsyncDevice is not None and isinstance(obj, _AsyncDevice):
        # TODO: ophyd-async has no public connection state; replace these private
        # attributes once it does.
        if getattr(obj, "_mock", None) is not None:
            return True
        task = getattr(obj, "_connect_task", None)
        return bool(
            task is not None
            and task.done()
            and not task.cancelled()
            and task.exception() is None
        )
    # Classic ophyd objects expose a ``connected`` property.
    return bool(getattr(obj, "connected", False))


def is_device_type(hint: Any) -> bool:
    """Return True if ``hint`` is a device class or a runtime-checkable protocol."""
    if not isinstance(hint, type):
        return False
    # Classes implementing a protocol inherit _is_runtime_protocol, not _is_protocol.
    if getattr(hint, "_is_protocol", False):
        return getattr(hint, "_is_runtime_protocol", False)
    return bool(DEVICE_BASES) and issubclass(hint, DEVICE_BASES)
