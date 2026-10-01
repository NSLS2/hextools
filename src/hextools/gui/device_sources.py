"""Pluggable sources of device names for plan-input widgets.

A :class:`DeviceSource` answers two questions for a device-typed plan argument:
which device names may be offered, and whether a chosen name is a valid device.
Two implementations are provided:

* :class:`NamespaceDeviceSource` validates against an in-process namespace
  (typically the IPython ``user_ns``), using real ``isinstance`` checks. This
  is exact but only works when the GUI runs in the same process as the devices.
* :class:`QueueServerDeviceSource` validates against the devices the Queue
  Server currently allows, matching on the reported ophyd class name. This works
  out-of-process but relies on class-name matching.

Both feed candidate names into the selector widgets and can be used to validate
selections before a plan is submitted. Device annotations are passed around as a
tuple of Python classes (e.g. ``(KinetixDetector, PhantomDetector)``); pass
``None`` to mean "any device".
"""

from __future__ import annotations


from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from abc import ABC, abstractmethod
from typing import Any

import IPython

_DEVICE_BASES: tuple[type, ...]
_SIGNAL_BASES: list[type] = []
try:  # ophyd_async devices (used throughout hextools).
    from ophyd_async.core import Device as _AsyncDevice
    from ophyd_async.core import DeviceVector as _DeviceVector
    from ophyd_async.core import Signal as _AsyncSignal

    _BASES: list[type] = [_AsyncDevice]
    _SIGNAL_BASES.append(_AsyncSignal)
except Exception:  # pragma: no cover - optional dependency layout
    _AsyncDevice = _DeviceVector = None  # type: ignore[assignment,misc]
    _BASES = []
try:
    from ophyd_async.core import DeviceMap as _DeviceMap
except Exception:  # pragma: no cover - older ophyd_async
    _DeviceMap = None  # type: ignore[assignment,misc]
try:  # classic ophyd devices, if present.
    from ophyd.ophydobj import OphydObject as _OphydObject
    from ophyd.signal import Signal as _OphydSignal

    _BASES.append(_OphydObject)
    _SIGNAL_BASES.append(_OphydSignal)
except Exception:  # pragma: no cover - optional dependency layout
    pass
_DEVICE_BASES = tuple(_BASES)
_SIGNAL_TYPES = tuple(_SIGNAL_BASES)


def _iter_children(name: str, obj: Any) -> Iterator[tuple[str, Any]]:
    """Yield ``(path, child)`` for each direct child device of ``obj``.

    Paths are valid Python expressions relative to the namespace, so they can be
    rendered directly into ``RE(plan(...))`` calls.
    """
    if _AsyncDevice is not None and isinstance(obj, _AsyncDevice):
        for key, child in obj.children():
            if _DeviceVector is not None and isinstance(obj, _DeviceVector) and key.isdigit():
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
        if _looks_like_device(child):
            yield f"{name}.{key}", child


def _walk_namespace(namespace: Mapping[str, Any]) -> dict[str, Any]:
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
                if not name.startswith("_") and _looks_like_device(obj)
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
        queue.extend(_iter_children(path, obj))
    return found


def _looks_like_device(obj: Any) -> bool:
    """Check if an object is an instance of an ophyd(-async) device
    
    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is an instance of an ophyd(-async) device, False otherwise.
    """

    if _DEVICE_BASES and isinstance(obj, _DEVICE_BASES):
        return True
    return False


class DeviceSource(ABC):
    """Abstract source of device names for a device-typed plan argument.

    Subclasses implement :meth:`list_devices` and :meth:`is_valid_device`. The
    ``types`` argument, when given, is a tuple of acceptable device classes taken
    from the plan's type hints; ``None`` means any device is acceptable.
    """

    @abstractmethod
    def list_devices(self, types: tuple[type, ...] | None = None) -> list[str]:
        """Return the sorted names of devices matching ``types``."""
        ...

    @abstractmethod
    def is_valid_device(
        self, name: str, types: tuple[type, ...] | None = None
    ) -> bool:
        """Return True if ``name`` is a known device matching ``types``."""
        ...

    def subscribe(self, callback: Callable[..., None]) -> None:
        """Register ``callback`` to be notified when the device set changes."""
        ...

    def unsubscribe(self, callback: Callable[..., None]) -> None:
        """Remove a previously registered ``callback``."""
        ...


class NamespaceDeviceSource(DeviceSource):
    """Validate device inputs against an in-process (IPython) namespace.

    Parameters
    ----------
    namespace : Mapping[str, object], optional
        Name-to-object mapping to search. Defaults to the current IPython
        ``user_ns`` when available, otherwise an empty mapping.
    """

    def __init__(self, namespace: Mapping[str, Any]):
        self._namespace = namespace

    def list_devices(self, types: tuple[type, ...] | None = None) -> list[str]:
        """Get a list of device_names matching the given types.
        
        Parameters
        ----------
        types : tuple[type, ...] | None, optional
            The acceptable device types. If None, any device type is acceptable.

        Returns
        -------
        list[str]
            The sorted list of device names matching the given types.
        """

        names = [
            name
            for name, obj in _walk_namespace(self._namespace).items()
            if self._matches(obj, types)
        ]
        return sorted(names)

    def is_valid_device(
        self, name: str, types: tuple[type, ...] | None = None
    ) -> bool:
        """Check if the given name corresponds to a valid device of the specified types.
        
        Parameters
        ----------
        name : str
            The name of the device to check.
        types : tuple[type, ...] | None, optional
            The acceptable device types. If None, any device type is acceptable.

        Returns
        -------
        bool
            True if the device is valid and matches the specified types, False otherwise.
        """

        obj = _walk_namespace(self._namespace).get(name)
        if obj is None:
            return False
        return self._matches(obj, types)

    @staticmethod
    def _matches(obj: Any, types: tuple[type, ...] | None) -> bool:
        """Check if the object matches the given types or looks like a device.
        
        Parameters
        ----------
        obj : Any
            The object to check.
        types : tuple[type, ...] | None
            The acceptable types. If None, any object that looks like a device is acceptable.

        Returns
        -------
        bool
            True if the object matches the given types or looks like a device, False otherwise.
        """

        if types:
            return isinstance(obj, types)
        # Untyped inputs would otherwise list every signal of every device.
        return _looks_like_device(obj) and not isinstance(obj, _SIGNAL_TYPES)


class QueueServerDeviceSource(DeviceSource):
    """Validate device inputs against the Queue Server's allowed devices.

    Parameters
    ----------
    model : bluesky_widgets.models.run_engine_client.RunEngineClient
        The Queue Server client model (e.g. ``viewer.run_engine``).
    """

    def __init__(self, model):
        self.model = model

    def _allowed(self) -> dict:
        return getattr(self.model, "_allowed_devices", {}) or {}  # noqa: SLF001

    @staticmethod
    def _wanted_classnames(types: tuple[type, ...] | None) -> set[str] | None:
        if not types:
            return None
        return {t.__name__ for t in types}

    def _walk(self) -> dict[str, dict]:
        """Map dotted path -> info for allowed devices and their components."""
        found: dict[str, dict] = {}
        queue: deque[tuple[str, Any]] = deque(sorted(self._allowed().items()))
        while queue:
            path, info = queue.popleft()
            if not isinstance(info, dict) or path in found:
                continue
            found[path] = info
            for key, child in (info.get("components") or {}).items():
                queue.append((f"{path}.{key}", child))
        return found

    def list_devices(self, types: tuple[type, ...] | None = None) -> list[str]:
        wanted = self._wanted_classnames(types)
        names = [
            name
            for name, info in self._walk().items()
            if wanted is None or info.get("classname") in wanted
        ]
        return sorted(names)

    def is_valid_device(
        self, name: str, types: tuple[type, ...] | None = None
    ) -> bool:
        info = self._walk().get(name)
        if not isinstance(info, dict):
            return False
        wanted = self._wanted_classnames(types)
        return wanted is None or info.get("classname") in wanted

    def subscribe(self, callback: Callable[..., None]) -> None:
        self.model.events.allowed_devices_changed.connect(callback)

    def unsubscribe(self, callback: Callable[..., None]) -> None:
        try:
            self.model.events.allowed_devices_changed.disconnect(callback)
        except (ValueError, TypeError, RuntimeError):
            pass
