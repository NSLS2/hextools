"""Where GUI widgets get live readings from, looked up by device or signal name.

:class:`ReadingSource` is the interface; :class:`NamespaceReadingSource` reads from
devices in an in-process (IPython) namespace, as
:class:`~hextools.gui.device_sources.NamespaceDeviceSource` lists them. A remote
implementation would look names up in, and subscribe through, an
ophyd-as-a-service device list instead.
"""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Any

from bluesky.protocols import Reading, Subscribable

#: Called with each new reading, possibly from a thread other than the caller's.
ReadingCallback = Callable[[Reading[Any]], None]


class ReadingSource(ABC):
    """Look up devices and signals by name, and subscribe to their readings."""

    @abstractmethod
    def lookup(self, name: str) -> Any:
        """Return the device or signal called ``name``.

        Raises
        ------
        KeyError
            If there is no device or signal called ``name``.
        """

    @abstractmethod
    def subscribe(self, name: str, callback: ReadingCallback) -> Hashable:
        """Call ``callback`` with each new reading of ``name``.

        Returns
        -------
        Hashable
            A token to pass to :meth:`unsubscribe`.

        Raises
        ------
        KeyError
            If ``name`` is not a subscribable device or signal.
        """

    @abstractmethod
    def unsubscribe(self, token: Hashable) -> None:
        """Remove a subscription made by :meth:`subscribe`."""

    @abstractmethod
    def fetch_units(self, name: str, callback: Callable[[str], None]) -> None:
        """Call ``callback`` with the units of ``name``, if it has any."""


def reading_key(obj: Any, keys: Mapping[str, Any]) -> str | None:
    """Return the key of ``obj``'s own reading in ``keys``: its name, else the first."""
    if getattr(obj, "name", None) in keys:
        return obj.name
    return next(iter(keys), None)


@dataclass(frozen=True, eq=False)
class _NamespaceSubscription:
    obj: Subscribable[Any]
    callback: Callable[..., None]


class NamespaceReadingSource(ReadingSource):
    """Readings from devices in an in-process namespace.

    Names are namespace paths such as ``"a_slits.horizontal_gap"``.

    Parameters
    ----------
    namespace : Mapping[str, object]
        Namespace holding the devices, typically the IPython ``user_ns``.
    loop : asyncio.AbstractEventLoop
        The bluesky event loop (e.g. ``RE.loop``) that the devices live on.
    """

    def __init__(self, namespace: Mapping[str, Any], loop: asyncio.AbstractEventLoop):
        self._namespace = namespace
        self._loop = loop

    def lookup(self, name: str) -> Any:
        parts = name.split(".")
        # Attribute access only, so no code in ``name`` is ever evaluated.
        if not all(p.isidentifier() and not p.startswith("_") for p in parts):
            raise KeyError(name)
        try:
            obj = self._namespace[parts[0]]
            for part in parts[1:]:
                obj = getattr(obj, part)
        except (KeyError, AttributeError):
            raise KeyError(name) from None
        return obj

    def subscribe(self, name: str, callback: ReadingCallback) -> Hashable:
        obj = self.lookup(name)
        if not isinstance(obj, Subscribable):
            raise KeyError(f"{name!r} is not subscribable")

        def on_readings(readings: Mapping[str, Reading[Any]]):
            key = reading_key(obj, readings)
            if key is not None:
                callback(readings[key])

        # ophyd-async subscriptions must be made on the bluesky event loop.
        self._loop.call_soon_threadsafe(obj.subscribe_reading, on_readings)
        return _NamespaceSubscription(obj, on_readings)

    def unsubscribe(self, token: Hashable) -> None:
        if not isinstance(token, _NamespaceSubscription):
            raise TypeError(f"{token!r} is not a subscription from this source")

        def _clear():
            try:
                token.obj.clear_sub(token.callback)
            except KeyError:
                pass

        self._loop.call_soon_threadsafe(_clear)

    def fetch_units(self, name: str, callback: Callable[[str], None]) -> None:
        obj = self.lookup(name)
        describe = getattr(obj, "describe", None)
        if describe is None:
            return

        async def _fetch():
            try:
                descriptions = describe()
                if inspect.isawaitable(descriptions):
                    descriptions = await descriptions
            except Exception:
                return
            key = reading_key(obj, descriptions)
            units = descriptions[key].get("units") if key is not None else None
            if units:
                callback(str(units))

        asyncio.run_coroutine_threadsafe(_fetch(), self._loop)
