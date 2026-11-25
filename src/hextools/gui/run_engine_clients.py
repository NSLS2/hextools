"""How GUI widgets talk to a RunEngine and its devices, wherever they run.

:class:`RunEngineClient` is the interface, and :class:`InProcessRunEngineClient`
drives a RunEngine in the attached IPython session. Plans and devices are
described by a :class:`~hextools.catalog.model.PlanCatalog` and
:class:`~hextools.catalog.model.DeviceRegistry`, and devices are only reached
through the client's reading subscriptions, so widgets never touch plan
functions or device objects.
"""

from __future__ import annotations

import asyncio
import enum
import getpass
import inspect
import json
import keyword
import logging
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

from bluesky import RunEngine
from bluesky.protocols import Reading, Subscribable
from bluesky.utils import RunEngineInterrupted

from hextools.catalog.generate import to_value
from hextools.catalog.model import (
    DeviceRegistry,
    Parameter,
    PlanCatalog,
    PlanDescription,
)
from hextools.catalog.namespace import iter_children, walk_namespace
from hextools.gui._hook_chain import (
    ChainedHook,
    install_hook,
    live_hook,
    uninstall_hook,
)
from hextools.gui._plan_queue import QueueEditor, QueueError, QueuePosition
from hextools.log import _unwrap_failed_status

#: Called with each new reading, possibly from a thread other than the caller's.
ReadingCallback = Callable[[Reading[Any]], None]

#: Called with each progress update: a dict of ``name``, ``initial``, ``current``,
#: ``target``, ``unit``, ``precision``, ``time_elapsed``, ``time_remaining``,
#: ``fraction`` and ``done`` (any may be missing), or ``{"completed": True}``
#: once the RunEngine stops waiting. May be called from any thread.
ProgressCallback = Callable[[dict[str, Any]], None]
#: Called with the RunEngine's new state, e.g. ``"paused"``, from any thread.
StateCallback = Callable[[str], None]
#: Called with the new plan catalog and device registry when either changes.
#: May be called from any thread.
CatalogCallback = Callable[[PlanCatalog, DeviceRegistry], None]
#: Called with ``(created, levelno, levelname, message)`` of each log record,
#: from any thread.
LogCallback = Callable[[float, int, str, str], None]

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: The user group the Queue Server API gives items by default.
_USER_GROUP = "primary"


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)  # atomic, so a crash can't truncate the file


def _qt_call_soon(callback: Callable[[], None]) -> None:
    from qtpy.QtCore import QTimer

    QTimer.singleShot(0, callback)


def _arguments(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> str:
    params = [_source(a) for a in args]
    params += [f"{k}={_source(v)}" for k, v in kwargs.items()]
    return ", ".join(params)


def _json(value: Any) -> Any:
    """Return ``value`` as JSON, with devices as their names, as the Queue Server."""
    if isinstance(value, DeviceName):
        return str(value)
    if isinstance(value, list | tuple):
        return [_json(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class RunInfo:
    """A run opened by the current plan.

    The Queue Server only lists run uids; the rest comes from the start document.
    """

    uid: str
    scan_id: int | None = None
    #: The ``plan_name`` given to ``open_run``, which may name an inner plan.
    plan_name: str | None = None

    @classmethod
    def from_dict(cls, run: Mapping[str, Any]) -> RunInfo:
        """Return the run of a Queue Server run, or of a start document."""
        return cls(run["uid"], run.get("scan_id"), run.get("plan_name"))


@dataclass(frozen=True)
class PlanInfo:
    """The plan the RunEngine is executing, as the Queue Server describes it.

    Devices among the arguments are :class:`DeviceName`, or plain strings from the
    Queue Server; other arguments are JSON-like values.
    """

    name: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    #: The runs the plan has opened so far, oldest first.
    runs: tuple[RunInfo, ...] = ()
    #: When the plan started, in seconds since the epoch.
    start_time: float | None = None
    #: Identifies the plan, and its entry in the plan history once it finishes.
    item_uid: str | None = None

    @property
    def arguments(self) -> str:
        """The call's arguments as Python source, e.g. ``"[det], num=3"``."""
        return _arguments(self.args, self.kwargs)

    @property
    def call(self) -> str:
        """The plan call as Python source, e.g. ``"count([det], num=3)"``."""
        return f"{self.name}({self.arguments})"

    @classmethod
    def from_dict(
        cls, item: Mapping[str, Any], runs: tuple[RunInfo, ...] = ()
    ) -> PlanInfo:
        """Return the plan of a Queue Server ``running_item``."""
        return cls(
            item["name"],
            tuple(item.get("args", ())),
            dict(item.get("kwargs", {})),
            runs,
            (item.get("properties") or {}).get("time_start"),
            item.get("item_uid"),
        )


@dataclass(frozen=True)
class PlanResult:
    """How a finished plan ended, as the Queue Server reports it."""

    #: ``"completed"``, ``"failed"``, ``"stopped"``, ``"aborted"``, ``"halted"``
    #: or ``"unknown"``.
    exit_status: str = "unknown"
    run_uids: tuple[str, ...] = ()
    scan_ids: tuple[int | None, ...] = ()
    #: When the plan started and stopped, in seconds since the epoch.
    time_start: float | None = None
    time_stop: float | None = None
    #: Why the plan failed or was aborted, if known.
    msg: str = ""
    traceback: str = ""

    @classmethod
    def from_dict(cls, result: Mapping[str, Any]) -> PlanResult:
        """Return the result of a Queue Server history item."""
        return cls(
            result.get("exit_status") or "unknown",
            tuple(result.get("run_uids") or ()),
            tuple(result.get("scan_ids") or ()),
            result.get("time_start"),
            result.get("time_stop"),
            result.get("msg") or "",
            result.get("traceback") or "",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a Queue Server history item has it."""
        return {
            "exit_status": self.exit_status,
            "run_uids": list(self.run_uids),
            "scan_ids": list(self.scan_ids),
            "time_start": self.time_start,
            "time_stop": self.time_stop,
            "msg": self.msg,
            "traceback": self.traceback,
        }


@dataclass(frozen=True)
class HistoryItem:
    """A finished plan, as the Queue Server reports it in its plan history.

    Arguments are as in :class:`PlanInfo`.
    """

    name: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    item_uid: str = ""
    result: PlanResult = field(default_factory=PlanResult)

    @property
    def arguments(self) -> str:
        """The call's arguments as Python source, e.g. ``"[det], num=3"``."""
        return _arguments(self.args, self.kwargs)

    @property
    def call(self) -> str:
        """The plan call as Python source, e.g. ``"count([det], num=3)"``."""
        return f"{self.name}({self.arguments})"

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> HistoryItem:
        """Return the plan of a Queue Server history item."""
        return cls(
            item["name"],
            tuple(item.get("args") or ()),
            dict(item.get("kwargs") or {}),
            item.get("item_uid") or "",
            PlanResult.from_dict(item.get("result") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as a Queue Server history item, with devices as names."""
        return {
            "item_type": "plan",
            "name": self.name,
            "args": _json(self.args),
            "kwargs": _json(self.kwargs),
            "item_uid": self.item_uid,
            "result": self.result.to_dict(),
        }


@dataclass(frozen=True)
class QueueItem:
    """A plan waiting in the queue, as the Queue Server describes it.

    Arguments are as in :class:`PlanInfo`. A new item has no ``item_uid``; the
    client gives it one, and fills in ``user`` and ``user_group``, as it is added.
    """

    name: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    item_uid: str = ""
    user: str = ""
    user_group: str = ""

    @property
    def arguments(self) -> str:
        """The call's arguments as Python source, e.g. ``"[det], num=3"``."""
        return _arguments(self.args, self.kwargs)

    @property
    def call(self) -> str:
        """The plan call as Python source, e.g. ``"count([det], num=3)"``."""
        return f"{self.name}({self.arguments})"

    @classmethod
    def from_dict(cls, item: Mapping[str, Any]) -> QueueItem:
        """Return the plan of a Queue Server queue item."""
        return cls(
            item["name"],
            tuple(item.get("args") or ()),
            dict(item.get("kwargs") or {}),
            item.get("item_uid") or "",
            item.get("user") or "",
            item.get("user_group") or "",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as a Queue Server queue item, with devices as names."""
        return {
            "item_type": "plan",
            "name": self.name,
            "args": _json(self.args),
            "kwargs": _json(self.kwargs),
            "item_uid": self.item_uid,
            "user": self.user,
            "user_group": self.user_group,
        }


@dataclass(frozen=True)
class PlanQueue:
    """The plans waiting to run, front first, as the Queue Server's ``queue_get``."""

    items: tuple[QueueItem, ...] = ()
    #: Changes with every edit, so a stale copy of the queue can be detected.
    plan_queue_uid: str = ""
    #: The queued plan being run, taken off the front of the queue.
    running_item: QueueItem | None = None

    @classmethod
    def from_dict(cls, queue: Mapping[str, Any]) -> PlanQueue:
        """Return the queue of a Queue Server ``queue_get`` response."""
        running = queue.get("running_item")
        return cls(
            tuple(QueueItem.from_dict(item) for item in queue.get("items") or ()),
            queue.get("plan_queue_uid") or "",
            QueueItem.from_dict(running) if running else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the queue as a Queue Server ``queue_get`` response has it."""
        return {
            "items": [item.to_dict() for item in self.items],
            "running_item": self.running_item.to_dict() if self.running_item else {},
            "plan_queue_uid": self.plan_queue_uid,
        }


@dataclass(frozen=True)
class QueueStatus:
    """Whether the queue runs, as the Queue Server's ``status`` reports it."""

    #: The queue is running plans, including while one is paused.
    running: bool = False
    #: The queue stops once the running plan finishes.
    stop_pending: bool = False
    #: The queue starts by itself whenever it has plans and the RunEngine is idle.
    autostart: bool = False


def _source(value: Any) -> str:
    if isinstance(value, DeviceName):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_source(v) for v in value) + "]"
    if isinstance(value, tuple):
        inner = ", ".join(_source(v) for v in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, dict):
        items = (f"{_source(k)}: {_source(v)}" for k, v in value.items())
        return "{" + ", ".join(items) + "}"
    return repr(value)


#: Called with the current plan when it starts and when it opens a run, and with
#: None once it finishes. May be called from any thread.
PlanCallback = Callable[[PlanInfo | None], None]
#: Called with the whole plan history, oldest first, each time it changes. May
#: be called from any thread.
HistoryCallback = Callable[[tuple[HistoryItem, ...]], None]
#: Called with the whole plan queue each time it changes. May be called from any
#: thread.
QueueCallback = Callable[[PlanQueue], None]
#: Called with the queue's status each time it changes. May be called from any
#: thread.
QueueStatusCallback = Callable[[QueueStatus], None]


class PlanError(Exception):
    """A plan run through a client didn't complete; ``str`` is a short description.

    Shown to users as an error, unless it is a :class:`PlanPausedError`.
    """


class PlanFailedError(PlanError):
    """A plan failed, or couldn't be run."""


class PlanPausedError(PlanError):
    """A plan paused, and waits to be resumed, stopped, aborted, or halted."""

    def __init__(
        self,
        message: str = (
            "The plan is paused. Press Resume to continue it, "
            "or Stop, Abort, or Halt to end it."
        ),
    ):
        super().__init__(message)


def _plan_error(error: BaseException) -> PlanError:
    """Return ``error`` as a :class:`PlanError`, naming the cause of a failed status."""
    if isinstance(error, PlanError):
        return error
    if isinstance(error, RunEngineInterrupted):
        return PlanPausedError()
    cause = _unwrap_failed_status(error)
    return PlanFailedError(f"{type(cause).__name__}: {cause}")


class DeviceName(str):
    """A plan argument naming a device, rather than a plain string value."""


def check_plan_name(name: str) -> str:
    """Return ``name`` if it is a (dotted) Python name, so it is safe in a command.

    Raises
    ------
    ValueError
        If ``name`` is anything else.
    """
    parts = name.split(".")
    if not all(p.isidentifier() and not keyword.iskeyword(p) for p in parts):
        raise ValueError(f"{name!r} is not a plan name")
    return name


# Dotted names with DeviceVector ``[1]`` and DeviceMap ``['x']`` indexes.
_DEVICE_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\[\d+\]|\['\w+'\])*")


def check_device_name(name: str) -> str:
    """Return ``name`` if it is a device path, so it is safe in a command.

    Raises
    ------
    ValueError
        If ``name`` is anything else.
    """
    words = re.findall(r"[A-Za-z_]\w*", name)
    if not _DEVICE_NAME.fullmatch(name) or any(keyword.iskeyword(w) for w in words):
        raise ValueError(f"{name!r} is not a device name")
    return name


def serialize(value: Any) -> str:
    """Return Python source for a plan argument: a device name, or a literal.

    Raises
    ------
    TypeError
        If ``value`` is not a device name, enum, bool, int, float, string, or a
        list or tuple of those.
    """
    if isinstance(value, DeviceName):
        return check_device_name(value)
    if isinstance(value, enum.Enum):
        value = value.value
    if isinstance(value, bool | int | float | str):
        return repr(value)
    if isinstance(value, list | tuple):
        items = ", ".join(serialize(v) for v in value)
        return f"[{items}]" if isinstance(value, list) else f"({items},)"
    raise TypeError(f"Cannot serialize {type(value).__name__} for a plan call")


def _plan_value(value: Any, names: Mapping[int, str]) -> Any:
    """Return ``value`` as a Queue Server plan argument would be."""
    if id(value) in names:
        return DeviceName(names[id(value)])
    if isinstance(value, enum.Enum):
        return value.value
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if getattr(value, "ndim", None) == 0 and hasattr(value, "item"):
        return value.item()  # numpy scalar
    if isinstance(value, list | tuple):
        return type(value)(_plan_value(v, names) for v in value)
    if isinstance(value, dict):
        return {k: _plan_value(v, names) for k, v in value.items()}
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else repr(value)


def _is_default(value: Any, parameter: Parameter, names: Mapping[int, str]) -> bool:
    if not parameter.has_default:
        return False
    try:
        return to_value(value, names) == parameter.default
    except ValueError:
        return False


def _describe_plan(
    plan: Any,
    describe: Callable[[str], PlanDescription | None],
    names: Mapping[int, str],
    start_time: float | None = None,
) -> PlanInfo:
    """Describe a plan from its generator, before its first message is sent.

    ``describe`` returns the catalog entry of a plan name, whose parameters say
    which of the generator's locals are arguments and how they were passed.
    """
    gen = getattr(plan, "_iter", plan)  # unwrap bluesky's @plan ``Plan`` object
    if not inspect.isgenerator(gen):
        name = getattr(plan, "__name__", None) or type(plan).__name__
        return PlanInfo(name, start_time=start_time)
    name = gen.gi_code.co_name
    description = describe(name)
    if description is None:
        return PlanInfo(name, start_time=start_time)

    # Before the first send, the generator's locals are exactly its arguments.
    local = inspect.getgeneratorlocals(gen)
    extra = any(
        p.kind == "var_positional" and local.get(p.name) for p in description.parameters
    )
    # Required parameters are passed positionally, the rest by keyword.
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for p in description.parameters:
        if p.name not in local:
            continue
        value = local[p.name]
        if p.kind == "var_positional":
            args.extend(value)
        elif p.kind == "var_keyword":
            kwargs.update(value)
        elif p.kind == "positional_only" or (
            p.kind == "positional_or_keyword" and (p.required or extra)
        ):
            args.append(value)
        elif not _is_default(value, p, names):
            kwargs[p.name] = value

    return PlanInfo(
        name,
        tuple(_plan_value(a, names) for a in args),
        {k: _plan_value(v, names) for k, v in kwargs.items()},
        start_time=start_time,
    )


class RunEngineClient(ABC):
    """Query, drive, and watch a RunEngine, and read its devices.

    Subclasses set ``_device_registry`` and ``_plan_catalog``, and call
    :meth:`_set_catalog` whenever either changes.
    """

    _device_registry: DeviceRegistry
    _plan_catalog: PlanCatalog

    def __init__(self):
        self._catalog_callbacks: dict[object, CatalogCallback] = {}

    @property
    def device_registry(self) -> DeviceRegistry:
        """The devices that plans may be given and readings may be taken of."""
        return self._device_registry

    @property
    def plan_catalog(self) -> PlanCatalog:
        """The catalog of available plans."""
        return self._plan_catalog

    def subscribe_catalog(self, callback: CatalogCallback) -> Hashable:
        """Call ``callback`` each time the plan catalog or device registry changes.

        Returns
        -------
        Hashable
            A token to pass to :meth:`unsubscribe_catalog`.
        """
        token = object()
        self._catalog_callbacks[token] = callback
        return token

    def unsubscribe_catalog(self, token: Hashable) -> None:
        """Stop the updates started by :meth:`subscribe_catalog`."""
        self._catalog_callbacks.pop(token, None)

    def _set_catalog(self, catalog: PlanCatalog, registry: DeviceRegistry) -> None:
        """Replace the plan catalog and device registry, and notify subscribers."""
        self._plan_catalog = catalog
        self._device_registry = registry
        for callback in list(self._catalog_callbacks.values()):
            try:
                callback(catalog, registry)
            except Exception:
                pass

    @abstractmethod
    def subscribe_reading(self, name: str, callback: ReadingCallback) -> Hashable:
        """Call ``callback`` with each new reading of device or signal ``name``.

        Returns
        -------
        Hashable
            A token to pass to :meth:`unsubscribe_reading`.

        Raises
        ------
        KeyError
            If ``name`` is not a subscribable device or signal.
        """

    @abstractmethod
    def unsubscribe_reading(self, token: Hashable) -> None:
        """Stop the readings started by :meth:`subscribe_reading`."""

    @abstractmethod
    def fetch_units(self, name: str) -> Future[str | None]:
        """Start fetching the units of ``name``'s reading; the future never raises.

        Raises
        ------
        KeyError
            If there is no device or signal called ``name``.
        """

    @property
    @abstractmethod
    def state(self) -> str:
        """The RunEngine state, e.g. ``"idle"``, ``"running"`` or ``"paused"``."""

    @property
    @abstractmethod
    def mock(self) -> bool:
        """Whether the session's devices are simulated rather than real."""

    @abstractmethod
    def metadata(self) -> Mapping[str, Any]:
        """Return the RunEngine metadata (``RE.md``)."""

    @abstractmethod
    def update_metadata(self, updates: Mapping[str, Any]) -> None:
        """Set the RunEngine metadata keys in ``updates``, leaving the rest."""

    @abstractmethod
    def serialize_plan(self, name: str, *args: Any, **kwargs: Any) -> str:
        """Return the plan call that :meth:`execute_plan` would make, as text.

        Raises
        ------
        ValueError, TypeError
            If a name or argument cannot be passed to a plan.
        """

    @abstractmethod
    def execute_plan(self, name: str, *args: Any, **kwargs: Any) -> None:
        """Run the plan called ``name`` with the given arguments.

        Pass devices as :class:`DeviceName`; other arguments must be enums,
        bools, ints, floats, or strings.

        Raises
        ------
        PlanPausedError
            If the plan paused.
        PlanFailedError
            If the plan could not be run, e.g. as the queue is running, or (in
            process) failed.
        """

    @abstractmethod
    def subscribe_progress(self, callback: ProgressCallback) -> Hashable:
        """Call ``callback`` with progress updates of moves and other waits.

        Returns
        -------
        Hashable
            A token to pass to :meth:`unsubscribe_progress`.
        """

    @abstractmethod
    def unsubscribe_progress(self, token: Hashable) -> None:
        """Stop the updates started by :meth:`subscribe_progress`."""

    @abstractmethod
    def subscribe_state(self, callback: StateCallback) -> Hashable:
        """Call ``callback`` with the new state each time the RunEngine's changes.

        Returns
        -------
        Hashable
            A token to pass to :meth:`unsubscribe_state`.
        """

    @abstractmethod
    def unsubscribe_state(self, token: Hashable) -> None:
        """Stop the updates started by :meth:`subscribe_state`."""

    @abstractmethod
    def current_plan(self) -> PlanInfo | None:
        """Return the plan being executed, or None if there is none."""

    @abstractmethod
    def subscribe_plan(self, callback: PlanCallback) -> Hashable:
        """Call ``callback`` each time the current plan starts, opens a run or ends.

        Returns
        -------
        Hashable
            A token to pass to :meth:`unsubscribe_plan`.
        """

    @abstractmethod
    def unsubscribe_plan(self, token: Hashable) -> None:
        """Stop the updates started by :meth:`subscribe_plan`."""

    @abstractmethod
    def plan_history(self) -> tuple[HistoryItem, ...]:
        """Return the finished plans, oldest first."""

    @abstractmethod
    def subscribe_history(self, callback: HistoryCallback) -> Hashable:
        """Call ``callback`` with the plan history each time it changes.

        Returns
        -------
        Hashable
            A token to pass to :meth:`unsubscribe_history`.
        """

    @abstractmethod
    def unsubscribe_history(self, token: Hashable) -> None:
        """Stop the updates started by :meth:`subscribe_history`."""

    @abstractmethod
    def clear_history(self) -> None:
        """Remove every plan from the history."""

    @abstractmethod
    def save_history(self, path: str | Path) -> None:
        """Write the history to ``path`` as a JSON list of history items.

        Raises
        ------
        OSError
            If the file can't be written.
        """

    # The queue methods are named, and behave, as in the Queue Server's API
    # (bluesky_queueserver_api). Positions are an index (negative counts from the
    # back), "front" or "back". Each raises QueueError, saying why, if it fails, and
    # leaves the queue as it was.

    @abstractmethod
    def queue_get(self) -> PlanQueue:
        """Return the plans waiting to run."""

    @abstractmethod
    def subscribe_queue(self, callback: QueueCallback) -> Hashable:
        """Call ``callback`` with the plan queue each time it changes.

        Returns
        -------
        Hashable
            A token to pass to :meth:`unsubscribe_queue`.
        """

    @abstractmethod
    def unsubscribe_queue(self, token: Hashable) -> None:
        """Stop the updates started by :meth:`subscribe_queue`."""

    @abstractmethod
    def item_add(
        self,
        item: QueueItem,
        *,
        pos: QueuePosition | None = None,
        before_uid: str | None = None,
        after_uid: str | None = None,
    ) -> QueueItem:
        """Add a plan to the queue, at the back by default; return it as queued.

        The plan gets a new UID. An index past either end of the queue adds it
        at that end.
        """

    @abstractmethod
    def item_add_batch(
        self,
        items: Sequence[QueueItem],
        *,
        pos: QueuePosition | None = None,
        before_uid: str | None = None,
        after_uid: str | None = None,
    ) -> tuple[QueueItem, ...]:
        """Add plans together, in order, where :meth:`item_add` adds one.

        If any plan can't be added, none are.
        """

    @abstractmethod
    def item_get(
        self, *, pos: QueuePosition | None = None, uid: str | None = None
    ) -> QueueItem:
        """Return the queued plan at ``pos`` or with ``uid``; the back by default."""

    @abstractmethod
    def item_update(self, item: QueueItem, *, replace: bool = False) -> QueueItem:
        """Put ``item`` in place of the queued plan with its ``item_uid``.

        The plan keeps its UID, unless ``replace`` gives it a new one, for a
        different plan rather than an edit of it. Returns it as queued.
        """

    @abstractmethod
    def item_remove(
        self, *, pos: QueuePosition | None = None, uid: str | None = None
    ) -> QueueItem:
        """Remove and return the plan at ``pos`` or with ``uid``; the back one."""

    @abstractmethod
    def item_remove_batch(
        self, *, uids: Sequence[str], ignore_missing: bool = True
    ) -> tuple[QueueItem, ...]:
        """Remove and return the plans with ``uids``.

        Unless ``ignore_missing``, fail if a UID is repeated or not in the queue.
        """

    @abstractmethod
    def item_move(
        self,
        *,
        pos: QueuePosition | None = None,
        uid: str | None = None,
        pos_dest: QueuePosition | None = None,
        before_uid: str | None = None,
        after_uid: str | None = None,
    ) -> QueueItem:
        """Move the plan at ``pos`` or with ``uid`` to ``pos_dest``, or next to another.

        A plan moved to an index takes the place of the plan there.
        """

    @abstractmethod
    def item_move_batch(
        self,
        *,
        uids: Sequence[str],
        pos_dest: QueuePosition | None = None,
        before_uid: str | None = None,
        after_uid: str | None = None,
        reorder: bool = False,
    ) -> tuple[QueueItem, ...]:
        """Move the plans with ``uids`` together to where :meth:`item_move` moves one.

        They keep the order of ``uids``, or with ``reorder`` their order in the queue.
        """

    @abstractmethod
    def queue_clear(self) -> None:
        """Remove every plan from the queue."""

    @abstractmethod
    def queue_status(self) -> QueueStatus:
        """Return whether the queue is running, stopping, or starts by itself."""

    @abstractmethod
    def subscribe_queue_status(self, callback: QueueStatusCallback) -> Hashable:
        """Call ``callback`` with the queue's status each time it changes.

        Returns
        -------
        Hashable
            A token to pass to :meth:`unsubscribe_queue_status`.
        """

    @abstractmethod
    def unsubscribe_queue_status(self, token: Hashable) -> None:
        """Stop the updates started by :meth:`subscribe_queue_status`."""

    @abstractmethod
    def queue_start(self) -> None:
        """Start running the queued plans, front first, until it is empty or stops.

        A plan that doesn't complete, or is stopped, stops the queue and turns
        autostart off. One that fails, or is aborted or halted, goes back to the
        front of the queue with a new UID. Returns without waiting for plans.
        """

    @abstractmethod
    def queue_stop(self) -> None:
        """Stop the queue once the running plan finishes, and turn autostart off."""

    @abstractmethod
    def queue_stop_cancel(self) -> None:
        """Cancel a :meth:`queue_stop` before the running plan finishes."""

    @abstractmethod
    def queue_autostart(self, enable: bool) -> None:
        """Turn on or off starting the queue whenever it has plans and is idle."""

    @abstractmethod
    def pause(self, defer: bool = False) -> None:
        """Pause the RunEngine.

        Parameters
        ----------
        defer : bool, optional
            If True, pause after the current plan finishes. If False, pause immediately.
            Default is False.
        """

    @abstractmethod
    def resume(self) -> None:
        """Resume the RunEngine if it is paused.

        Raises
        ------
        PlanPausedError
            If the plan paused again.
        PlanFailedError
            If the resumed plan (in process) failed.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop the RunEngine immediately."""

    @abstractmethod
    def abort(self) -> None:
        """Abort the current plan immediately."""

    @abstractmethod
    def halt(self) -> None:
        """Halt the RunEngine immediately."""

    @abstractmethod
    def subscribe_logs(self, callback: LogCallback) -> Hashable:
        """Call ``callback`` for each new log entry.

        Returns
        -------
        Hashable
            A token to pass to :meth:`unsubscribe_logs`.
        """

    @abstractmethod
    def unsubscribe_logs(self, token: Hashable) -> None:
        """Stop the updates started by :meth:`subscribe_logs`."""


class _WaitingHook:
    """A ``RE.waiting_hook`` that reports status-object progress to a callback.

    Any previously installed hook is kept in ``previous`` and still called.
    """

    def __init__(self, callback: ProgressCallback, previous=None):
        self._callback = callback
        self.previous = previous
        self.detached = False
        self._names: dict[int, str] = {}  # id(status) -> device name

    def __call__(self, status_objects):
        previous = live_hook(self.previous)
        if previous is not None:
            try:
                previous(status_objects)
            except Exception:
                pass
        if self.detached:
            return
        if status_objects is None:
            self._names.clear()
            self._callback({"completed": True})
            return
        for status in status_objects:
            if not hasattr(status, "watch"):
                continue
            try:
                status.watch(partial(self._on_watch, status))
            except Exception:
                pass
            try:
                status.add_callback(self._on_finished)
            except Exception:
                pass

    def _on_watch(self, status, *, name=None, **kwargs):
        if self.detached:
            return
        if name is None:
            name = self._names.get(id(status))
        else:
            self._names[id(status)] = name
        self._callback(
            {**kwargs, "name": name, "done": bool(getattr(status, "done", False))}
        )

    def _on_finished(self, status=None, *args, **kwargs):
        if self.detached:
            return
        name = self._names.get(id(status)) if status is not None else None
        if name is None and status is not None:
            name = getattr(getattr(status, "device", None), "name", None)
        if name is not None:
            self._callback({"name": name, "done": True})


class _LogHandler(logging.Handler):
    """Pass each record to a :data:`LogCallback`."""

    def __init__(self, callback: LogCallback):
        super().__init__()
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._callback(
                record.created, record.levelno, record.levelname, record.getMessage()
            )
        except Exception:
            pass


def _exit_status(re: RunEngine, old_state: str) -> str:
    """Return how a plan ended, as a Queue Server exit status.

    ``old_state`` is the state the RunEngine left for ``"idle"``; it tells a stop
    (reported as success) and a halt (reported as abort) from the rest.
    """
    interrupted = {"stopping": "stopped", "halting": "halted", "aborting": "aborted"}
    if old_state in interrupted:
        return interrupted[old_state]
    statuses = {"success": "completed", "fail": "failed", "abort": "aborted"}
    return statuses.get(getattr(re, "_exit_status", ""), "unknown")


def _reading_key(obj: Any, keys: Mapping[str, Any]) -> str | None:
    """Return the key of ``obj``'s own reading in ``keys``: its name, else the first."""
    if getattr(obj, "name", None) in keys:
        return obj.name
    return next(iter(keys), None)


@dataclass(frozen=True, eq=False)
class _ReadingSubscription:
    obj: Subscribable[Any]
    callback: Callable[..., None]


class InProcessRunEngineClient(RunEngineClient):
    """Drive a RunEngine in the attached IPython session.

    Plans run as ``RE(plan(...))`` cells in IPython, so they show in the terminal
    and its history, and :meth:`execute_plan` returns once the plan has finished.
    Plans are only tracked for :meth:`current_plan` while something is subscribed
    with :meth:`subscribe_plan`, and recorded in the plan history while something
    is subscribed with :meth:`subscribe_history`, as that needs hooks on the
    RunEngine.

    The plan catalog and device registry describe the namespace when the client
    is made; :meth:`refresh_catalog` describes it again.

    Parameters
    ----------
    namespace : Mapping[str, Any]
        A namespace (typically ``ipython.user_ns``) holding the RunEngine, as
        ``RE``, and the plans and devices.
    logger_name : str, optional
        The logger whose records :meth:`subscribe_logs` reports.
    history_file : str or Path, optional
        JSON file the plan history is loaded from, and saved to as it changes.
        A file that can't be read is left alone, and the history isn't saved.
    queue_file : str or Path, optional
        JSON file the plan queue is loaded from, and saved to as it changes, in
        the form of a Queue Server ``queue_get`` response. A file that can't be
        read is left alone, and the queue isn't saved.
    call_soon : Callable, optional
        Calls a function on the main thread after the caller returns; queued
        plans run through it, one at a time. Default: a zero-delay Qt timer.
    mock : bool, optional
        Whether the namespace's devices are simulated. Default: False.
    """

    def __init__(
        self,
        namespace: Mapping[str, Any],
        logger_name: str | None = None,
        history_file: str | Path | None = None,
        queue_file: str | Path | None = None,
        call_soon: Callable[[Callable[[], None]], None] | None = None,
        mock: bool = False,
    ):
        super().__init__()
        self._mock = mock
        self._logger = logging.getLogger(logger_name)
        re = namespace.get("RE", None)
        if not isinstance(re, RunEngine):
            raise ValueError("RE not found in the namespace!")
        self._re = re
        self._namespace = namespace
        self._device_registry = DeviceRegistry.from_namespace(namespace)
        self._plan_catalog = PlanCatalog.from_namespace(namespace)
        self._state_callbacks: dict[object, StateCallback] = {}
        self._state_hook: ChainedHook | None = None
        self._plan_callbacks: dict[object, PlanCallback] = {}
        self._history_callbacks: dict[object, HistoryCallback] = {}
        self._plan_hook: ChainedHook | None = None
        self._doc_tokens: list[int] = []
        self._plan: PlanInfo | None = None
        self._stop_reasons: list[str] = []
        self._history_file = Path(history_file).expanduser() if history_file else None
        self._history = self._load_history()
        self._queue_callbacks: dict[object, QueueCallback] = {}
        self._queue_file = Path(queue_file).expanduser() if queue_file else None
        self._queue = self._load_queue()
        self._call_soon = call_soon or _qt_call_soon
        self._queue_status = QueueStatus()
        self._queue_status_callbacks: dict[object, QueueStatusCallback] = {}
        # The running queued plan paused, so resume() or stop() etc. finishes it.
        self._queue_item_paused = False
        # How the running queued plan was interrupted: "stopped", "aborted" etc.
        self._queue_interrupt: str | None = None

    @property
    def state(self) -> str:
        return str(self._re.state)

    @property
    def mock(self) -> bool:
        return self._mock

    def metadata(self) -> Mapping[str, Any]:
        return self._re.md

    def update_metadata(self, updates: Mapping[str, Any]) -> None:
        self._re.md.update(updates)

    def refresh_catalog(self) -> None:
        """Describe the namespace's plans and devices again, and notify subscribers."""
        revision = self._plan_catalog.revision + 1
        self._set_catalog(
            PlanCatalog.from_namespace(self._namespace, revision),
            DeviceRegistry.from_namespace(self._namespace, revision),
        )

    def _device(self, name: str) -> Any:
        """Return the object at registry path ``name``.

        Raises
        ------
        KeyError
            If ``name`` is not in the device registry.
        """
        self._device_registry.node(name)
        top = re.match(r"[A-Za-z_]\w*", name)
        if top is None:
            raise KeyError(name)
        path = self._device_registry.canonical(top.group())
        obj = self._namespace[path]
        name = path + name[top.end() :]
        while path != name:
            for child_path, child in iter_children(path, obj):
                rest = name[len(child_path) :]
                if name.startswith(child_path) and rest[:1] in ("", ".", "["):
                    path, obj = child_path, child
                    break
            else:
                raise KeyError(name)
        return obj

    def subscribe_reading(self, name: str, callback: ReadingCallback) -> Hashable:
        obj = self._device(name)
        if not isinstance(obj, Subscribable):
            raise KeyError(f"{name!r} is not subscribable")

        def on_readings(readings: Mapping[str, Reading[Any]]):
            key = _reading_key(obj, readings)
            if key is not None:
                callback(readings[key])

        # ophyd-async subscriptions must be made on the bluesky event loop.
        self._re.loop.call_soon_threadsafe(obj.subscribe_reading, on_readings)
        return _ReadingSubscription(obj, on_readings)

    def unsubscribe_reading(self, token: Hashable) -> None:
        if not isinstance(token, _ReadingSubscription):
            raise TypeError(f"{token!r} is not a subscription from this client")

        def clear():
            try:
                token.obj.clear_sub(token.callback)
            except KeyError:
                pass

        self._re.loop.call_soon_threadsafe(clear)

    def fetch_units(self, name: str) -> Future[str | None]:
        obj = self._device(name)

        async def fetch() -> str | None:
            try:
                descriptions = obj.describe()
                if inspect.isawaitable(descriptions):
                    descriptions = await descriptions
            except Exception:
                return None
            key = _reading_key(obj, descriptions)
            units = descriptions[key].get("units") if key is not None else None
            return str(units) if units else None

        return asyncio.run_coroutine_threadsafe(fetch(), self._re.loop)

    def serialize_plan(self, name: str, *args: Any, **kwargs: Any) -> str:
        """Return the IPython cell that :meth:`execute_plan` runs."""
        params = [serialize(a) for a in args]
        params += [f"{check_plan_name(k)}={serialize(v)}" for k, v in kwargs.items()]
        return f"RE({check_plan_name(name)}({', '.join(params)}))"

    def execute_plan(self, name: str, *args: Any, **kwargs: Any) -> None:
        if self._queue_status.running:
            raise PlanFailedError(
                "The queue is running: add the plan to the queue, or stop the "
                "queue first."
            )
        try:
            self._run_plan(name, *args, **kwargs)
        finally:
            self._autostart_push()

    def _run_plan(self, name: str, *args: Any, **kwargs: Any) -> None:
        try:
            code = self.serialize_plan(name, *args, **kwargs)
        except (TypeError, ValueError) as error:
            raise _plan_error(error) from error
        self._run_cell(code)

    def _run_cell(self, code: str) -> None:
        from hextools.gui._ipython import run_in_ipython

        error = run_in_ipython(code)
        if error is not None:
            raise _plan_error(error) from error

    def subscribe_progress(self, callback: ProgressCallback) -> Hashable:
        hook = _WaitingHook(callback, getattr(self._re, "waiting_hook", None))
        self._re.waiting_hook = hook  # type: ignore[assignment]
        return hook

    def unsubscribe_progress(self, token: Hashable) -> None:
        uninstall_hook(self._re, "waiting_hook", token)

    def subscribe_state(self, callback: StateCallback) -> Hashable:
        token = object()
        self._state_callbacks[token] = callback
        if self._state_hook is None:
            # One hook for all subscribers, chained so other hooks still run.
            self._state_hook = install_hook(
                self._re, "state_hook", self._on_state_changed
            )
        return token

    def unsubscribe_state(self, token: Hashable) -> None:
        self._state_callbacks.pop(token, None)
        if not self._state_callbacks and self._state_hook is not None:
            uninstall_hook(self._re, "state_hook", self._state_hook)
            self._state_hook = None

    def _on_state_changed(self, new_state, old_state=None):
        for callback in list(self._state_callbacks.values()):
            try:
                callback(str(new_state))
            except Exception:
                pass

    def current_plan(self) -> PlanInfo | None:
        return self._plan

    def subscribe_plan(self, callback: PlanCallback) -> Hashable:
        token = object()
        self._plan_callbacks[token] = callback
        self._update_tracking()
        return token

    def unsubscribe_plan(self, token: Hashable) -> None:
        self._plan_callbacks.pop(token, None)
        self._update_tracking()

    def _update_tracking(self):
        """Install the hooks while plans or history are subscribed to, else remove."""
        wanted = bool(self._plan_callbacks or self._history_callbacks)
        if wanted and self._plan_hook is None:
            self._plan_hook = install_hook(self._re, "state_hook", self._on_plan_state)
            self._doc_tokens = [
                self._re.subscribe(self._on_start, "start"),
                self._re.subscribe(self._on_stop, "stop"),
            ]
        elif not wanted and self._plan_hook is not None:
            uninstall_hook(self._re, "state_hook", self._plan_hook)
            self._plan_hook = None
            for doc_token in self._doc_tokens:
                self._re.unsubscribe(doc_token)
            self._doc_tokens = []
            self._plan = None

    def _on_plan_state(self, new_state, old_state=None):
        if str(new_state) == "running" and str(old_state) == "idle":
            # The state hook runs before the plan's first message, so its
            # arguments are still the generator's only locals.
            namespace = self._namespace
            names = {id(obj): path for path, obj in walk_namespace(namespace).items()}
            plan = _describe_plan(
                getattr(self._re, "_plan", None),  # noqa: SLF001
                self.plan_catalog.get,
                names,
                time.time(),
            )
            self._stop_reasons = []
            running = self._queue.running_item
            uid = running.item_uid if running is not None else str(uuid.uuid4())
            self._set_plan(replace(plan, item_uid=uid))
        elif str(new_state) == "idle" and self._plan is not None:
            plan = self._plan
            self._set_plan(None)
            result = PlanResult(
                _exit_status(self._re, str(old_state)),
                tuple(run.uid for run in plan.runs),
                tuple(run.scan_id for run in plan.runs),
                plan.start_time,
                time.time(),
                "; ".join(self._stop_reasons),
            )
            item = HistoryItem(
                plan.name, plan.args, plan.kwargs, plan.item_uid or "", result
            )
            self._history.append(item)
            self._history_changed()

    def _on_start(self, name, doc):
        if self._plan is None:
            return
        run = RunInfo.from_dict(doc)
        self._set_plan(replace(self._plan, runs=(*self._plan.runs, run)))

    def _on_stop(self, name, doc):
        if reason := doc.get("reason"):
            self._stop_reasons.append(str(reason))

    def _set_plan(self, plan: PlanInfo | None):
        self._plan = plan
        for callback in list(self._plan_callbacks.values()):
            try:
                callback(plan)
            except Exception:
                pass

    def plan_history(self) -> tuple[HistoryItem, ...]:
        return tuple(self._history)

    def subscribe_history(self, callback: HistoryCallback) -> Hashable:
        token = object()
        self._history_callbacks[token] = callback
        self._update_tracking()
        return token

    def unsubscribe_history(self, token: Hashable) -> None:
        self._history_callbacks.pop(token, None)
        self._update_tracking()

    def clear_history(self) -> None:
        self._history.clear()
        self._history_changed()

    def save_history(self, path: str | Path) -> None:
        """Write the history to ``path``, and keep saving there as it changes."""
        path = Path(path).expanduser()
        self._write_history(path)
        self._history_file = path

    def _load_history(self) -> list[HistoryItem]:
        path = self._history_file
        if path is None or not path.exists():
            return []
        try:
            items = [HistoryItem.from_dict(e) for e in json.loads(path.read_text())]
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as ex:
            # Don't overwrite a file we couldn't read; it may be worth recovering.
            logger.warning("Not saving plan history: failed to load %s: %s", path, ex)
            self._history_file = None
            return []
        return [
            replace(
                item,
                args=self._device_names(item.args),
                kwargs=self._device_names(item.kwargs),
            )
            for item in items
        ]

    def _device_names(self, value: Any) -> Any:
        """Mark strings that name registry devices, as the Queue Server does."""
        if isinstance(value, str) and not isinstance(value, DeviceName):
            try:
                self._device_registry.node(value)
            except KeyError:
                return value
            return DeviceName(value)
        if isinstance(value, list | tuple):
            return type(value)(self._device_names(v) for v in value)
        if isinstance(value, Mapping):
            return {k: self._device_names(v) for k, v in value.items()}
        return value

    def _write_history(self, path: Path):
        _write_json(path, [item.to_dict() for item in self._history])

    def _history_changed(self):
        if self._history_file is not None:
            try:
                self._write_history(self._history_file)
            except OSError as ex:
                logger.warning(
                    "Failed to save plan history to %s: %s", self._history_file, ex
                )
        history = tuple(self._history)
        for callback in list(self._history_callbacks.values()):
            try:
                callback(history)
            except Exception:
                pass

    def queue_get(self) -> PlanQueue:
        return self._queue

    def subscribe_queue(self, callback: QueueCallback) -> Hashable:
        token = object()
        self._queue_callbacks[token] = callback
        return token

    def unsubscribe_queue(self, token: Hashable) -> None:
        self._queue_callbacks.pop(token, None)

    def item_add(
        self,
        item: QueueItem,
        *,
        pos: QueuePosition | None = None,
        before_uid: str | None = None,
        after_uid: str | None = None,
    ) -> QueueItem:
        item = self._queued(item, str(uuid.uuid4()))
        return self._edit_queue(
            lambda queue: queue.add(
                item, pos=pos, before_uid=before_uid, after_uid=after_uid
            )
        )

    def item_add_batch(
        self,
        items: Sequence[QueueItem],
        *,
        pos: QueuePosition | None = None,
        before_uid: str | None = None,
        after_uid: str | None = None,
    ) -> tuple[QueueItem, ...]:
        queued = [self._queued(item, str(uuid.uuid4())) for item in items]
        return tuple(
            self._edit_queue(
                lambda queue: queue.add_batch(
                    queued, pos=pos, before_uid=before_uid, after_uid=after_uid
                )
            )
        )

    def item_get(
        self, *, pos: QueuePosition | None = None, uid: str | None = None
    ) -> QueueItem:
        return QueueEditor(self._queue.items).get(pos=pos, uid=uid)

    def item_update(self, item: QueueItem, *, replace: bool = False) -> QueueItem:
        if not item.item_uid:
            raise QueueError("Item description contains no UID")
        queued = self._queued(item, str(uuid.uuid4()) if replace else item.item_uid)
        return self._edit_queue(
            lambda queue: queue.replace(queued, item_uid=item.item_uid)
        )

    def item_remove(
        self, *, pos: QueuePosition | None = None, uid: str | None = None
    ) -> QueueItem:
        return self._edit_queue(lambda queue: queue.remove(pos=pos, uid=uid))

    def item_remove_batch(
        self, *, uids: Sequence[str], ignore_missing: bool = True
    ) -> tuple[QueueItem, ...]:
        return tuple(
            self._edit_queue(
                lambda queue: queue.remove_batch(
                    uids=uids, ignore_missing=ignore_missing
                )
            )
        )

    def item_move(
        self,
        *,
        pos: QueuePosition | None = None,
        uid: str | None = None,
        pos_dest: QueuePosition | None = None,
        before_uid: str | None = None,
        after_uid: str | None = None,
    ) -> QueueItem:
        return self._edit_queue(
            lambda queue: queue.move(
                pos=pos,
                uid=uid,
                pos_dest=pos_dest,
                before_uid=before_uid,
                after_uid=after_uid,
            )
        )

    def item_move_batch(
        self,
        *,
        uids: Sequence[str],
        pos_dest: QueuePosition | None = None,
        before_uid: str | None = None,
        after_uid: str | None = None,
        reorder: bool = False,
    ) -> tuple[QueueItem, ...]:
        return tuple(
            self._edit_queue(
                lambda queue: queue.move_batch(
                    uids=uids,
                    pos_dest=pos_dest,
                    before_uid=before_uid,
                    after_uid=after_uid,
                    reorder=reorder,
                )
            )
        )

    def queue_clear(self) -> None:
        self._edit_queue(lambda queue: queue.items.clear())

    def _queued(self, item: QueueItem, uid: str) -> QueueItem:
        """Return ``item`` with ``uid`` and its user, if the plan can be called so.

        Raises
        ------
        QueueError
            If the plan isn't in the catalog, or its arguments don't fit it.
        """
        description = self.plan_catalog.get(item.name)
        if description is None:
            raise QueueError(f"Plan {item.name!r} is not in the plan catalog")
        signature = inspect.Signature(
            [
                inspect.Parameter(
                    p.name,
                    getattr(inspect.Parameter, p.kind.upper()),
                    default=(
                        inspect.Parameter.empty
                        if p.required or p.kind.startswith("var_")
                        else None
                    ),
                )
                for p in description.parameters
            ],
            __validate_parameters__=False,
        )
        try:
            signature.bind(*item.args, **item.kwargs)
            self.serialize_plan(item.name, *item.args, **item.kwargs)
            self._check_devices([*item.args, *item.kwargs.values()])
        except (TypeError, ValueError) as error:
            raise QueueError(f"Invalid arguments for {item.name!r}: {error}") from error
        return replace(
            item, item_uid=uid, user=getpass.getuser(), user_group=_USER_GROUP
        )

    def _check_devices(self, value: Any) -> None:
        if isinstance(value, DeviceName):
            try:
                self._device_registry.node(value)
            except KeyError:
                raise ValueError(f"There is no device called {value!r}") from None
        elif isinstance(value, list | tuple):
            for v in value:
                self._check_devices(v)

    def _edit_queue(self, edit: Callable[[QueueEditor[QueueItem]], _T]) -> _T:
        """Apply ``edit`` to a copy of the queue, and keep the copy if it succeeds."""
        queue = QueueEditor(self._queue.items)
        result = edit(queue)
        self._set_queue(tuple(queue.items), self._queue.running_item)
        self._autostart_push()
        return result

    def _set_queue(
        self, items: tuple[QueueItem, ...], running_item: QueueItem | None
    ) -> None:
        self._queue = PlanQueue(items, str(uuid.uuid4()), running_item)
        if self._queue_file is not None:
            try:
                _write_json(self._queue_file, self._queue.to_dict())
            except OSError as ex:
                logger.warning(
                    "Failed to save plan queue to %s: %s", self._queue_file, ex
                )
        for callback in list(self._queue_callbacks.values()):
            try:
                callback(self._queue)
            except Exception:
                pass

    def queue_status(self) -> QueueStatus:
        return self._queue_status

    def subscribe_queue_status(self, callback: QueueStatusCallback) -> Hashable:
        token = object()
        self._queue_status_callbacks[token] = callback
        return token

    def unsubscribe_queue_status(self, token: Hashable) -> None:
        self._queue_status_callbacks.pop(token, None)

    def _set_queue_status(self, **changes: bool) -> None:
        status = replace(self._queue_status, **changes)
        if status == self._queue_status:
            return
        self._queue_status = status
        for callback in list(self._queue_status_callbacks.values()):
            try:
                callback(status)
            except Exception:
                pass

    def queue_start(self) -> None:
        if self._queue_status.running:
            raise QueueError("Failed to start the queue: the queue is already running")
        if self.state != "idle":
            raise QueueError(
                f"Failed to start the queue: the RunEngine is {self.state}"
            )
        self._set_queue_status(running=True, stop_pending=False)
        self._call_soon(self._start_next_item)

    def queue_stop(self) -> None:
        if not self._queue_status.running:
            raise QueueError("Failed to stop the queue: the queue is not running")
        self._set_queue_status(stop_pending=True)

    def queue_stop_cancel(self) -> None:
        self._set_queue_status(stop_pending=False)

    def queue_autostart(self, enable: bool) -> None:
        self._set_queue_status(autostart=enable)
        self._autostart_push()

    def _autostart_push(self) -> None:
        """Start the queue if autostart is on and there is something to run."""
        status = self._queue_status
        if (
            status.autostart
            and not status.running
            and self._queue.items
            and self.state == "idle"
        ):
            self._set_queue_status(running=True, stop_pending=False)
            self._call_soon(self._start_next_item)

    def _start_next_item(self) -> None:
        """Run the front plan of the queue, or stop the queue if it shouldn't go on."""
        status = self._queue_status
        if not status.running or self._queue.running_item is not None:
            return
        if status.stop_pending or not self._queue.items or self.state != "idle":
            self._set_queue_status(
                running=False,
                stop_pending=False,
                autostart=status.autostart and not status.stop_pending,
            )
            return
        item, *rest = self._queue.items
        self._set_queue(tuple(rest), item)
        try:
            self._run_plan(item.name, *item.args, **item.kwargs)
        except PlanPausedError:
            self._queue_item_paused = True
            return
        except PlanError as error:
            logger.error("Queued plan %s failed: %s", item.call, error)
            self._finish_queue_item(failed=True)
            return
        self._finish_queue_item(failed=False)

    def _finish_queue_item(self, *, failed: bool) -> None:
        """Take the running queued plan off; go on, or stop if it didn't complete."""
        item = self._queue.running_item
        interrupt, self._queue_interrupt = self._queue_interrupt, None
        self._queue_item_paused = False
        items = self._queue.items
        # As the Queue Server, a stopped plan counts as done; others are run again.
        if item is not None and (failed or interrupt in ("aborted", "halted")):
            items = (replace(item, item_uid=str(uuid.uuid4())), *items)
        self._set_queue(items, None)
        if failed or interrupt:
            self._set_queue_status(running=False, stop_pending=False, autostart=False)
        else:
            self._call_soon(self._start_next_item)

    def _interrupt(self, exit_status: str, request: Callable[[], Any]) -> None:
        """Stop, abort or halt the plan, and if it is queued, stop the queue."""
        if self._queue.running_item is not None:
            self._queue_interrupt = exit_status
        request()
        if self._queue_item_paused:
            self._finish_queue_item(failed=False)

    def _load_queue(self) -> PlanQueue:
        path = self._queue_file
        if path is None or not path.exists():
            return PlanQueue(plan_queue_uid=str(uuid.uuid4()))
        try:
            queue = PlanQueue.from_dict(json.loads(path.read_text()))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as ex:
            # Don't overwrite a file we couldn't read; it may be worth recovering.
            logger.warning("Not saving plan queue: failed to load %s: %s", path, ex)
            self._queue_file = None
            return PlanQueue(plan_queue_uid=str(uuid.uuid4()))
        items = tuple(
            replace(
                item,
                args=self._device_names(item.args),
                kwargs=self._device_names(item.kwargs),
            )
            # A plan running when the session ended never finished, so runs again.
            for item in (*filter(None, [queue.running_item]), *queue.items)
        )
        return replace(queue, items=items, running_item=None)

    def pause(self, defer: bool = False) -> None:
        self._re.request_pause(defer=defer)

    def resume(self) -> None:
        """Resume the paused plan; returns once it finishes or pauses again.

        Runs ``RE.resume()`` as an IPython cell when ``RE`` there is this engine,
        like :meth:`execute_plan`. Must be called on the main thread, since
        resuming installs a SIGINT handler.

        Raises
        ------
        PlanPausedError
            If the plan paused again.
        PlanFailedError
            If the resumed plan failed.
        """
        queued = self._queue_item_paused
        self._queue_item_paused = False
        try:
            self._resume_plan()
        except PlanPausedError:
            self._queue_item_paused = queued
            raise
        except PlanError:
            if queued:
                self._finish_queue_item(failed=True)
            raise
        if queued:
            self._finish_queue_item(failed=False)
        else:
            self._autostart_push()

    def _resume_plan(self) -> None:
        from IPython.core.getipython import get_ipython

        ipython = get_ipython()
        if ipython is not None and ipython.user_ns.get("RE") is self._re:
            self._run_cell("RE.resume()")
            return
        try:
            self._re.resume()
        except Exception as error:
            raise _plan_error(error) from error

    def stop(self) -> None:
        self._interrupt("stopped", self._re.stop)

    def abort(self) -> None:
        self._interrupt("aborted", self._re.abort)

    def halt(self) -> None:
        self._interrupt("halted", self._re.halt)

    def subscribe_logs(self, callback: LogCallback) -> Hashable:
        handler = _LogHandler(callback)
        self._logger.addHandler(handler)
        return handler

    def unsubscribe_logs(self, token: Hashable) -> None:
        if isinstance(token, logging.Handler):
            self._logger.removeHandler(token)
