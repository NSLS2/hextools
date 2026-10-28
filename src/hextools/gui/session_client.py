"""The GUI's side of the session link, a client of the session's server.

:class:`SessionClient` stands in for the RunEngine in the GUI process, talking to
:class:`~hextools.gui.session_server.SessionServer`. It has the
signals the widgets used to get from :class:`~hextools.gui.plan_status.PlanMonitor`,
keeps the session's latest state, ``RE.md`` and device list, and sends commands
that run as cells in the session.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bluesky.utils import RunEngineInterrupted
from qtpy.QtCore import QCoreApplication, QObject, Signal, Slot
from qtpy.QtNetwork import QLocalSocket

from hextools.gui.device_sources import DeviceSource
from hextools.gui.plan_status import PlanRecord
from hextools.gui.session_protocol import decode, encode

logger = logging.getLogger(__name__)


class SessionError(Exception):
    """An error raised in the session, as reported to the GUI."""

    def __init__(self, info: dict[str, Any]):
        super().__init__(info.get("message", ""))
        self.type_name = info.get("type") or "Error"


def session_error(info: dict[str, Any]) -> BaseException:
    """Return the exception to show for an error the session reported."""
    if info.get("interrupted"):
        return RunEngineInterrupted(info.get("message", ""))
    return SessionError(info)


class SessionClient(QObject):
    """A connection to the IPython session that runs the RunEngine.

    Parameters
    ----------
    path : str or Path
        The session's socket, as given by
        :func:`~hextools.gui.session_server.start_gui`.
    """

    state_changed = Signal(str)
    plan_started = Signal(str, str)
    plan_finished = Signal(object)
    run_started = Signal(object, str, str)
    document = Signal(str, object)
    progress = Signal(object)
    progress_done = Signal()
    devices_changed = Signal()
    md_changed = Signal()
    reading = Signal(str, object)
    disconnected = Signal()

    def __init__(self, path: str | Path, parent=None):
        super().__init__(parent)
        self._path = str(path)
        self._socket = QLocalSocket(self)
        self._buffer = b""
        self._ids = itertools.count(1)
        self._callbacks: dict[int, Callable[[bool, Any], None]] = {}
        self.state = "unknown"
        self.md: dict[str, Any] = {}
        self.re_name = "RE"
        self.logger_name = "bluesky"
        self.tiled: dict[str, Any] | None = None
        self._classes: list[list[str]] = []
        self._devices: dict[str, list] = {}
        self._socket.readyRead.connect(self._on_ready_read)
        self._socket.disconnected.connect(self.disconnected)

    def connect_to_session(self, timeout_ms: int = 10_000) -> None:
        """Connect and read the session's current state; raises ``ConnectionError``."""
        self._socket.connectToServer(self._path)
        if not self._socket.waitForConnected(timeout_ms):
            raise ConnectionError(
                f"Can't reach the session at {self._path}: {self._socket.errorString()}"
            )
        hello: dict[str, Any] = {}
        self.request("hello", lambda ok, result: hello.update(ok=ok, result=result))
        # Not waitForReadyRead: PySide6 holds the GIL while it blocks, which stalls
        # every Python thread here, including a server in the same process.
        deadline = time.monotonic() + timeout_ms / 1000
        while "ok" not in hello:
            if time.monotonic() > deadline:
                raise ConnectionError("The session did not answer")
            time.sleep(0.01)
            QCoreApplication.processEvents()
        if not hello["ok"]:
            raise ConnectionError(f"The session refused the GUI: {hello['result']}")
        result = hello["result"]
        self.state = result["state"]
        self.md = result["md"]
        self.re_name = result["re_name"]
        self.logger_name = result["logger"]
        self.tiled = result["tiled"]
        self._set_devices(result)

    # -- What PlanMonitor offered -----------------------------------------------

    @property
    def run_engine(self) -> SessionClient:
        """The client itself; widgets read its ``state`` as they would a RunEngine's."""
        return self

    # -- Commands ---------------------------------------------------------------

    def request(
        self, op: str, callback: Callable[[bool, Any], None] | None = None, **fields
    ) -> None:
        """Send a request; ``callback(ok, result_or_error)`` runs on the reply."""
        rid = next(self._ids)
        if callback is not None:
            self._callbacks[rid] = callback
        self._socket.write(encode({"id": rid, "op": op, **fields}))
        self._socket.flush()

    def execute(
        self,
        code: str,
        callback: Callable[[BaseException | None], None] | None = None,
    ) -> None:
        """Run ``code`` as a cell in the session; ``callback(error)`` when it ends."""
        self.request("exec", _error_callback(callback), code=code)

    def rerun(
        self,
        record: PlanRecord,
        callback: Callable[[BaseException | None], None] | None = None,
    ) -> None:
        """Run ``record``'s plan again with the same arguments."""
        self.request("rerun", _error_callback(callback), record=record.to_json())

    def request_pause(self, defer: bool = False) -> None:
        self.request("pause", defer=defer)

    def close(self) -> None:
        """Disconnect from the session; the session carries on."""
        self._socket.disconnectFromServer()

    def watch(self, names: list[str]) -> None:
        """Have :attr:`reading` emitted for each new reading of these devices."""
        self.request("watch", names=names)

    def device_status(self, callback: Callable[[list], None]) -> None:
        """``callback([[label, type name, connected], ...])`` for top-level devices."""
        self.request("device_status", lambda ok, result: callback(result if ok else []))

    # -- Devices ----------------------------------------------------------------

    def device_paths(self) -> list[str]:
        return list(self._devices)

    def device_classes(self, path: str) -> list[str] | None:
        """Qualified names of the device's class and its bases, or None if unknown."""
        entry = self._devices.get(path)
        return None if entry is None else self._classes[entry[0]]

    def is_signal(self, path: str) -> bool:
        entry = self._devices.get(path)
        return bool(entry and entry[1])

    def _set_devices(self, table: dict[str, Any]) -> None:
        self._classes = table.get("classes", [])
        self._devices = table.get("devices", {})

    # -- Incoming ---------------------------------------------------------------

    @Slot()
    def _on_ready_read(self):
        self._buffer += bytes(self._socket.readAll().data())
        *lines, self._buffer = self._buffer.split(b"\n")
        for line in lines:
            if not line:
                continue
            try:
                message = decode(line)
            except ValueError:
                logger.warning("Ignoring a malformed message from the session")
                continue
            try:
                if "reply" in message:
                    self._on_reply(message)
                else:
                    self._on_event(message)
            except Exception:
                logger.exception("Failed to handle %r from the session", message)

    def _on_reply(self, message: dict[str, Any]) -> None:
        callback = self._callbacks.pop(message["reply"], None)
        if callback is None:
            return
        ok = bool(message.get("ok"))
        callback(ok, message.get("result") if ok else message.get("error") or {})

    def _on_event(self, message: dict[str, Any]) -> None:
        event = message.get("event")
        if event == "state":
            self.state = message["state"]
            self.state_changed.emit(self.state)
        elif event == "plan_started":
            self.plan_started.emit(message["name"], message["call"])
        elif event == "plan_finished":
            self.plan_finished.emit(PlanRecord.from_json(message["record"]))
        elif event == "run_started":
            self.run_started.emit(
                message["scan_id"], message["uid"], message["plan_name"]
            )
        elif event == "document":
            self.document.emit(message["name"], message["doc"])
        elif event == "progress":
            self.progress.emit(message["msg"])
        elif event == "progress_done":
            self.progress_done.emit()
        elif event == "log":
            _handle_log(message)
        elif event == "md":
            self.md = message["md"]
            self.md_changed.emit()
        elif event == "devices":
            self._set_devices(message)
            self.devices_changed.emit()
        elif event == "reading":
            self.reading.emit(message["name"], message["value"])


def _error_callback(
    callback: Callable[[BaseException | None], None] | None,
) -> Callable[[bool, Any], None] | None:
    if callback is None:
        return None
    return lambda ok, result: callback(None if ok else session_error(result))


def _handle_log(message: dict[str, Any]) -> None:
    """Give a session log record to the GUI's own handlers on that logger.

    Only that logger's handlers (the GUI's log views): propagating would print it
    in the terminal a second time.
    """
    record = logging.makeLogRecord(
        {
            "name": message["logger"],
            "msg": message["message"],
            "levelno": message["levelno"],
            "levelname": message["levelname"],
            "created": message["created"],
        }
    )
    for handler in logging.getLogger(message["logger"]).handlers:
        if record.levelno >= handler.level:
            handler.handle(record)


class SessionDeviceSource(DeviceSource):
    """Devices in the session's namespace, matched on their class and its bases.

    The session sends each device's full class hierarchy, so a device matches a
    type exactly when ``isinstance`` would match it in the session.
    """

    def __init__(self, client: SessionClient):
        self._client = client

    def _matches(self, path: str, types: tuple[type, ...] | None) -> bool:
        classes = self._client.device_classes(path)
        if classes is None:
            return False
        if types:
            wanted = {f"{t.__module__}.{t.__qualname__}" for t in types}
            return not wanted.isdisjoint(classes)
        # Untyped inputs would otherwise list every signal of every device.
        return not self._client.is_signal(path)

    def list_devices(self, types: tuple[type, ...] | None = None) -> list[str]:
        return sorted(p for p in self._client.device_paths() if self._matches(p, types))

    def is_valid_device(self, name: str, types: tuple[type, ...] | None = None) -> bool:
        return self._matches(name, types)

    def subscribe(self, callback: Callable[..., None]) -> None:
        self._client.devices_changed.connect(callback)

    def unsubscribe(self, callback: Callable[..., None]) -> None:
        try:
            self._client.devices_changed.disconnect(callback)
        except (RuntimeError, TypeError):
            pass
