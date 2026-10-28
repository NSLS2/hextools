"""The IPython session's side of the GUI: a local server the GUI process talks to.

The GUI runs in its own process (D-0049), so a crash in it cannot stop a plan or
its cleanup. :func:`start_gui` starts a :class:`SessionServer` in the session and
launches the GUI as a client of it. The server

* runs the GUI's commands as cells at the IPython prompt, echoed to the
  terminal, exactly as if typed there;
* pauses the RunEngine from its own thread, since a plan blocks the main one;
* sends the GUI what it shows: RunEngine state, plans started and finished,
  documents, the narrator's log lines, device progress, the devices in the
  namespace, ``RE.md`` and the readings of devices it asks to watch.

Nothing the GUI does can block the RunEngine: each client has a bounded queue,
and a client that falls behind is dropped. The socket is in a directory only
this user can open, since a client can run any code in the session.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bluesky import RunEngine
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.utils import RunEngineInterrupted
from qtpy.QtCore import QObject, Qt, Signal, Slot
from qtpy.QtWidgets import QApplication

from hextools.gui._hook_chain import uninstall_hook
from hextools.gui._ipython import run_in_ipython
from hextools.gui.available_devices import _discover_devices, _is_connected
from hextools.gui.device_progress_viewer import _WaitingHookDispatcher
from hextools.gui.device_sources import (
    _SIGNAL_TYPES,
    _looks_like_device,
    _walk_namespace,
)
from hextools.gui.errors import unwrap_failed_status
from hextools.gui.narrator import MsgHookNarrator
from hextools.gui.plan_status import PlanMonitor, PlanRecord
from hextools.gui.session_protocol import decode, encode, new_socket_path, to_jsonable
from hextools.gui.spec import load_spec

logger = logging.getLogger(__name__)

#: Messages a client may have waiting before it is dropped as too slow.
MAX_PENDING = 10_000
_MD_POLL_S = 1.0


class _MainThread(QObject):
    """Runs callables on the main thread, from the Qt event loop IPython drives."""

    _call = Signal(object)

    def __init__(self):
        super().__init__()
        self._call.connect(self._run, Qt.ConnectionType.QueuedConnection)

    def submit(self, fn: Callable[[], Any]) -> None:
        self._call.emit(fn)

    @Slot(object)
    def _run(self, fn):
        try:
            fn()
        except Exception:
            logger.exception("GUI request failed in the session")


def error_info(error: BaseException) -> dict[str, Any]:
    """Return what the GUI needs to report ``error``."""
    cause = unwrap_failed_status(error)
    return {
        "type": type(cause).__name__,
        "message": str(cause),
        "interrupted": isinstance(error, RunEngineInterrupted),
    }


def exec_in(namespace: dict[str, Any]) -> Callable[[str], BaseException | None]:
    """Run code in ``namespace`` and return any error; for sessions without IPython."""

    def execute(code: str) -> BaseException | None:
        try:
            exec(compile(code, "<gui>", "exec"), namespace)  # noqa: S102
        except BaseException as error:
            return error
        return None

    return execute


class _Client:
    """One connected GUI: a reader thread and a writer fed by a bounded queue."""

    def __init__(self, server: SessionServer, conn: socket.socket):
        self._server = server
        self._conn = conn
        self._out: queue.Queue[dict[str, Any] | None] = queue.Queue(MAX_PENDING)
        self.watches: list[tuple[Any, Callable]] = []
        self.closed = False
        threading.Thread(target=self._write, daemon=True, name="gui-writer").start()
        threading.Thread(target=self._read, daemon=True, name="gui-reader").start()

    def send(self, message: dict[str, Any]) -> None:
        """Queue ``message``; never blocks. A client that falls behind is dropped."""
        if self.closed:
            return
        try:
            self._out.put_nowait(message)
        except queue.Full:
            logger.warning("GUI client fell behind; disconnecting it")
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._server._forget(self)  # noqa: SLF001
        try:
            self._out.put_nowait(None)
        except queue.Full:
            pass
        try:
            self._conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._conn.close()

    def _write(self):
        while True:
            message = self._out.get()
            if message is None:
                return
            try:
                self._conn.sendall(encode(message))
            except OSError:
                self.close()
                return
            except Exception:
                logger.exception("Could not send %r to the GUI", message.get("event"))

    def _read(self):
        try:
            with self._conn.makefile("rb") as lines:
                for line in lines:
                    try:
                        request = decode(line)
                    except ValueError:
                        logger.warning("Ignoring a malformed GUI request")
                        continue
                    self._server._handle(self, request)  # noqa: SLF001
        except OSError:
            pass
        finally:
            self.close()


class SessionServer:
    """Serves one IPython session's RunEngine and namespace to GUI processes.

    Parameters
    ----------
    re : RunEngine
        The session's RunEngine.
    namespace : dict
        The session's namespace (IPython's ``user_ns``).
    logger_name : str
        Logger whose records the GUI shows; the narrator writes to it.
    re_name : str
        Name of ``re`` in the namespace, used in the commands the GUI sends.
    execute : callable, optional
        Runs a command and returns any error. Default: run it as an IPython
        cell when IPython is running, else in ``namespace``.
    path : Path, optional
        Socket to listen on. Default: a new one only this user can open.
    """

    def __init__(
        self,
        re: RunEngine,
        namespace: dict[str, Any],
        *,
        logger_name: str = "bluesky",
        re_name: str = "RE",
        execute: Callable[[str], BaseException | None] | None = None,
        path: Path | None = None,
    ):
        self._re = re
        self._namespace = namespace
        self._logger_name = logger_name
        self._re_name = re_name
        self._execute = execute or _default_execute(namespace)
        self._clients: set[_Client] = set()
        self._lock = threading.Lock()
        self._closed = False
        # BestEffortCallbacks whose plots were switched off for an attached GUI.
        self._quieted: list[BestEffortCallback] = []
        QApplication.instance() or QApplication([])
        self._main = _MainThread()

        self._devices: dict[str, Any] = {}
        self._device_key: tuple = ()
        self._device_table: dict[str, Any] = {"classes": [], "devices": {}}
        self._refresh_devices()
        # Plans defined at the prompt or in the profile can't be imported by reference.
        self._plan_functions: dict[str, Callable] = {}

        self._monitor = PlanMonitor(
            re, narrator=MsgHookNarrator(logger=logger_name), namespace=namespace
        )
        direct = Qt.ConnectionType.DirectConnection
        # Emitted on the RunEngine's thread; sending only queues the message.
        self._monitor.state_changed.connect(
            lambda state: self.broadcast({"event": "state", "state": state}), direct
        )
        self._monitor.plan_started.connect(
            lambda name, call: self.broadcast(
                {"event": "plan_started", "name": name, "call": call}
            ),
            direct,
        )
        self._monitor.plan_finished.connect(
            self._on_plan_finished,
            direct,
        )
        self._monitor.run_started.connect(
            lambda scan_id, uid, plan_name: self.broadcast(
                {
                    "event": "run_started",
                    "scan_id": scan_id,
                    "uid": uid,
                    "plan_name": plan_name,
                }
            ),
            direct,
        )
        self._monitor.document.connect(
            lambda name, doc: self.broadcast(
                {"event": "document", "name": name, "doc": doc}
            ),
            direct,
        )

        sink = SimpleNamespace(
            signal_progress_update=SimpleNamespace(
                emit=lambda msg: self.broadcast({"event": "progress", "msg": msg})
            ),
            signal_completed=SimpleNamespace(
                emit=lambda: self.broadcast({"event": "progress_done"})
            ),
        )
        self._waiting_hook = _WaitingHookDispatcher(
            sink,  # type: ignore[arg-type]
            getattr(re, "waiting_hook", None),
        )
        re.waiting_hook = self._waiting_hook  # type: ignore[assignment]

        self._log_handler = _ForwardHandler(self, logger_name)
        logging.getLogger(logger_name).addHandler(self._log_handler)

        self._md = self._read_md()
        threading.Thread(target=self._watch_md, daemon=True, name="gui-md").start()

        self.path = path or new_socket_path()
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.bind(str(self.path))
        self._socket.listen()
        # Any process that makes a server, not only start_gui's, must remove its socket.
        atexit.register(self.close)
        threading.Thread(target=self._accept, daemon=True, name="gui-accept").start()

    # -- Connections ------------------------------------------------------------

    def _accept(self):
        while not self._closed:
            try:
                conn, _ = self._socket.accept()
            except OSError:
                return
            with self._lock:
                self._clients.add(_Client(self, conn))
                first = len(self._clients) == 1
            if first:
                self._quiet_session_plots()

    def _forget(self, client: _Client) -> None:
        with self._lock:
            self._clients.discard(client)
            last = not self._clients
        for obj, callback in client.watches:
            self._re.loop.call_soon_threadsafe(_clear_sub, obj, callback)
        client.watches.clear()
        if last:
            self._restore_session_plots()

    def _quiet_session_plots(self):
        """Stop the session's own plots; the GUI draws them."""
        callbacks = _retry_while_changing(
            lambda: [
                obj
                for obj in self._namespace.values()
                if isinstance(obj, BestEffortCallback)
            ]
        )
        for bec in callbacks:
            if bec._plots_enabled:  # noqa: SLF001
                bec.disable_plots()
                self._quieted.append(bec)

    def _restore_session_plots(self):
        while self._quieted:
            self._quieted.pop().enable_plots()

    def broadcast(self, message: dict[str, Any]) -> None:
        """Queue ``message`` for every connected GUI; safe from any thread."""
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            client.send(message)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def close(self) -> None:
        """Disconnect every GUI, take the hooks off the RunEngine, remove the socket."""
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        self._monitor.detach()
        uninstall_hook(self._re, "waiting_hook", self._waiting_hook)
        logging.getLogger(self._logger_name).removeHandler(self._log_handler)
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            client.close()
        self._socket.close()
        shutil.rmtree(self.path.parent, ignore_errors=True)

    # -- Requests ---------------------------------------------------------------

    def _handle(self, client: _Client, request: dict[str, Any]) -> None:
        rid, op = request.get("id"), request.get("op")
        handler = {
            "hello": self._op_hello,
            "exec": self._op_exec,
            "rerun": self._op_rerun,
            "pause": self._op_pause,
            "device_status": self._op_device_status,
            "watch": self._op_watch,
        }.get(str(op))
        if handler is None:
            client.send(
                {"reply": rid, "ok": False, "error": {"message": f"Unknown op {op!r}"}}
            )
            return
        try:
            handler(client, rid, request)
        except Exception as error:
            client.send({"reply": rid, "ok": False, "error": error_info(error)})

    @staticmethod
    def _reply(client: _Client, rid, result: Any = None) -> None:
        client.send({"reply": rid, "ok": True, "result": result})

    def _op_hello(self, client, rid, request):
        self._reply(
            client,
            rid,
            {
                "state": str(self._re.state),
                "md": self._md,
                "re_name": self._re_name,
                "logger": self._logger_name,
                "tiled": self._tiled_info(),
                **self._device_table,
            },
        )

    def _op_exec(self, client, rid, request):
        code = str(request["code"])
        self._main.submit(lambda: self._run_command(client, rid, code))

    def _op_rerun(self, client, rid, request):
        record = PlanRecord.from_json(request["record"])
        self._main.submit(lambda: self._rerun(client, rid, record))

    def _op_pause(self, client, rid, request):
        # A plan holds the main thread; bluesky allows this from any other one.
        self._re.request_pause(bool(request.get("defer", False)))
        self._reply(client, rid)

    def _op_device_status(self, client, rid, request):
        entries = _retry_while_changing(lambda: _discover_devices(self._namespace))
        self._reply(
            client,
            rid,
            [
                [label, type_name, _is_connected(obj)]
                for label, type_name, obj in entries
            ],
        )

    def _op_watch(self, client, rid, request):
        for name in request.get("names", []):
            obj = self._devices.get(name)
            if obj is None or not hasattr(obj, "subscribe_reading"):
                continue
            callback = partial(self._on_reading, client, name)
            client.watches.append((obj, callback))
            # ophyd-async subscriptions belong on the bluesky event loop.
            self._re.loop.call_soon_threadsafe(obj.subscribe_reading, callback)
        self._reply(client, rid)

    @staticmethod
    def _on_reading(client: _Client, name: str, readings: Mapping[str, Any]):
        for reading in readings.values():
            client.send({"event": "reading", "name": name, "value": reading["value"]})

    def _on_plan_finished(self, record: PlanRecord):
        data = record.to_json()
        if record.func is not None and data["function"]:
            self._plan_functions[data["function"]] = record.func
        self.broadcast({"event": "plan_finished", "record": data})

    def _run_command(self, client, rid, code: str):
        error = self._execute(code)
        if error is None:
            self._reply(client, rid)
        else:
            client.send({"reply": rid, "ok": False, "error": error_info(error)})
        self._after_command()

    def _rerun(self, client, rid, record: PlanRecord):
        func = self._plan_functions.get(record.function or "") or record.func
        existing = self._namespace.get(record.name)
        if func is None:
            message = f"Can't find the plan function for {record.name!r}."
            client.send({"reply": rid, "ok": False, "error": {"message": message}})
            return
        if existing is None:
            self._namespace[record.name] = func
        elif existing is not func:
            client.send(
                {
                    "reply": rid,
                    "ok": False,
                    "error": {
                        "type": "NameConflict",
                        "message": f"'{record.name}' in the session refers to a "
                        "different object than the plan that originally ran, so "
                        "it won't be re-run.",
                    },
                }
            )
            return
        self._run_command(client, rid, f"{self._re_name}({record.call})")

    # -- What the GUI shows -----------------------------------------------------

    def _after_command(self):
        """Tell the GUI about devices created or deleted by a command."""
        if self._refresh_devices():
            self.broadcast({"event": "devices", **self._device_table})

    def _refresh_devices(self) -> bool:
        """Re-walk the namespace if its top-level devices changed; True if they did."""
        key = _retry_while_changing(
            lambda: tuple(
                sorted(
                    (name, id(obj))
                    for name, obj in self._namespace.items()
                    if not name.startswith("_") and _looks_like_device(obj)
                )
            )
        )
        if key == self._device_key:
            return False
        self._device_key = key
        self._devices = _retry_while_changing(lambda: _walk_namespace(self._namespace))
        classes: list[list[str]] = []
        index: dict[type, int] = {}
        table: dict[str, list] = {}
        for path, obj in self._devices.items():
            cls = type(obj)
            if cls not in index:
                index[cls] = len(classes)
                classes.append(
                    [f"{c.__module__}.{c.__qualname__}" for c in cls.__mro__]
                )
            table[path] = [index[cls], isinstance(obj, _SIGNAL_TYPES)]
        self._device_table = {"classes": classes, "devices": table}
        return True

    def after_cell(self, *_):
        """For IPython's ``post_run_cell`` event: devices typed at the prompt."""
        self._after_command()

    def _tiled_info(self) -> dict[str, Any] | None:
        client = self._namespace.get("tiled_reading_client") or self._namespace.get("c")
        uri = getattr(client, "uri", None)
        if not isinstance(uri, str):
            return None
        context = getattr(client, "context", None)
        return {"uri": uri, "api_key": getattr(context, "api_key", None)}

    def _read_md(self) -> dict[str, Any]:
        try:
            return to_jsonable(dict(self._re.md))
        except Exception:
            return {}

    def _watch_md(self):
        while not self._closed:
            time.sleep(_MD_POLL_S)
            md = self._read_md()
            if json.dumps(md, sort_keys=True) != json.dumps(self._md, sort_keys=True):
                self._md = md
                self.broadcast({"event": "md", "md": md})


class _ForwardHandler(logging.Handler):
    """Sends the GUI each record logged to the GUI's logger."""

    def __init__(self, server: SessionServer, logger_name: str):
        super().__init__()
        self._server = server
        self._logger_name = logger_name

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        self._server.broadcast(
            {
                "event": "log",
                "logger": self._logger_name,
                "created": record.created,
                "levelno": record.levelno,
                "levelname": record.levelname,
                "message": message,
            }
        )


def _clear_sub(obj, callback):
    try:
        obj.clear_sub(callback)
    except KeyError:
        pass


def _retry_while_changing(read: Callable[[], Any], attempts: int = 5) -> Any:
    """Call ``read``, retrying if the namespace changed size while it iterated."""
    for _ in range(attempts - 1):
        try:
            return read()
        except RuntimeError:
            time.sleep(0.01)
    return read()


def _default_execute(
    namespace: dict[str, Any],
) -> Callable[[str], BaseException | None]:
    from IPython.core.getipython import get_ipython

    if get_ipython() is not None:
        return run_in_ipython
    return exec_in(namespace)


# -- Launching the GUI from the session ----------------------------------------

_server: SessionServer | None = None


def start_gui(spec_ref: str, *, end_session_on_close: bool = True) -> SessionServer:
    """Serve this session and open the GUI for ``spec_ref`` in its own process.

    Run in the IPython session that defines ``RE``. With
    ``end_session_on_close``, closing the GUI window ends the session too. If the
    GUI crashes, the session and any running plan carry on; ``restart_gui()``
    opens it again.
    """
    global _server
    from IPython.core.getipython import get_ipython

    ipython = get_ipython()
    if ipython is None:
        raise RuntimeError("start_gui must run in an IPython session")
    namespace = ipython.user_ns
    re = namespace.get("RE")
    if not isinstance(re, RunEngine):
        raise RuntimeError("RE is not a RunEngine in this session's namespace")
    if _server is None:
        spec = load_spec(spec_ref)
        _server = SessionServer(re, namespace, logger_name=spec.logger_name)
        ipython.events.register("post_run_cell", _server.after_cell)
    server = _server
    namespace["restart_gui"] = partial(_launch, server, spec_ref, False)
    _launch(server, spec_ref, end_session_on_close)
    return server


def _launch(server: SessionServer, spec_ref: str, end_session_on_close: bool) -> None:
    """Start the GUI process, unless one is already connected."""
    if server.client_count:
        _print("[GUI] already open")
        return
    log_path = server.path.parent / "gui.log"
    env = dict(os.environ)
    # The GUI must run the same hextools as the session, or the two may disagree.
    package_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (package_root, env.get("PYTHONPATH")) if p
    )
    with open(log_path, "ab") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "hextools.gui._client_main",
                "--connect",
                str(server.path),
                spec_ref,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    threading.Thread(
        target=_wait_for_gui,
        args=(server, process, log_path, end_session_on_close),
        daemon=True,
        name="gui-process",
    ).start()


def _wait_for_gui(server, process, log_path: Path, end_session_on_close: bool):
    code = process.wait()
    if server._closed:  # noqa: SLF001
        return
    if code == 0:
        if end_session_on_close:
            server._main.submit(_end_ipython_session)  # noqa: SLF001
        else:
            _print("[GUI] closed; restart_gui() opens it again.")
        return
    _print(
        f"[GUI] exited unexpectedly (code {code}). The session and any running plan"
        f" are unaffected. Its output is in {log_path}; restart_gui() opens it again."
    )


def _print(text: str) -> None:
    out = sys.__stdout__
    if out is None:
        return
    prefix = "\r\033[K" if out.isatty() else ""
    print(f"{prefix}{text}", file=out, flush=True)


def _end_ipython_session() -> None:
    """Make the IPython shell exit, without waiting for a keypress."""
    from IPython.core.getipython import get_ipython
    from IPython.terminal.interactiveshell import TerminalInteractiveShell

    ipython = get_ipython()
    # Only a terminal session; a Jupyter kernel's shell has ask_exit too, and
    # is left running.
    if not isinstance(ipython, TerminalInteractiveShell):
        return
    ipython.ask_exit()
    pt_app = getattr(ipython, "pt_app", None)
    if pt_app is None or not pt_app.app.is_running:
        return
    # Qt runs inside prompt_toolkit's input hook, which only returns when its
    # event loop is woken; call_soon_threadsafe wakes it.
    asyncio.get_running_loop().call_soon_threadsafe(pt_app.app.exit, "")
