import os
import signal
import sys
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

import IPython  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from bluesky.plans import count  # noqa: E402
from IPython.terminal.interactiveshell import TerminalInteractiveShell  # noqa: E402
from ophyd_async.core import init_devices  # noqa: E402
from ophyd_async.sim import PatternGenerator, SimPointDetector  # noqa: E402
from qtpy.QtCore import QCoreApplication, QEvent, QProcess  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

import hextools.gui.__main__ as gui_main  # noqa: E402
import hextools.gui.session_server as session_server  # noqa: E402
from hextools.gui.auto_exporter_view import QtAutoExporterView  # noqa: E402
from hextools.gui.session_client import SessionClient  # noqa: E402
from hextools.gui.session_server import SessionServer, exec_in  # noqa: E402
from hextools.gui.window import QtDataAcquisitionWindow  # noqa: E402
from hextools.gui_config import HEX_GUI  # noqa: E402


@pytest.fixture
def det(RE):
    with init_devices():
        det = SimPointDetector(PatternGenerator())
    return det


def _hooks(RE):
    callbacks = sum(len(cbs) for cbs in RE.dispatcher.cb_registry.callbacks.values())
    return RE.state_hook, RE.msg_hook, RE.waiting_hook, callbacks


def _wait_until(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        QCoreApplication.processEvents()
        time.sleep(0.01)


@pytest.fixture
def server(RE, det):
    QApplication.instance() or QApplication([])
    namespace = {"RE": RE, "det": det, "count": count}
    before = _hooks(RE)
    server = SessionServer(RE, namespace, execute=exec_in(namespace))
    yield server
    server.close()
    # The session outlives every GUI and must be left as it was.
    assert _hooks(RE) == before, "the session server left its hooks on the RunEngine"
    RE(count([det], 1))
    assert RE.state == "idle"


def _connect(server):
    client = SessionClient(server.path)
    client.connect_to_session()
    return client


def _open_window(server, tmp_path, request):
    """Build the GUI with the auto exporter running as a stand-in child.

    The child records a SIGTERM to ``tmp_path / "sigterm"``, so a graceful stop
    is distinguishable from the kill QProcess's destructor does.
    """
    window = QtDataAcquisitionWindow(re_client=_connect(server), spec=HEX_GUI)
    exporter = window.qt_widget.extra_tabs["NXS Export"]
    assert isinstance(exporter, QtAutoExporterView)
    marker = tmp_path / "sigterm"
    child = (
        "import signal, sys, time\n"
        "def on_term(*_):\n"
        f"    open({str(marker)!r}, 'w').close()\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, on_term)\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    start = exporter._process.start
    exporter._process.start = lambda *_: start(sys.executable, ["-c", child])  # type: ignore
    exporter.start()
    assert exporter._process.waitForReadyRead(10000)
    assert exporter._process.state() == QProcess.ProcessState.Running
    pid = exporter._process.processId()

    def kill_leftover_child():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    request.addfinalizer(kill_leftover_child)
    return window, marker


def _flush_deletes():
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QCoreApplication.processEvents()


# Closing the window ran none of the GUI's teardown: the exporter child was killed
# by QProcess's destructor, and RE kept hooks into deleted widgets, so the next
# RE(...) raised "Signal source has been deleted". The hooks now belong to the
# session server (checked by the fixture); the window must still stop its child
# and let go of the session.
def test_closing_the_window_tears_down(server, tmp_path, request):
    window, marker = _open_window(server, tmp_path, request)
    # Closing one deletes its viewport mid-teardown, as the camera viewers do.
    window.qt_widget.layout().addWidget(pg.GraphicsLayoutWidget())

    window._qt_window.close()
    _flush_deletes()

    assert marker.exists(), "auto exporter was killed, not stopped"
    _wait_until(lambda: server.client_count == 0)


def test_the_window_closes_when_the_session_ends(server, tmp_path, request):
    window, marker = _open_window(server, tmp_path, request)

    server.close()

    _wait_until(marker.exists)
    assert window._torn_down


# PR 97 review: closing one of two windows must leave the other working.
@pytest.mark.parametrize("first", ["older", "newer"])
def test_closing_windows_that_share_a_session(server, first):
    older = QtDataAcquisitionWindow(re_client=_connect(server), spec=HEX_GUI)
    newer = QtDataAcquisitionWindow(re_client=_connect(server), spec=HEX_GUI)
    assert server.client_count == 2

    for left, window in enumerate(
        [older, newer] if first == "older" else [newer, older]
    ):
        window._qt_window.close()
        _flush_deletes()
        _wait_until(lambda n=1 - left: server.client_count == n)


class _Process:
    def __init__(self, code):
        self._code = code

    def wait(self):
        return self._code


# Nghia (2026-10-21): closing the GUI started with `hex-gui` left its IPython
# session running underneath. Since D-0049 the GUI is a separate process, so a
# crash must not end the session, only a clean close when asked.
@pytest.mark.parametrize(
    ("code", "end_session", "ends"),
    [(0, True, True), (0, False, False), (1, True, False), (-9, True, False)],
)
def test_the_gui_process_ends_the_session_only_on_a_clean_close(
    monkeypatch, tmp_path, code, end_session, ends
):
    ended, printed = [], []
    monkeypatch.setattr(session_server, "_end_ipython_session", lambda: ended.append(1))
    monkeypatch.setattr(session_server, "_print", printed.append)
    server = SimpleNamespace(_closed=False, _main=SimpleNamespace(submit=lambda f: f()))

    session_server._wait_for_gui(
        server, _Process(code), tmp_path / "gui.log", end_session
    )

    assert ended == ([1] if ends else [])
    crashed = [line for line in printed if "exited unexpectedly" in line]
    assert bool(crashed) == (code != 0)


class _TerminalShell(TerminalInteractiveShell):
    """A terminal shell with no prompt running (pt_app is None)."""

    def __init__(self):
        self.asked = []

    def ask_exit(self):
        self.asked.append(True)


class _KernelShell:
    """Stands in for ipykernel's ZMQInteractiveShell, which also has ask_exit."""

    kernel = object()

    def __init__(self):
        self.asked = []

    def ask_exit(self):
        self.asked.append(True)


# PR 101 review: a Jupyter kernel's shell has ask_exit too and must be left alone.
@pytest.mark.parametrize(
    ("shell", "ends"), [(_TerminalShell, True), (_KernelShell, False)]
)
def test_only_a_terminal_ipython_session_is_ended(monkeypatch, shell, ends):
    ipython = shell()
    monkeypatch.setattr("IPython.core.getipython.get_ipython", lambda: ipython)

    session_server._end_ipython_session()

    assert ipython.asked == ([True] if ends else [])


class _PromptApp:
    def __init__(self, is_running):
        self.is_running = is_running

    def exit(self, result):
        pass


class _Loop:
    def __init__(self):
        self.scheduled = []

    def call_soon_threadsafe(self, callback, *args):
        self.scheduled.append((callback, args))


# A waiting prompt must be woken, or the session only ends on the next keypress.
@pytest.mark.parametrize("prompt_running", [True, False])
def test_a_waiting_prompt_is_woken_to_exit(monkeypatch, prompt_running):
    ipython = _TerminalShell()
    app = _PromptApp(prompt_running)
    monkeypatch.setattr(ipython, "pt_app", SimpleNamespace(app=app))
    loop = _Loop()
    monkeypatch.setattr("IPython.core.getipython.get_ipython", lambda: ipython)
    monkeypatch.setattr(session_server.asyncio, "get_running_loop", lambda: loop)

    session_server._end_ipython_session()

    assert ipython.asked == [True]
    expected = [(app.exit, ("",))] if prompt_running else []
    assert loop.scheduled == expected


# PR 101 review (Jakub): `hex-gui --keep-ipython` leaves the session open on close.
@pytest.mark.parametrize(("argv", "end"), [([], True), (["--keep-ipython"], False)])
def test_hex_gui_ends_ipython_on_close_unless_asked_to_keep_it(monkeypatch, argv, end):
    for var in (
        "SESSION_MANAGER",
        "PYTHONPATH",
        "PYTHONUSERBASE",
        "MPLBACKEND",
        "BEAMLINE_ACRONYM",
    ):
        monkeypatch.delenv(var, raising=False)
    configs = []
    monkeypatch.setattr(
        IPython, "start_ipython", lambda argv, config: configs.append(config)
    )
    monkeypatch.setattr(sys, "argv", ["hex-gui", *argv])

    gui_main.main()

    (config,) = configs
    assert (
        f"start_gui('hextools.gui_config:HEX_GUI', end_session_on_close={end});"
    ) in config.InteractiveShellApp.exec_lines
