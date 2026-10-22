import atexit
import os
import signal
import sys
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

import IPython  # noqa: E402
import pyqtgraph as pg  # noqa: E402
from bluesky.plans import count  # noqa: E402
from IPython.core.interactiveshell import InteractiveShell  # noqa: E402
from IPython.terminal.interactiveshell import TerminalInteractiveShell  # noqa: E402
from ophyd_async.core import init_devices  # noqa: E402
from ophyd_async.sim import PatternGenerator, SimPointDetector  # noqa: E402
from qtpy.QtCore import QCoreApplication, QEvent, QProcess  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

import hextools.gui.__main__ as gui_main  # noqa: E402
from hextools.gui.__main__ import QtDataAcquisitionWindow  # noqa: E402


@pytest.fixture
def det(RE):
    with init_devices():
        det = SimPointDetector(PatternGenerator())
    return det


def _hooks(RE):
    callbacks = sum(len(cbs) for cbs in RE.dispatcher.cb_registry.callbacks.values())
    return RE.state_hook, RE.msg_hook, RE.waiting_hook, callbacks


def _open_window(RE, tmp_path, request):
    """Build the GUI on ``RE`` with the auto exporter running as a stand-in child.

    The child records a SIGTERM to ``tmp_path / "sigterm"``, so a graceful stop
    is distinguishable from the kill QProcess's destructor does.
    """
    QApplication.instance() or QApplication([])
    InteractiveShell.instance()
    window = QtDataAcquisitionWindow(re_client=RE)
    exporter = window.qt_widget.auto_exporter
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
    return window, exporter, marker


def _assert_torn_down(RE, det, before, marker):
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QCoreApplication.processEvents()
    assert marker.exists(), "auto exporter was killed, not stopped"
    assert _hooks(RE) == before, "the closed window left its hooks on the RunEngine"
    # The RunEngine outlives the window in IPython and must still run plans.
    RE(count([det], 1))
    assert RE.state == "idle"


# Closing the window ran none of the GUI's teardown: the exporter child was killed
# by QProcess's destructor, and RE kept hooks into deleted widgets, so the next
# RE(...) raised "Signal source has been deleted".
def test_closing_the_window_tears_down(RE, det, tmp_path, request):
    before = _hooks(RE)
    window, _, marker = _open_window(RE, tmp_path, request)
    assert _hooks(RE) != before
    # Closing one deletes its viewport mid-teardown, as the camera viewers do.
    window.qt_widget.layout().addWidget(pg.GraphicsLayoutWidget())

    window._qt_window.close()

    _assert_torn_down(RE, det, before, marker)


def test_interpreter_exit_tears_down_an_open_window(
    RE, det, tmp_path, request, monkeypatch
):
    registered = []
    monkeypatch.setattr(atexit, "register", lambda f, *a, **k: registered.append(f))
    monkeypatch.setattr(atexit, "unregister", lambda f: None)
    before = _hooks(RE)
    _, _, marker = _open_window(RE, tmp_path, request)

    for exit_hook in registered:
        exit_hook()

    _assert_torn_down(RE, det, before, marker)


# PR 97 review: a closed window's hooks sit inside a newer window's chain when the
# older window closes first; they must stop calling into the deleted window.
@pytest.mark.parametrize("first", ["older", "newer"])
def test_closing_windows_that_share_a_run_engine(RE, det, first):
    QApplication.instance() or QApplication([])
    InteractiveShell.instance()
    before = _hooks(RE)
    older = QtDataAcquisitionWindow(re_client=RE)
    newer = QtDataAcquisitionWindow(re_client=RE)

    for window in [older, newer] if first == "older" else [newer, older]:
        window._qt_window.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        QCoreApplication.processEvents()
        RE(count([det], 1))
        assert RE.state == "idle"

    assert _hooks(RE) == before


# Nghia (2026-10-21): closing the GUI started with `hex-gui` left its IPython
# session running underneath.
@pytest.mark.parametrize("end_ipython", [False, True])
def test_closing_the_window_can_end_the_ipython_session(RE, monkeypatch, end_ipython):
    QApplication.instance() or QApplication([])
    InteractiveShell.instance()
    ended = []
    monkeypatch.setattr(gui_main, "_end_ipython_session", lambda: ended.append(True))
    window = QtDataAcquisitionWindow(re_client=RE, end_ipython_on_close=end_ipython)

    window._qt_window.close()

    assert ended == ([True] if end_ipython else [])


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
    monkeypatch.setattr(gui_main.IPython, "get_ipython", lambda: ipython)

    gui_main._end_ipython_session()

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
    ipython.pt_app = SimpleNamespace(app=_PromptApp(prompt_running))
    loop = _Loop()
    monkeypatch.setattr(gui_main.IPython, "get_ipython", lambda: ipython)
    monkeypatch.setattr(gui_main.asyncio, "get_running_loop", lambda: loop)

    gui_main._end_ipython_session()

    assert ipython.asked == [True]
    expected = [(ipython.pt_app.app.exit, ("",))] if prompt_running else []
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
    assert f"launch_local_viewer(end_ipython_on_close={end})" in (
        config.InteractiveShellApp.exec_lines
    )
