import atexit
import os
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
from qtpy.QtCore import QCoreApplication, QEvent  # noqa: E402
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


def _open_window(RE):
    if QApplication.instance() is None:
        QApplication([])
    InteractiveShell.instance()
    return QtDataAcquisitionWindow(re_client=RE)


def _assert_torn_down(RE, det, before):
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QCoreApplication.processEvents()
    assert _hooks(RE) == before, "the closed window left its hooks on the RunEngine"
    # The RunEngine outlives the window in IPython and must still run plans.
    RE(count([det], 1))
    assert RE.state == "idle"


# Closing the window ran none of the GUI's teardown: RE kept hooks into deleted
# widgets, so the next RE(...) raised "Signal source has been deleted".
def test_closing_the_window_tears_down(RE, det):
    before = _hooks(RE)
    window = _open_window(RE)
    assert _hooks(RE) != before
    # Closing one deletes its viewport mid-teardown, as the camera viewers do.
    layout = window.qt_widget.layout()
    assert layout is not None
    layout.addWidget(pg.GraphicsLayoutWidget())

    window._qt_window.close()

    _assert_torn_down(RE, det, before)


def test_interpreter_exit_tears_down_an_open_window(RE, det, monkeypatch):
    registered = []
    monkeypatch.setattr(atexit, "register", lambda f, *a, **k: registered.append(f))
    monkeypatch.setattr(atexit, "unregister", lambda f: None)
    before = _hooks(RE)
    _open_window(RE)

    for exit_hook in registered:
        exit_hook()

    _assert_torn_down(RE, det, before)


# PR 97 review: a closed window's hooks sit inside a newer window's chain when the
# older window closes first; they must stop calling into the deleted window.
@pytest.mark.parametrize("first", ["older", "newer"])
def test_closing_windows_that_share_a_run_engine(RE, det, first):
    if QApplication.instance() is None:
        QApplication([])
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
    if QApplication.instance() is None:
        QApplication([])
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
    app = _PromptApp(prompt_running)
    monkeypatch.setattr(ipython, "pt_app", SimpleNamespace(app=app))
    loop = _Loop()
    monkeypatch.setattr(gui_main.IPython, "get_ipython", lambda: ipython)
    monkeypatch.setattr(gui_main.asyncio, "get_running_loop", lambda: loop)

    gui_main._end_ipython_session()

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
    assert f"launch_local_viewer(end_ipython_on_close={end})" in (
        config.InteractiveShellApp.exec_lines
    )
