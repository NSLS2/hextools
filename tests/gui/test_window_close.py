import atexit
import os
import signal
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

import pyqtgraph as pg  # noqa: E402
from bluesky.plans import count  # noqa: E402
from IPython.core.interactiveshell import InteractiveShell  # noqa: E402
from ophyd.sim import det  # noqa: E402  # ty: ignore[unresolved-import]
from qtpy.QtCore import QCoreApplication, QEvent, QProcess  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from hextools.gui.__main__ import QtDataAcquisitionWindow  # noqa: E402


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
    exporter._process.start = lambda *_: start(sys.executable, ["-c", child])  # ty: ignore[invalid-assignment]
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


def _assert_torn_down(RE, before, marker):
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
def test_closing_the_window_tears_down(RE, tmp_path, request):
    before = _hooks(RE)
    window, _, marker = _open_window(RE, tmp_path, request)
    assert _hooks(RE) != before
    # Closing one deletes its viewport mid-teardown, as the camera viewers do.
    window.qt_widget.layout().addWidget(pg.GraphicsLayoutWidget())

    window._qt_window.close()

    _assert_torn_down(RE, before, marker)


def test_interpreter_exit_tears_down_an_open_window(RE, tmp_path, request, monkeypatch):
    registered = []
    monkeypatch.setattr(atexit, "register", lambda f, *a, **k: registered.append(f))
    monkeypatch.setattr(atexit, "unregister", lambda f: None)
    before = _hooks(RE)
    _, _, marker = _open_window(RE, tmp_path, request)

    for exit_hook in registered:
        exit_hook()

    _assert_torn_down(RE, before, marker)


# PR 97 review: a closed window's hooks sit inside a newer window's chain when the
# older window closes first; they must stop calling into the deleted window.
@pytest.mark.parametrize("first", ["older", "newer"])
def test_closing_windows_that_share_a_run_engine(RE, first):
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
