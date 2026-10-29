import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from bluesky import RunEngine
from bluesky import plan_stubs as bps
from bluesky.utils import RunEngineInterrupted
from ophyd_async.core import init_devices
from qtpy.QtWidgets import QApplication

from hextools.gui.shutter_status import QtShutterStatus
from hextools.photon_delivery_system import Shutter


@pytest.mark.parametrize(
    "error", [None, RuntimeError("failed"), RunEngineInterrupted()]
)
def test_shutter_buttons_exclude_duplicate_and_counterpart_submissions(
    monkeypatch, error
):
    app = QApplication.instance() or QApplication([])
    re = RunEngine({})
    with init_devices(mock=True):
        fe = Shutter("FE:", name="fe")
        photon = Shutter("PH:", name="photon")
    widget = QtShutterStatus(re, {"fe": fe, "photon": photon})
    calls = []

    def execute(code):
        calls.append(code)
        assert not any(b.isEnabled() for b in widget._buttons)
        widget._toggle(widget._buttons[1])
        return error

    monkeypatch.setattr("hextools.gui.shutter_status.run_in_ipython", execute)
    monkeypatch.setattr("hextools.gui.shutter_status.show_plan_error", lambda *a: None)
    try:
        widget._toggle(widget._buttons[0])
        assert not any(b.isEnabled() for b in widget._buttons)
        widget._toggle(widget._buttons[1])
        app.processEvents()
        assert len(calls) == 1
        assert all(b.isEnabled() for b in widget._buttons)
    finally:
        widget.close()
        app.processEvents()


@pytest.mark.parametrize("queued", [False, True])
def test_busy_or_paused_engine_blocks_shutters_until_abort(monkeypatch, queued):
    app = QApplication.instance() or QApplication([])
    re = RunEngine({})
    with init_devices(mock=True):
        fe = Shutter("FE:", name="fe")
        photon = Shutter("PH:", name="photon")
    widget = QtShutterStatus(re, {"fe": fe, "photon": photon})
    calls = []
    monkeypatch.setattr(
        "hextools.gui.shutter_status.run_in_ipython", lambda code: calls.append(code)
    )
    monkeypatch.setattr(
        "hextools.gui.shutter_status.QMessageBox.warning", lambda *args: None
    )
    try:
        if queued:
            widget._toggle(widget._buttons[0])
        with pytest.raises(RunEngineInterrupted):
            re(bps.pause())
        widget._update_enabled()
        assert not any(b.isEnabled() for b in widget._buttons)
        if not queued:
            widget._toggle(widget._buttons[1])
        app.processEvents()
        assert calls == []
        assert not any(b.isEnabled() for b in widget._buttons)
        re.abort()
        widget._update_enabled()
        assert all(b.isEnabled() for b in widget._buttons)
    finally:
        widget.close()
        app.processEvents()
