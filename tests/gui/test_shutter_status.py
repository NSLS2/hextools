import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

from bluesky import RunEngine
from bluesky import plan_stubs as bps
from bluesky.utils import RunEngineInterrupted
from ophyd_async.core import init_devices
from qtpy.QtWidgets import QApplication

from hextools.gui.hex_config import shutter_controls
from hextools.gui.shutter_status import QtShutterStatus
from hextools.gui.signal_status import QtSignalStatus
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
    widget = QtShutterStatus(re, *shutter_controls({"fe": fe, "photon": photon}))
    calls = []

    def execute(code):
        calls.append(code)
        assert "sleep_time=0.0" in code
        assert not any(i.button.isEnabled() for i in widget._indicators)
        widget._toggle(widget._indicators[1])
        return error

    monkeypatch.setattr("hextools.gui.shutter_status.run_in_ipython", execute)
    monkeypatch.setattr("hextools.gui.shutter_status.show_plan_error", lambda *a: None)
    try:
        widget._toggle(widget._indicators[0])
        assert not any(i.button.isEnabled() for i in widget._indicators)
        widget._toggle(widget._indicators[1])
        app.processEvents()
        assert len(calls) == 1
        assert all(i.button.isEnabled() for i in widget._indicators)
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
    widget = QtShutterStatus(re, *shutter_controls({"fe": fe, "photon": photon}))
    calls = []
    monkeypatch.setattr(
        "hextools.gui.shutter_status.run_in_ipython", lambda code: calls.append(code)
    )
    try:
        indicators = widget._indicators
        assert indicators[0].button.toolTip() == "Click to open the fe."
        if queued:
            widget._toggle(indicators[0])
        with pytest.raises(RunEngineInterrupted):
            re(bps.pause())
        widget._update_enabled()
        assert not any(i.button.isEnabled() for i in indicators)
        assert all("not idle" in i.button.toolTip() for i in indicators)
        if not queued:
            widget._toggle(indicators[1])
        app.processEvents()
        assert calls == []
        assert not any(i.button.isEnabled() for i in indicators)
        re.abort()
        widget._update_enabled()
        assert all(i.button.isEnabled() for i in indicators)
    finally:
        widget.close()
        app.processEvents()


@pytest.mark.parametrize(
    "value, severity, is_open",
    [(True, 0, True), (False, 0, False), (True, 3, None)],
)
def test_indicator_follows_readings_and_shows_invalid_as_unknown(
    value, severity, is_open
):
    app = QApplication.instance() or QApplication([])
    re = RunEngine({})
    with init_devices(mock=True):
        fe = Shutter("FE:", name="fe")
    widget = QtShutterStatus(re, *shutter_controls({"fe": fe}))
    try:
        (indicator,) = widget._indicators
        assert isinstance(indicator, QtSignalStatus)
        indicator._set_reading(
            {"value": value, "timestamp": 0.0, "alarm_severity": severity}
        )
        assert indicator.is_open is is_open
    finally:
        widget.close()
        app.processEvents()
