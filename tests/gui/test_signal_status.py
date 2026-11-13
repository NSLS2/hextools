import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from bluesky import RunEngine  # noqa: E402
from bluesky.run_engine import call_in_bluesky_event_loop  # noqa: E402
from ophyd_async.core import init_devices, soft_signal_rw  # noqa: E402
from ophyd_async.sim import SimMotor  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from hextools.gui.reading_sources import (  # noqa: E402
    NamespaceReadingSource,
    ReadingSource,
)
from hextools.gui.signal_status import QtSignalStatus, format_value  # noqa: E402


@pytest.mark.parametrize(
    "value, expected",
    [(True, "True"), (3, "3"), (1.23456, "1.235"), ("open", "open")],
)
def test_format_value(value, expected):
    assert format_value(value) == expected


def test_format_value_rejects_other_types():
    with pytest.raises(TypeError):
        format_value([1, 2])


def _wait_for_text(app, widget, text):
    for _ in range(200):
        app.processEvents()
        if widget._value.text() == text:
            return
        call_in_bluesky_event_loop(_noop())
    assert widget._value.text() == text


async def _noop():
    pass


async def _set(signal, value):
    await signal.set(value)


def _status(RE, name, obj, **kwargs) -> QtSignalStatus:
    return QtSignalStatus(name, NamespaceReadingSource({name: obj}, RE.loop), **kwargs)


def test_signal_status_tracks_signal_and_defaults_label_to_name():
    app = QApplication.instance() or QApplication([])
    RE = RunEngine()
    signal = soft_signal_rw(float, initial_value=1.5, name="temperature")
    widget = _status(RE, "temperature", signal)
    try:
        assert widget._label.text() == "temperature:"
        _wait_for_text(app, widget, "1.500")
        call_in_bluesky_event_loop(_set(signal, 2.25))
        _wait_for_text(app, widget, "2.250")
    finally:
        widget.close()


def test_signal_status_shows_device_reading_with_custom_label():
    app = QApplication.instance() or QApplication([])
    RE = RunEngine()
    with init_devices():
        motor = SimMotor(name="sim_x")
    widget = _status(RE, "sim_x", motor, label="X")
    try:
        assert widget._label.text() == "X:"
        _wait_for_text(app, widget, "0.000 mm")
    finally:
        widget.close()


def test_signal_status_shows_units():
    app = QApplication.instance() or QApplication([])
    RE = RunEngine()
    signal = soft_signal_rw(float, initial_value=4.0, units="keV", name="energy")
    widget = _status(RE, "energy", signal, precision=1)
    try:
        _wait_for_text(app, widget, "4.0 keV")
    finally:
        widget.close()


@pytest.mark.parametrize("severity, colored", [(0, False), (1, True), (2, True)])
def test_signal_status_colors_by_alarm_severity(severity, colored):
    app = QApplication.instance() or QApplication([])
    RE = RunEngine()
    signal = soft_signal_rw(int, name="counter")
    widget = _status(RE, "counter", signal)
    try:
        _wait_for_text(app, widget, "0")
        widget._signal_reading.emit(
            {"value": 7, "timestamp": 0.0, "alarm_severity": severity}
        )
        _wait_for_text(app, widget, "7")
        assert bool(widget._value.styleSheet()) is colored
    finally:
        widget.close()


class _FakeSource(ReadingSource):
    def __init__(self):
        self.subscribed: list[str] = []
        self.unsubscribed: list[object] = []

    def lookup(self, name):
        return object()

    def subscribe(self, name, callback):
        self.subscribed.append(name)
        return name

    def unsubscribe(self, token):
        self.unsubscribed.append(token)

    def fetch_units(self, name, callback):
        pass


def test_signal_status_subscribes_by_name_and_unsubscribes_once():
    if QApplication.instance() is None:
        QApplication([])
    source = _FakeSource()
    widget = QtSignalStatus("fake.signal", source)
    widget.close()
    widget.close()
    assert source.subscribed == ["fake.signal"]
    assert source.unsubscribed == ["fake.signal"]


def test_namespace_reading_source_looks_up_dotted_paths_only():
    RE = RunEngine()
    with init_devices(mock=True):
        motor = SimMotor(name="motor")
    source = NamespaceReadingSource({"motor": motor}, RE.loop)
    assert source.lookup("motor.user_readback") is motor.user_readback
    for bad in ("missing", "motor.nope", "motor._connector", "motor; import os"):
        with pytest.raises(KeyError):
            source.lookup(bad)
