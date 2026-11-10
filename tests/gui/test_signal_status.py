import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from bluesky import RunEngine  # noqa: E402
from bluesky.run_engine import call_in_bluesky_event_loop  # noqa: E402
from ophyd_async.core import init_devices, soft_signal_rw  # noqa: E402
from ophyd_async.sim import SimMotor  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

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


def test_signal_status_tracks_signal_and_defaults_label_to_name():
    app = QApplication.instance() or QApplication([])
    RE = RunEngine()
    signal = soft_signal_rw(float, initial_value=1.5, name="temperature")
    widget = QtSignalStatus(signal, RE.loop)
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
    widget = QtSignalStatus(motor, RE.loop, label="X")
    try:
        assert widget._label.text() == "X:"
        _wait_for_text(app, widget, "0.000 mm")
    finally:
        widget.close()


def test_signal_status_shows_units():
    app = QApplication.instance() or QApplication([])
    RE = RunEngine()
    signal = soft_signal_rw(float, initial_value=4.0, units="keV", name="energy")
    widget = QtSignalStatus(signal, RE.loop, precision=1)
    try:
        _wait_for_text(app, widget, "4.0 keV")
    finally:
        widget.close()


@pytest.mark.parametrize("severity, colored", [(0, False), (1, True), (2, True)])
def test_signal_status_colors_by_alarm_severity(severity, colored):
    app = QApplication.instance() or QApplication([])
    RE = RunEngine()
    signal = soft_signal_rw(int, name="counter")
    widget = QtSignalStatus(signal, RE.loop)
    try:
        _wait_for_text(app, widget, "0")
        widget._on_reading(
            {"counter": {"value": 7, "timestamp": 0.0, "alarm_severity": severity}}
        )
        _wait_for_text(app, widget, "7")
        assert bool(widget._value.styleSheet()) is colored
    finally:
        widget.close()


class _FakeSubscribable:
    name = "fake"

    def __init__(self):
        self.cleared = []

    def subscribe_reading(self, function):
        pass

    def clear_sub(self, function):
        self.cleared.append(function)


def test_signal_status_unsubscribes_once_when_closed_repeatedly():
    QApplication.instance() or QApplication([])
    RE = RunEngine()
    fake = _FakeSubscribable()
    widget = QtSignalStatus(fake, RE.loop)  # type: ignore[arg-type]
    widget.close()
    widget.close()
    call_in_bluesky_event_loop(_noop())
    assert fake.cleared == [widget._on_reading]
