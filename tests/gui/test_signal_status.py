import os
from typing import cast

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from concurrent.futures import Future  # noqa: E402

from bluesky import RunEngine  # noqa: E402
from bluesky.run_engine import call_in_bluesky_event_loop  # noqa: E402
from ophyd_async.core import (  # noqa: E402
    Device,
    DeviceMap,
    DeviceVector,
    init_devices,
    soft_signal_rw,
)
from ophyd_async.sim import SimMotor  # noqa: E402
from qtpy.QtCore import QCoreApplication, QEvent  # noqa: E402
from qtpy.QtWidgets import QApplication, QLabel, QWidget  # noqa: E402

from hextools.gui.run_engine_clients import (  # noqa: E402
    InProcessRunEngineClient,
    RunEngineClient,
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
    client = InProcessRunEngineClient({"RE": RE, name: obj})
    return QtSignalStatus(client, name, **kwargs)


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


class _FakeSource:
    def __init__(self):
        self.subscribed: list[str] = []
        self.unsubscribed: list[object] = []
        self.units: Future[str | None] = Future()

    def subscribe_reading(self, name, callback):
        self.subscribed.append(name)
        return name

    def unsubscribe_reading(self, token):
        self.unsubscribed.append(token)

    def fetch_units(self, name):
        return self.units


def test_signal_status_subscribes_by_name_and_unsubscribes_once():
    if QApplication.instance() is None:
        QApplication([])
    source = _FakeSource()
    widget = QtSignalStatus(cast(RunEngineClient, source), "fake.signal")
    widget.close()
    widget.close()
    assert source.subscribed == ["fake.signal"]
    assert source.unsubscribed == ["fake.signal"]


def test_units_are_none_until_fetched():
    app = QApplication.instance() or QApplication([])
    source = _FakeSource()
    widget = QtSignalStatus(cast(RunEngineClient, source), "fake.signal")
    value = cast(QLabel, widget._value)
    try:
        widget._signal_reading.emit({"value": 2, "timestamp": 0.0})
        app.processEvents()
        assert value.text() == "2"
        source.units.set_result("mm")
        app.processEvents()
        assert value.text() == "2 mm"
    finally:
        widget.close()


def test_signal_status_deleted_without_close_unsubscribes_and_drops_updates():
    if QApplication.instance() is None:
        QApplication([])
    callbacks = []

    class _Source(_FakeSource):
        def subscribe_reading(self, name, callback):
            callbacks.append(callback)
            return super().subscribe_reading(name, callback)

    source = _Source()
    parent = QWidget()
    QtSignalStatus(cast(RunEngineClient, source), "fake.signal", parent=parent)
    parent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert source.unsubscribed == ["fake.signal"]
    (reading,) = callbacks
    reading({"value": 1, "timestamp": 0.0})
    source.units.set_result("mm")


def test_client_reads_devices_by_registry_path_only():
    app = QApplication.instance() or QApplication([])
    RE = RunEngine()
    with init_devices(mock=True):
        motor = SimMotor(name="motor")
    client = InProcessRunEngineClient({"RE": RE, "motor": motor, "alias": motor})
    for name in ("motor.user_readback", "alias.user_readback"):
        widget = QtSignalStatus(client, name)
        try:
            _wait_for_text(app, widget, "0.000 mm")
        finally:
            widget.close()
    for bad in (
        "missing",
        "motor.nope",
        "motor._connector",
        "motor; import os",
        "motor.user_readback.parent",
    ):
        with pytest.raises(KeyError):
            client.subscribe_reading(bad, print)


class _Axes(Device):
    def __init__(self, name=""):
        self.vector = DeviceVector({1: soft_signal_rw(float)})
        self.map = DeviceMap({"x": soft_signal_rw(float)})
        super().__init__(name=name)


def test_client_reads_indexed_children():
    RE = RunEngine()
    axes = _Axes(name="axes")
    client = InProcessRunEngineClient({"RE": RE, "axes": axes})
    for name in ("axes.vector[1]", "axes.map['x']"):
        client.unsubscribe_reading(client.subscribe_reading(name, print))
    for bad in ("axes.vector[2]", "axes.vector[__import__('os')]"):
        with pytest.raises(KeyError):
            client.subscribe_reading(bad, print)
