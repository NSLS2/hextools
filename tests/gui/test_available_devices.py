import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from bluesky import RunEngine  # noqa: E402
from ophyd_async.core import init_devices  # noqa: E402
from ophyd_async.sim import SimMotor  # noqa: E402
from qtpy.QtWidgets import QApplication, QLabel  # noqa: E402

from hextools.catalog.model import DeviceRegistry  # noqa: E402
from hextools.gui.available_devices import QtAvailableDevices  # noqa: E402
from hextools.gui.run_engine_clients import InProcessRunEngineClient  # noqa: E402


def _rows(widget: QtAvailableDevices) -> dict[str, str]:
    return {label: status.text() for label, status in widget._status_labels.items()}


def test_registry_reports_devices_with_aliases():
    RunEngine()
    with init_devices(mock=True):
        motor = SimMotor(name="motor")
    registry = DeviceRegistry.from_namespace(
        {"motor": motor, "m": motor, "_hidden": motor}
    )
    assert list(registry) == ["motor"]
    node = registry["motor"]
    assert node.aliases == ("m",)
    assert node.classes[0].rpartition(".")[2] == "SimMotor"
    assert node.connected


def test_widget_lists_registry_devices():
    app = QApplication.instance() or QApplication([])
    RE = RunEngine()
    with init_devices(mock=True):
        motor = SimMotor(name="motor")
    client = InProcessRunEngineClient({"RE": RE, "motor": motor, "m": motor})
    widget = QtAvailableDevices(client)
    try:
        assert _rows(widget) == {"motor (m)": "Connected"}
        types = [w.text() for w in widget.findChildren(QLabel)]
        assert "SimMotor" in types
    finally:
        widget.close()
        app.processEvents()


def test_widget_follows_catalog_changes():
    app = QApplication.instance() or QApplication([])
    RE = RunEngine()
    with init_devices(mock=True):
        motor = SimMotor(name="motor")
        slits = SimMotor(name="slits")
    namespace = {"RE": RE, "motor": motor}
    client = InProcessRunEngineClient(namespace)
    widget = QtAvailableDevices(client)
    try:
        assert _rows(widget) == {"motor": "Connected"}
        namespace["slits"] = slits
        client.refresh_catalog()
        app.processEvents()
        assert _rows(widget) == {"motor": "Connected", "slits": "Connected"}
    finally:
        widget.close()
    assert client._catalog_callbacks == {}
