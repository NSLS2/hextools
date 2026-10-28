import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

from bluesky import RunEngine  # noqa: E402
from bluesky_widgets.models.run_engine_client import RunEngineClient  # noqa: E402
from IPython.core.interactiveshell import InteractiveShell  # noqa: E402
from qtpy.QtWidgets import QApplication, QLineEdit  # noqa: E402

from hextools.gui.available_devices import QtAvailableDevices  # noqa: E402
from hextools.gui.device_selector_widget import (  # noqa: E402
    DynamicDeviceSelector,
    QtDeviceSelector,
)
from hextools.gui.plan_widget import QtPlanWidget  # noqa: E402
from hextools.motors import SampleSmall, move_motor  # noqa: E402


@pytest.mark.parametrize("remote_form", [False, True])
def test_sample_small_labels_preserve_device_identifiers(monkeypatch, remote_form):
    app = QApplication.instance() or QApplication([])
    namespace = {"sample_small": SampleSmall("TEST:", name="sample-small")}
    shell = InteractiveShell.instance()
    for name, motor in namespace.items():
        monkeypatch.setitem(shell.user_ns, name, motor)
    re = RunEngine({}, call_returns_result=True)
    model = RunEngineClient()
    model._allowed_devices = {
        "sample_small": {
            "classname": "SampleSmall",
            "components": {
                axis: {"classname": "Motor", "is_movable": True} for axis in ("x", "y")
            },
        }
    }
    dynamic = DynamicDeviceSelector(["sample_small.x", "sample_small.y"])
    single = QtDeviceSelector(model, device_type="movable", include_components=True)
    form = QtPlanWidget(model if remote_form else re, move_motor)
    available = QtAvailableDevices(re, namespace=namespace)
    remote = QtAvailableDevices(model, [("sample_small", SampleSmall)])
    widgets = [dynamic, single, form, available, remote]
    try:
        for name, _, widget, _ in form._scalar_fields:
            if name == "value":
                assert isinstance(widget, QLineEdit)
                widget.setText("1")
        combo = next(
            field["widget"] for field in form._device_fields if field["name"] == "motor"
        )
        for axis in ("x", "y"):
            name, label = f"sample_small.{axis}", f"Sample Small {axis.upper()}"
            for selector in (combo, single._combo, dynamic._rows[0]["combo"]):
                index = selector.findData(name)
                assert index >= 0
                assert selector.itemText(index) == label
                selector.setCurrentIndex(index)
                assert selector.currentData() == name
            kwargs, device_keys = form._collect_inputs()
            assert kwargs["motor"] == name
            assert "motor" in device_keys
        assert "Sample Small" in available._status_labels
        assert remote._status_labels["Sample Small"].text() == "Allowed"
    finally:
        for widget in widgets:
            widget.close()
            widget.deleteLater()
        app.processEvents()
