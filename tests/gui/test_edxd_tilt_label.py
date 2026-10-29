import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from bluesky import RunEngine
from IPython.core.interactiveshell import InteractiveShell
from qtpy.QtWidgets import QApplication, QCheckBox, QLabel, QLineEdit

from hextools.edxd import edxd_2theta_tilt, edxd_custom_pos_list_grid, edxd_scan
from hextools.gui.plan_widget import QtPlanWidget


def test_two_theta_label_preserves_theta_keyword():
    app = QApplication.instance() or QApplication([])
    InteractiveShell.instance()
    widget = QtPlanWidget(RunEngine({}), edxd_2theta_tilt)
    try:
        assert "Two theta" in [label.text() for label in widget.findChildren(QLabel)]
        theta = next(w for n, _, w, _ in widget._scalar_fields if n == "theta")
        assert isinstance(theta, QLineEdit)
        theta.setText("3.0")
        kwargs, device_keys = widget._collect_inputs()
        assert kwargs["theta"] == 3.0
        assert "theta=3.0" in widget._format_plan_call(kwargs, device_keys)
    finally:
        widget.close()
        app.processEvents()


@pytest.mark.parametrize("plan", [edxd_scan, edxd_custom_pos_list_grid])
def test_edxd_tiff_defaults_checked_and_can_be_disabled(plan):
    app = QApplication.instance() or QApplication([])
    InteractiveShell.instance()
    widget = QtPlanWidget(RunEngine({}), plan)
    try:
        checkbox = next(
            w for n, _, w, _ in widget._scalar_fields if n == "save_as_tiff"
        )
        assert isinstance(checkbox, QCheckBox)
        assert checkbox.isChecked()
        for _name, _kind, field, required in widget._scalar_fields:
            if required and isinstance(field, QLineEdit):
                field.setText("1")
        for _name, _kind, field, required in widget._scalar_list_fields:
            if required:
                assert isinstance(field, QLineEdit)
                field.setText("1")
        # Isolate the scalar form from mandatory motor selection.
        widget._device_fields = []
        assert widget._collect_inputs()[0]["save_as_tiff"] is True
        checkbox.setChecked(False)
        assert widget._collect_inputs()[0]["save_as_tiff"] is False
    finally:
        widget.close()
        app.processEvents()
