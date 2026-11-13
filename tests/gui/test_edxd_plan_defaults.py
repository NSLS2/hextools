import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import inspect

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

from bluesky import RunEngine
from IPython.core.interactiveshell import InteractiveShell
from qtpy.QtWidgets import QApplication, QCheckBox, QLineEdit

from hextools.edxd import edxd_custom_pos_list_grid, edxd_scan
from hextools.gui.plan_widget import QtPlanWidget


@pytest.mark.parametrize("plan", [edxd_scan, edxd_custom_pos_list_grid])
def test_tiff_checkbox_follows_plan_default(plan):
    app = QApplication.instance() or QApplication([])
    InteractiveShell.instance()
    widget = QtPlanWidget(RunEngine({}), plan)
    try:
        default = inspect.signature(plan).parameters["save_as_tiff"].default
        assert default is True
        checkbox = next(
            w for name, _, w, _ in widget._scalar_fields if name == "save_as_tiff"
        )
        assert isinstance(checkbox, QCheckBox)
        assert checkbox.isChecked() is default
        checkbox.setChecked(False)
        assert not checkbox.isChecked()
    finally:
        widget.close()
        app.processEvents()


@pytest.mark.parametrize("plan", [edxd_scan, edxd_custom_pos_list_grid])
def test_iteration_fields_offered_with_defaults(plan):
    app = QApplication.instance() or QApplication([])
    InteractiveShell.instance()
    widget = QtPlanWidget(RunEngine({}), plan)
    try:
        fields = {name: (kind, w) for name, kind, w, _ in widget._scalar_fields}
        iterations_kind, iterations = fields["num_iterations"]
        delay_kind, delay = fields["time_between_iterations"]
        assert isinstance(iterations, QLineEdit)
        assert isinstance(delay, QLineEdit)
        assert iterations_kind is int
        assert iterations.text() == "1"
        assert delay_kind is float
        assert delay.text() == "0.0"
    finally:
        widget.close()
        app.processEvents()
