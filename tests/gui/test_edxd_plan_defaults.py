import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import inspect

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

from bluesky import RunEngine
from IPython.core.interactiveshell import InteractiveShell
from qtpy.QtWidgets import QApplication, QCheckBox

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
