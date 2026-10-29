import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from bluesky import RunEngine
from IPython.core.interactiveshell import InteractiveShell
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit

from hextools.edxd import edxd_2theta_tilt
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
