import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("pyqtgraph")

from bluesky import RunEngine  # noqa: E402
from IPython.core.interactiveshell import InteractiveShell  # noqa: E402
from qtpy.QtWidgets import QApplication, QTabWidget  # noqa: E402

from hextools.gui.__main__ import QtXRDView  # noqa: E402


def test_xrd_tab_offers_the_xrd_plans(RE: RunEngine):
    _app = QApplication.instance() or QApplication([])
    # the plan widgets validate their arguments in the IPython session
    InteractiveShell.instance()
    view = QtXRDView(RE)
    plan_tabs = [
        tabs
        for tabs in view.findChildren(QTabWidget)
        if tabs.count() and tabs.tabText(0) == "Count"
    ]
    assert len(plan_tabs) == 1
    labels = [plan_tabs[0].tabText(i) for i in range(plan_tabs[0].count())]
    assert labels == ["Count", "Scan", "Dark", "Calibration"]
    view.deleteLater()
