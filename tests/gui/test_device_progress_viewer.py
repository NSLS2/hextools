import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

from bluesky_widgets.models.run_engine_client import RunEngineClient  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from hextools.gui.device_progress_viewer import QtReWaitingHookMonitor  # noqa: E402


def test_qserver_without_progress_api_shows_a_limitation():
    app = QApplication.instance() or QApplication([])
    monitor = QtReWaitingHookMonitor(RunEngineClient())
    try:
        assert (
            monitor._lb_status.text()
            == "QueueServer plan progress unavailable in this bluesky-widgets version."
        )
    finally:
        monitor.close()
        monitor.deleteLater()
        app.processEvents()
