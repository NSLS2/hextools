import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from bluesky.plans import count  # noqa: E402
from ophyd.sim import det  # noqa: E402  # type: ignore
from qtpy.QtWidgets import QApplication, QMessageBox  # noqa: E402

from hextools.gui.plan_status import PlanMonitor, QtPlanHistory  # noqa: E402
from hextools.utils.nl_msg_hook import MsgHookNarrator  # noqa: E402


def _history(RE, path):
    QApplication.instance() or QApplication([])
    monitor = PlanMonitor(
        RE, narrator=MsgHookNarrator(emit=lambda _: None), namespace={"count": count}
    )
    return monitor, QtPlanHistory(monitor, history_file=path)


def _finish_plan(RE, monitor):
    RE(count([det]))
    QApplication.processEvents()


def test_history_is_saved_and_cleared(RE, tmp_path, monkeypatch):
    path = tmp_path / "cache" / "history.json"
    monitor, history = _history(RE, path)
    assert not history._clear_button.isEnabled()

    _finish_plan(RE, monitor)
    assert len(json.loads(path.read_text())) == 1
    assert history._clear_button.isEnabled()

    # declining the confirmation keeps everything
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_: QMessageBox.StandardButton.No
    )
    history._on_clear_clicked()
    assert history._table.rowCount() == 1

    monkeypatch.setattr(
        QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes
    )
    history._on_clear_clicked()
    assert history._table.rowCount() == 0
    assert json.loads(path.read_text()) == []
    assert not history._clear_button.isEnabled()
    monitor.detach()


def test_history_is_loaded_on_startup(RE, tmp_path):
    path = tmp_path / "history.json"
    monitor, _ = _history(RE, path)
    _finish_plan(RE, monitor)
    monitor.detach()

    monitor, history = _history(RE, path)
    assert history._table.rowCount() == 1
    monitor.detach()
