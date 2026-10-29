import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from bluesky.plans import count  # noqa: E402
from ophyd_async.core import init_devices  # noqa: E402
from ophyd_async.sim import PatternGenerator, SimPointDetector  # noqa: E402
from qtpy.QtWidgets import QApplication, QMessageBox  # noqa: E402

from hextools.gui.plan_status import PlanMonitor, QtPlanHistory  # noqa: E402
from hextools.utils.nl_msg_hook import MsgHookNarrator  # noqa: E402


@pytest.fixture
def det(RE):
    with init_devices():
        det = SimPointDetector(PatternGenerator())
    return det


def _history(RE, path):
    if QApplication.instance() is None:
        QApplication([])
    monitor = PlanMonitor(
        RE, narrator=MsgHookNarrator(emit=lambda _: None), namespace={"count": count}
    )
    return monitor, QtPlanHistory(monitor, history_file=path)


def _finish_plan(RE, monitor, det):
    RE(count([det]))
    QApplication.processEvents()


def test_history_is_saved_and_cleared(RE, det, tmp_path, monkeypatch):
    path = tmp_path / "cache" / "history.json"
    monitor, history = _history(RE, path)
    assert not history._clear_button.isEnabled()

    _finish_plan(RE, monitor, det)
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


def test_history_is_loaded_on_startup(RE, det, tmp_path):
    path = tmp_path / "history.json"
    monitor, _ = _history(RE, path)
    _finish_plan(RE, monitor, det)
    monitor.detach()

    monitor, history = _history(RE, path)
    assert history._table.rowCount() == 1
    monitor.detach()


@pytest.mark.parametrize(
    ("mock", "name"), [("NO", "plan_history.json"), ("YES", "plan_history_mock.json")]
)
@pytest.mark.parametrize("xdg_state_home", [None, "/xdg/state"])
def test_default_history_file(monkeypatch, tmp_path, mock, name, xdg_state_home):
    from hextools.gui.__main__ import default_plan_history_file

    monkeypatch.setenv("HEXTOOLS_RUNNING_IN_CI", mock)
    monkeypatch.setenv("HOME", str(tmp_path))
    if xdg_state_home is None:
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        state_home = tmp_path / ".local" / "state"
    else:
        monkeypatch.setenv("XDG_STATE_HOME", xdg_state_home)
        state_home = Path(xdg_state_home)
    assert default_plan_history_file() == state_home / "hextools" / name
