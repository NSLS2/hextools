import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

import bluesky.plan_stubs as bps  # noqa: E402
from bluesky.plans import count  # noqa: E402
from ophyd_async.core import init_devices  # noqa: E402
from ophyd_async.sim import PatternGenerator, SimPointDetector  # noqa: E402
from qtpy.QtCore import QCoreApplication, QEvent  # noqa: E402
from qtpy.QtWidgets import QApplication, QMessageBox  # noqa: E402

from hextools.gui.plan_status import QtPlanHistory, _scan_id_ranges  # noqa: E402
from hextools.gui.run_engine_clients import (  # noqa: E402
    DeviceName,
    HistoryItem,
    InProcessRunEngineClient,
    PlanResult,
)


def _client(RE, det, path):
    return InProcessRunEngineClient(
        {"RE": RE, "det": det, "count": count}, history_file=path
    )


def _history(RE, det, path):
    if QApplication.instance() is None:
        QApplication([])
    client = _client(RE, det, path)
    return client, QtPlanHistory(client)


@pytest.fixture
def det(RE):
    # init_devices names devices after the variables they are assigned to
    with init_devices():
        det = SimPointDetector(PatternGenerator())
    return det


def _finish_plan(RE, det):
    RE(count([det], num=2))
    QApplication.processEvents()


def test_history_is_saved_and_cleared(RE, det, tmp_path, monkeypatch):
    path = tmp_path / "cache" / "history.json"
    client, history = _history(RE, det, path)
    assert not history._clear_button.isEnabled()

    _finish_plan(RE, det)
    assert len(json.loads(path.read_text())) == 1
    assert history._table.rowCount() == 1
    assert history._clear_button.isEnabled()

    # declining the confirmation keeps everything
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_: QMessageBox.StandardButton.No
    )
    history._on_clear_clicked()
    QApplication.processEvents()
    assert history._table.rowCount() == 1

    monkeypatch.setattr(
        QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes
    )
    history._on_clear_clicked()
    QApplication.processEvents()
    assert history._table.rowCount() == 0
    assert client.plan_history() == ()
    assert json.loads(path.read_text()) == []
    assert not history._clear_button.isEnabled()
    history.close()


def test_history_items_are_saved_in_the_queue_server_format(RE, det, tmp_path):
    path = tmp_path / "history.json"
    client, history = _history(RE, det, path)
    _finish_plan(RE, det)
    history.close()

    (entry,) = json.loads(path.read_text())
    assert entry["item_type"] == "plan"
    assert entry["name"] == "count"
    assert entry["args"] == [["det"]]
    assert entry["kwargs"] == {"num": 2}
    assert entry["item_uid"]
    result = entry["result"]
    assert result["exit_status"] == "completed"
    assert result["scan_ids"] == [1]
    assert len(result["run_uids"]) == 1
    assert result["time_start"] <= result["time_stop"]


def test_history_is_loaded_on_startup_with_device_names(RE, det, tmp_path):
    path = tmp_path / "history.json"
    _, history = _history(RE, det, path)
    _finish_plan(RE, det)
    history.close()

    client, history = _history(RE, det, path)
    assert history._table.rowCount() == 1
    (item,) = client.plan_history()
    assert isinstance(item, HistoryItem)
    assert item.args == ([DeviceName("det")],)
    assert isinstance(item.args[0][0], DeviceName)
    assert client.serialize_plan(item.name, *item.args, **item.kwargs) == (
        "RE(count([det], num=2))"
    )
    history.close()


def test_unreadable_history_file_is_left_alone(RE, det, tmp_path):
    path = tmp_path / "history.json"
    path.write_text("not json")
    client = _client(RE, det, path)
    token = client.subscribe_history(lambda history: None)
    RE(count([det]))
    client.unsubscribe_history(token)
    assert path.read_text() == "not json"
    assert len(client.plan_history()) == 1


def _pause():
    yield from bps.checkpoint()
    yield from bps.pause()
    yield from bps.null()


def _fail():
    yield from bps.null()
    raise RuntimeError("boom")


@pytest.mark.parametrize(
    ("end", "status"),
    [("stop", "stopped"), ("abort", "aborted"), ("halt", "halted"), (None, "failed")],
)
def test_history_reports_queue_server_exit_statuses(RE, end, status):
    client = InProcessRunEngineClient({"RE": RE})
    previous = RE.state_hook
    token = client.subscribe_history(lambda history: None)
    try:
        RE(_fail() if end is None else _pause())
    except Exception:
        pass
    if end is not None:
        getattr(RE, end)()
    client.unsubscribe_history(token)
    (item,) = client.plan_history()
    assert item.result.exit_status == status
    assert RE.state_hook is previous


def test_save_history_writes_and_keeps_saving_there(RE, det, tmp_path):
    client = _client(RE, det, None)
    token = client.subscribe_history(lambda history: None)
    RE(count([det]))
    path = tmp_path / "saved.json"
    client.save_history(path)
    assert len(json.loads(path.read_text())) == 1
    RE(count([det]))
    assert len(json.loads(path.read_text())) == 2
    client.unsubscribe_history(token)


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


@pytest.mark.parametrize(
    ("scan_ids", "shown"),
    [
        ([], ""),
        ([7], "7"),
        ([2, 3, 4, 5, 6], "2 - 6"),
        ([1, 3, 4, 5, 9, 10], "1, 3 - 5, 9 - 10"),
    ],
)
def test_consecutive_scan_ids_are_shown_as_ranges(scan_ids, shown):
    assert _scan_id_ranges(scan_ids) == shown


@pytest.mark.parametrize(("num_uids", "shown"), [(3, None), (4, "4 UIDs")])
def test_uids_beyond_the_maximum_are_counted(RE, det, tmp_path, num_uids, shown):
    if QApplication.instance() is None:
        QApplication([])
    history = QtPlanHistory(_client(RE, det, tmp_path / "history.json"))
    uids = tuple(f"uid-{i}" for i in range(num_uids))
    scan_ids = tuple(range(10, 10 + num_uids))
    history._show(
        (
            HistoryItem(
                "count", item_uid="a", result=PlanResult("completed", uids, scan_ids)
            ),
        )
    )

    uid_cell = history._table.item(0, 6)
    assert uid_cell.text() == (shown or "\n".join(uids))
    assert uid_cell.toolTip() == "\n".join(uids)
    scan_cell = history._table.item(0, 5)
    assert scan_cell.text() == f"10 - {9 + num_uids}"
    assert scan_cell.toolTip() == ", ".join(str(s) for s in scan_ids)


def test_edit_and_rerun_prefills_the_form_and_runs_the_edited_plan(
    RE, det, tmp_path, monkeypatch
):
    client, history = _history(RE, det, tmp_path / "history.json")
    RE(count([det], num=2, md={"note": "kept"}))
    QApplication.processEvents()
    history._table.selectRow(0)
    assert history._edit_button.isEnabled()
    ran: list[str] = []
    monkeypatch.setattr("hextools.gui._ipython.run_in_ipython", ran.append)

    # A modal error dialog would block the headless test forever.
    def fail(parent, title, error):
        raise AssertionError(f"{title}: {error}")

    monkeypatch.setattr("hextools.gui.plan_widget.show_plan_error", fail)
    try:
        history._on_edit_clicked()
        (editor,) = history._editors.values()
        (devices,) = [f for f in editor._device_fields if f["name"] == "detectors"]
        assert devices["widget"].selected_devices() == ["det"]
        (num,) = [w for n, _, w, _ in editor._scalar_fields if n == "num"]
        assert num.text() == "2"
        # md has no field and can't be passed in a plan call, so it is left out.
        assert "md" in editor._extras_label.text()

        num.setText("5")
        editor._on_submit_clicked()
        assert not editor.isVisible()
        QApplication.processEvents()
        (call,) = ran
        assert call.startswith("RE(count(") and "detectors=[det]" in call
        assert "num=5" in call and "md=" not in call
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert history._editors == {}
    finally:
        history.close()
        QApplication.processEvents()


def test_a_failed_edited_rerun_shows_its_form_again(RE, det, tmp_path, monkeypatch):
    client, history = _history(RE, det, tmp_path / "history.json")
    _finish_plan(RE, det)
    history._table.selectRow(0)
    monkeypatch.setattr(
        "hextools.gui._ipython.run_in_ipython", lambda code: RuntimeError("boom")
    )
    errors = []
    monkeypatch.setattr(
        "hextools.gui.plan_widget.show_plan_error", lambda *a: errors.append(a[2])
    )
    try:
        editor = history.edit_and_rerun(history._items[0])
        editor._on_submit_clicked()
        QApplication.processEvents()
        assert editor.isVisible() and history._editors
        assert "boom" in str(errors[0])
    finally:
        history.close()
        QApplication.processEvents()
