import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from bluesky.plans import count  # noqa: E402
from ophyd_async.core import init_devices  # noqa: E402
from ophyd_async.sim import PatternGenerator, SimPointDetector  # noqa: E402
from qtpy.QtCore import QCoreApplication, QEvent  # noqa: E402
from qtpy.QtWidgets import QApplication, QMessageBox  # noqa: E402

from hextools.gui.plan_status import QtPlanHistory  # noqa: E402
from hextools.gui.plan_widget import QtPlanWidget  # noqa: E402
from hextools.gui.queue_controls import QtQueueControls  # noqa: E402
from hextools.gui.queue_editor import QtQueueEditor, QtQueuePage  # noqa: E402
from hextools.gui.run_engine_clients import (  # noqa: E402
    DeviceName,
    InProcessRunEngineClient,
    QueueItem,
    QueueStatus,
)


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def deferred() -> list:
    """Queue steps the client asked to run later, run by the test."""
    return []


@pytest.fixture
def client(RE, monkeypatch, deferred):
    with init_devices():
        det = SimPointDetector(PatternGenerator())
    ns = {"RE": RE, "det": det, "count": count}

    def run(code: str):
        try:
            exec(code, ns)
        except BaseException as error:
            return error
        return None

    monkeypatch.setattr("hextools.gui._ipython.run_in_ipython", run)
    client = InProcessRunEngineClient(ns, call_soon=deferred.append)
    client.subscribe_history(lambda history: None)
    return client


def _count(num: int = 2) -> QueueItem:
    return QueueItem("count", ([DeviceName("det")],), {"num": num})


def _drain(deferred: list):
    while deferred:
        deferred.pop(0)()
    QApplication.processEvents()


def test_queue_buttons_follow_the_queue(app, client, deferred):
    controls = QtQueueControls(client)
    try:
        assert not controls._start_button.isEnabled()
        assert not controls._stop_button.isEnabled()
        assert not controls._clear_button.isEnabled()

        client.item_add(_count())
        QApplication.processEvents()
        assert controls._start_button.isEnabled()
        assert controls._clear_button.isEnabled()
        assert controls._summary.text() == "1 queued, idle"

        controls._start_button.click()
        QApplication.processEvents()
        assert not controls._start_button.isEnabled()
        assert controls._stop_button.isEnabled()

        controls._stop_button.click()
        QApplication.processEvents()
        assert client.queue_status().stop_pending
        assert controls._stop_button.text() == "Cancel Stop"
        assert controls._summary.text() == "1 queued, stopping"
        controls._stop_button.click()
        QApplication.processEvents()
        assert controls._stop_button.text() == "Stop"

        _drain(deferred)
        assert len(client.plan_history()) == 1
        assert controls._summary.text() == "0 queued, idle"
    finally:
        controls.close()


def test_clear_asks_first(app, client, monkeypatch):
    client.item_add(_count())
    controls = QtQueueControls(client)
    try:
        monkeypatch.setattr(
            QMessageBox, "question", lambda *_: QMessageBox.StandardButton.No
        )
        controls._clear_button.click()
        assert len(client.queue_get().items) == 1
        monkeypatch.setattr(
            QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes
        )
        controls._clear_button.click()
        assert client.queue_get().items == ()
    finally:
        controls.close()


def test_autostart_checkbox_runs_added_plans(app, client, deferred):
    controls = QtQueueControls(client)
    try:
        controls._autostart.setChecked(True)
        assert client.queue_status() == QueueStatus(autostart=True)
        client.item_add(_count(3))
        _drain(deferred)
        assert [h.kwargs for h in client.plan_history()] == [{"num": 3}]
        assert controls._autostart.isChecked()
    finally:
        controls.close()


def test_plan_form_adds_to_the_queue_and_runs_now_only_when_it_is_idle(
    app, client, deferred
):
    form = QtPlanWidget(client, "count")
    try:
        form.set_values({"detectors": ["det"], "num": 4})
        form._queue_button.click()
        (item,) = client.queue_get().items
        assert item.name == "count" and item.kwargs["num"] == 4
        assert item.kwargs["detectors"] == [DeviceName("det")]
        assert "1 queued" in form._status_label.text()

        assert form._submit_button.isEnabled()
        client.queue_start()
        QApplication.processEvents()
        assert not form._submit_button.isEnabled()
        _drain(deferred)
        assert form._submit_button.isEnabled()
    finally:
        form.close()


def test_plan_form_shows_why_a_plan_was_not_queued(app, client):
    form = QtPlanWidget(client, "count")
    try:
        form.set_values({"num": "lots"})
        form._queue_button.click()
        assert client.queue_get().items == ()
        assert "not a valid int" in form._status_label.text()
    finally:
        form.close()


def test_history_adds_the_selected_plan_to_the_queue(app, client, deferred, RE):
    history = QtPlanHistory(client)
    try:
        client.execute_plan("count", [DeviceName("det")], num=5)
        QApplication.processEvents()
        history._table.selectRow(0)
        assert history._queue_button.isEnabled()
        history._queue_button.click()
        (item,) = client.queue_get().items
        assert (item.name, item.args, item.kwargs) == (
            "count",
            ([DeviceName("det")],),
            {"num": 5},
        )

        assert history._rerun_button.isEnabled()
        client.queue_start()
        QApplication.processEvents()
        assert not history._rerun_button.isEnabled()
        assert history._queue_button.isEnabled()
        _drain(deferred)
    finally:
        history.close()


def _nums(client) -> list:
    return [item.kwargs["num"] for item in client.queue_get().items]


def test_queue_editor_moves_and_removes_the_selected_plan(app, client):
    client.item_add_batch([_count(1), _count(2), _count(3)])
    editor = QtQueueEditor(client)
    try:
        assert editor._table.rowCount() == 3
        assert editor._table.item(1, 2).text() == "[det], num=2"
        assert not editor._up_button.isEnabled()

        editor._table.selectRow(0)
        assert not editor._up_button.isEnabled()
        editor._down_button.click()
        QApplication.processEvents()
        assert _nums(client) == [2, 1, 3]
        # The moved plan stays selected, so it can be moved on.
        editor._down_button.click()
        QApplication.processEvents()
        assert _nums(client) == [2, 3, 1]
        assert not editor._down_button.isEnabled()
        editor._up_button.click()
        QApplication.processEvents()
        assert _nums(client) == [2, 1, 3]

        editor._remove_button.click()
        QApplication.processEvents()
        assert _nums(client) == [2, 3]
        assert editor._table.rowCount() == 2
    finally:
        editor.close()


def test_queue_editor_saves_an_edited_plan_in_place(app, client):
    first, middle, last = client.item_add_batch([_count(1), _count(2), _count(3)])
    editor = QtQueueEditor(client)
    try:
        editor._table.selectRow(1)
        editor._edit_button.click()
        (form,) = editor._editors.values()
        assert not form._submit_button.isVisibleTo(form)
        assert form._save_button.isVisibleTo(form)
        (num,) = [w for n, _, w, _ in form._scalar_fields if n == "num"]
        assert num.text() == "2"

        num.setText("7")
        form._save_button.click()
        QApplication.processEvents()
        assert _nums(client) == [1, 7, 3]
        saved = client.queue_get().items[1]
        assert saved.item_uid == middle.item_uid
        assert saved.kwargs["detectors"] == [DeviceName("det")]
        # Saving closes the form.
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert editor._editors == {}
    finally:
        editor.close()


def test_a_plan_edited_after_it_left_the_queue_is_not_saved(app, client):
    (item,) = client.item_add_batch([_count(1)])
    editor = QtQueueEditor(client)
    try:
        form = editor.edit(item)
        client.item_remove(uid=item.item_uid)
        form._save_button.click()
        assert "not in the queue" in form._status_label.text()
        assert form.isVisible()
    finally:
        editor.close()


def test_queue_editor_shows_the_running_plan_but_not_to_change(app, client, deferred):
    client.item_add_batch([_count(1), _count(2)])
    client.queue_start()
    # Hold the queue with the first plan taken off and running.
    queue = client.queue_get()
    client._set_queue(queue.items[1:], queue.items[0])  # noqa: SLF001
    editor = QtQueueEditor(client)
    try:
        assert editor._table.rowCount() == 2
        assert editor._table.item(0, 0).text() == "Running"
        editor._table.selectRow(0)
        assert not editor._remove_button.isEnabled()
        assert not editor._edit_button.isEnabled()
        editor._table.selectRow(1)
        assert editor._remove_button.isEnabled()
        assert not editor._up_button.isEnabled()
    finally:
        editor.close()
        deferred.clear()


def test_queue_page_queues_the_plan_picked(app, client, RE):
    def other(num: int = 1):
        yield from count([])

    client._namespace["other"] = other  # noqa: SLF001
    client.refresh_catalog()
    page = QtQueuePage(client)
    try:
        plans = [page._plans.itemText(i) for i in range(page._plans.count())]
        assert plans == sorted(plans) and {"count", "other"} <= set(plans)

        page._plans.setCurrentIndex(page._plans.findData("count"))
        assert page._form is not None and page._form._plan_name == "count"
        page._form.set_values({"detectors": ["det"], "num": 6})
        page._form._queue_button.click()
        QApplication.processEvents()
        assert _nums(client) == [6]
        assert page.queue_editor._table.rowCount() == 1

        page._plans.setCurrentIndex(page._plans.findData("other"))
        assert page._form._plan_name == "other"
        # A refreshed catalog keeps the plan picked.
        client.refresh_catalog()
        QApplication.processEvents()
        assert page._plans.currentData() == "other"
    finally:
        page.close()
