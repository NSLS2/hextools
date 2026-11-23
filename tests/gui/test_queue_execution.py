import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

import bluesky.plan_stubs as bps  # noqa: E402
from bluesky.plans import count  # noqa: E402
from ophyd_async.core import init_devices  # noqa: E402
from ophyd_async.sim import PatternGenerator, SimPointDetector  # noqa: E402

from hextools.gui.run_engine_clients import (  # noqa: E402
    DeviceName,
    InProcessRunEngineClient,
    PlanFailedError,
    PlanQueue,
    QueueError,
    QueueItem,
    QueueStatus,
)


def failing(num: int = 1):
    yield from bps.null()
    raise RuntimeError("boom")


def pausing(num: int = 1):
    yield from bps.checkpoint()
    yield from bps.pause()
    yield from bps.null()


@pytest.fixture
def namespace(RE, monkeypatch):
    with init_devices():
        det = SimPointDetector(PatternGenerator())
    ns = {"RE": RE, "det": det, "count": count, "failing": failing, "pausing": pausing}

    def run(code: str):
        try:
            exec(code, ns)
        except BaseException as error:
            return error
        return None

    monkeypatch.setattr("hextools.gui._ipython.run_in_ipython", run)
    return ns


def _client(namespace, *, deferred: list | None = None, path=None):
    """A client running queued plans at once, or when ``deferred`` is drained."""
    call_soon = deferred.append if deferred is not None else (lambda f: f())
    client = InProcessRunEngineClient(namespace, queue_file=path, call_soon=call_soon)
    client.subscribe_history(lambda history: None)  # records finished plans
    return client


def _count(num: int = 1) -> QueueItem:
    return QueueItem("count", ([DeviceName("det")],), {"num": num})


def _nums(items) -> list:
    return [item.kwargs.get("num") for item in items]


def test_the_queue_runs_front_first_into_the_history(namespace):
    client = _client(namespace)
    queued = client.item_add_batch([_count(4), _count(2), _count(3)])
    statuses: list[QueueStatus] = []
    client.subscribe_queue_status(statuses.append)

    client.queue_start()

    history = client.plan_history()
    assert _nums(history) == [4, 2, 3]
    # The history keeps each plan's queue UID, as the Queue Server's does.
    assert [h.item_uid for h in history] == [q.item_uid for q in queued]
    assert all(h.result.exit_status == "completed" for h in history)
    assert client.queue_get() == PlanQueue((), client.queue_get().plan_queue_uid)
    assert statuses == [QueueStatus(running=True), QueueStatus()]


def test_a_failed_plan_goes_back_to_the_front_and_stops_the_queue(namespace):
    client = _client(namespace)
    client.queue_autostart(True)
    deferred: list = []
    client._call_soon = deferred.append  # noqa: SLF001 - hold the queue
    failed = client.item_add(QueueItem("failing", (), {"num": 7}))
    client.item_add(_count(2))
    deferred.pop()()

    queue = client.queue_get()
    assert _nums(queue.items) == [7, 2] and queue.running_item is None
    assert queue.items[0].item_uid != failed.item_uid
    assert client.plan_history()[-1].result.exit_status == "failed"
    assert client.queue_status() == QueueStatus()  # autostart is off too


def test_a_pending_stop_stops_after_the_running_plan(namespace):
    deferred: list = []
    client = _client(namespace, deferred=deferred)
    client.item_add_batch([_count(1), _count(2)])
    client.queue_autostart(True)
    client.queue_stop()
    assert client.queue_status().stop_pending
    client.queue_stop_cancel()
    client.queue_stop()
    while deferred:
        deferred.pop(0)()

    # The stop came before the first plan started, so nothing ran.
    assert _nums(client.queue_get().items) == [1, 2]
    assert client.queue_status() == QueueStatus()


def test_stop_only_works_while_the_queue_runs(namespace):
    client = _client(namespace)
    with pytest.raises(QueueError, match="not running"):
        client.queue_stop()


def test_run_now_is_refused_while_the_queue_runs(namespace):
    deferred: list = []
    client = _client(namespace, deferred=deferred)
    client.item_add(_count())
    client.queue_start()
    with pytest.raises(PlanFailedError, match="queue is running"):
        client.execute_plan("count", [DeviceName("det")])
    with pytest.raises(QueueError, match="already running"):
        client.queue_start()


def test_autostart_runs_plans_as_they_are_added(namespace):
    client = _client(namespace)
    client.queue_autostart(True)
    client.item_add(_count(4))
    assert _nums(client.plan_history()) == [4]
    assert client.queue_get().items == ()
    # The queue went idle when it emptied, but autostart stays on.
    assert client.queue_status() == QueueStatus(autostart=True)


def test_autostart_waits_for_a_plan_run_now(namespace):
    client = _client(namespace)
    client.queue_autostart(True)

    def queue_while_running(num: int = 1):
        client.item_add(_count(9))
        yield from bps.null()

    namespace["queue_while_running"] = queue_while_running
    client.refresh_catalog()
    client.execute_plan("queue_while_running")

    assert _nums(client.plan_history()) == [None, 9]


def test_a_paused_queued_plan_continues_the_queue_once_resumed(namespace):
    client = _client(namespace)
    client.item_add_batch([QueueItem("pausing", (), {"num": 5}), _count(2)])
    client.queue_start()

    queue = client.queue_get()
    assert queue.running_item is not None and _nums(queue.items) == [2]
    assert client.queue_status().running and client.state == "paused"

    client.resume()
    assert _nums(client.plan_history()) == [5, 2]
    assert client.queue_status() == QueueStatus()


@pytest.mark.parametrize(
    ("interrupt", "requeued"), [("stop", False), ("abort", True), ("halt", True)]
)
def test_ending_a_paused_queued_plan_stops_the_queue(namespace, interrupt, requeued):
    client = _client(namespace)
    paused = client.item_add(QueueItem("pausing", (), {"num": 1}))
    client.item_add(_count(2))
    client.queue_start()

    getattr(client, interrupt)()

    queue = client.queue_get()
    # As the Queue Server, a stopped plan counts as done; others run again.
    assert _nums(queue.items) == ([1, 2] if requeued else [2])
    assert paused.item_uid not in [item.item_uid for item in queue.items]
    assert queue.running_item is None
    assert client.queue_status() == QueueStatus()


def test_a_plan_running_when_the_session_ended_is_requeued(namespace, tmp_path):
    path = tmp_path / "queue.json"
    deferred: list = []
    client = _client(namespace, deferred=deferred, path=path)
    client.item_add_batch([QueueItem("pausing"), _count(2)])
    client.queue_start()
    deferred.pop()()  # pauses with the plan running
    assert json.loads(path.read_text())["running_item"]["name"] == "pausing"
    client.abort()

    path.write_text(
        json.dumps(
            {
                "items": [_count(2).to_dict()],
                "running_item": QueueItem("pausing", item_uid="x").to_dict(),
                "plan_queue_uid": "u",
            }
        )
    )
    reloaded = _client(namespace, path=path)
    assert [i.name for i in reloaded.queue_get().items] == ["pausing", "count"]
    assert reloaded.queue_get().running_item is None
