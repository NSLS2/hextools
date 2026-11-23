import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from bluesky.plans import count, scan  # noqa: E402
from ophyd_async.core import init_devices  # noqa: E402
from ophyd_async.sim import PatternGenerator, SimMotor, SimPointDetector  # noqa: E402

from hextools.gui._plan_queue import QueueEditor  # noqa: E402
from hextools.gui.run_engine_clients import (  # noqa: E402
    DeviceName,
    InProcessRunEngineClient,
    PlanQueue,
    QueueError,
    QueueItem,
)


def _uids(queue) -> str:
    return "".join(item.item_uid for item in queue.items)


def _editor(uids: str = "abcd") -> QueueEditor[QueueItem]:
    return QueueEditor([QueueItem("count", item_uid=uid) for uid in uids])


# --- QueueEditor: positions and UIDs as in the Queue Server --------------------


@pytest.mark.parametrize(
    ("pos", "expected"),
    [
        (None, "abcdX"),
        ("back", "abcdX"),
        ("front", "Xabcd"),
        (0, "Xabcd"),
        (2, "abXcd"),
        (4, "abcdX"),
        # Past either end goes to that end.
        (10, "abcdX"),
        (-10, "Xabcd"),
        # Negative indices count from the back, -1 being the back.
        (-1, "abcdX"),
        (-2, "abcXd"),
        (-4, "aXbcd"),
    ],
)
def test_add_at_a_position(pos, expected):
    queue = _editor()
    queue.add(QueueItem("count", item_uid="X"), pos=pos)
    assert _uids(queue) == expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [({"before_uid": "b"}, "aXbcd"), ({"after_uid": "b"}, "abXcd")],
)
def test_add_next_to_another_item(kwargs, expected):
    queue = _editor()
    queue.add(QueueItem("count", item_uid="X"), **kwargs)
    assert _uids(queue) == expected


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"pos": 1, "before_uid": "b"}, "Ambiguous"),
        ({"before_uid": "b", "after_uid": "c"}, "Ambiguous"),
        ({"after_uid": "Z"}, "not in the queue"),
        ({"pos": "middle"}, "incorrect value"),
    ],
)
def test_add_refuses_bad_positions(kwargs, match):
    with pytest.raises(QueueError, match=match):
        _editor().add(QueueItem("count", item_uid="X"), **kwargs)


def test_add_refuses_a_uid_already_in_the_queue():
    with pytest.raises(QueueError, match="already in the queue"):
        _editor().add(QueueItem("count", item_uid="b"))


def test_add_batch_keeps_its_order_where_one_item_would_go():
    queue = _editor()
    queue.add_batch([QueueItem("count", item_uid=u) for u in "XY"], after_uid="a")
    assert _uids(queue) == "aXYbcd"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [({}, "d"), ({"pos": "front"}, "a"), ({"pos": -2}, "c"), ({"uid": "b"}, "b")],
)
def test_get_and_remove(kwargs, expected):
    assert _editor().get(**kwargs).item_uid == expected
    queue = _editor()
    assert queue.remove(**kwargs).item_uid == expected
    assert _uids(queue) == "abcd".replace(expected, "")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"pos": 4}, "out of range"),
        ({"pos": -5}, "out of range"),
        ({"uid": "Z"}, "not in the queue"),
        ({"pos": 0, "uid": "a"}, "Ambiguous"),
    ],
)
def test_get_refuses_missing_items(kwargs, match):
    with pytest.raises(QueueError, match=match):
        _editor().get(**kwargs)


def test_get_from_an_empty_queue_fails():
    with pytest.raises(QueueError, match="Queue is empty"):
        _editor("").get()


def test_remove_batch_ignores_missing_uids_unless_asked_not_to():
    queue = _editor()
    assert [i.item_uid for i in queue.remove_batch(uids=["c", "Z", "a"])] == ["c", "a"]
    assert _uids(queue) == "bd"
    with pytest.raises(QueueError, match="following UIDs"):
        _editor().remove_batch(uids=["c", "Z"], ignore_missing=False)
    with pytest.raises(QueueError, match="repeated"):
        _editor().remove_batch(uids=["c", "c"], ignore_missing=False)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        # A plan moved to an index takes that place, from either side.
        ({"uid": "a", "pos_dest": 2}, "bcad"),
        ({"uid": "d", "pos_dest": 1}, "adbc"),
        ({"pos": 0, "pos_dest": "back"}, "bcda"),
        ({"pos": "back", "pos_dest": "front"}, "dabc"),
        ({"uid": "a", "pos_dest": -1}, "bcda"),
        ({"uid": "a", "before_uid": "d"}, "bcad"),
        ({"uid": "d", "after_uid": "a"}, "adbc"),
        ({"uid": "b", "pos_dest": 1}, "abcd"),
    ],
)
def test_move(kwargs, expected):
    queue = _editor()
    queue.move(**kwargs)
    assert _uids(queue) == expected


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"pos_dest": 1}, "Source position or UID is not specified"),
        ({"uid": "a"}, "Destination position or UID is not specified"),
        ({"uid": "Z", "pos_dest": 1}, "Source plan"),
        ({"uid": "a", "pos_dest": 9}, "Destination plan"),
        ({"uid": "a", "before_uid": "b", "after_uid": "c"}, "Ambiguous"),
    ],
)
def test_move_refuses_bad_arguments(kwargs, match):
    with pytest.raises(QueueError, match=match):
        _editor().move(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"uids": ["d", "a"], "pos_dest": "front"}, "dabc"),
        ({"uids": ["d", "a"], "pos_dest": "front", "reorder": True}, "adbc"),
        ({"uids": ["a", "c"], "after_uid": "d"}, "bdac"),
    ],
)
def test_move_batch(kwargs, expected):
    queue = _editor()
    queue.move_batch(**kwargs)
    assert _uids(queue) == expected


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"uids": ["a"]}, "not specified"),
        ({"uids": ["a"], "pos_dest": 0, "after_uid": "b"}, "mutually exclusive"),
        ({"uids": ["a", "b"], "after_uid": "b"}, "is in the batch"),
        ({"uids": ["a", "Z"], "pos_dest": 0}, "following UIDs"),
    ],
)
def test_move_batch_refuses_bad_arguments(kwargs, match):
    with pytest.raises(QueueError, match=match):
        _editor().move_batch(**kwargs)


# --- InProcessRunEngineClient ----------------------------------------------------


@pytest.fixture
def devices(RE):
    # init_devices names devices after the variables they are assigned to
    with init_devices():
        det = SimPointDetector(PatternGenerator())
        motor = SimMotor()
    return det, motor


def _client(RE, devices, path=None):
    det, motor = devices
    namespace = {"RE": RE, "det": det, "motor": motor, "count": count, "scan": scan}
    return InProcessRunEngineClient(namespace, queue_file=path)


def _count(num: int = 1) -> QueueItem:
    return QueueItem("count", ([DeviceName("det")],), {"num": num})


def test_added_items_get_a_uid_and_user(RE, devices):
    client = _client(RE, devices)
    item = client.item_add(_count())
    assert item.item_uid and item.user and item.user_group == "primary"
    assert client.queue_get().items == (item,)
    assert client.item_get(uid=item.item_uid) == item


def test_every_edit_notifies_subscribers_with_a_new_queue_uid(RE, devices):
    client = _client(RE, devices)
    queues: list[PlanQueue] = []
    client.subscribe_queue(queues.append)
    first = client.item_add(_count(1))
    client.item_add(_count(2), pos="front")
    client.item_move(uid=first.item_uid, pos_dest="front")
    client.queue_clear()
    assert [[i.kwargs["num"] for i in q.items] for q in queues] == [
        [1],
        [2, 1],
        [1, 2],
        [],
    ]
    assert len({q.plan_queue_uid for q in queues}) == 4


@pytest.mark.parametrize(
    ("item", "match"),
    [
        (QueueItem("not_a_plan"), "not in the plan catalog"),
        (QueueItem("count", (), {"num": 1}), "missing a required argument"),
        (QueueItem("count", ([DeviceName("det")],), {"nope": 1}), "unexpected"),
        (QueueItem("count", ([DeviceName("ghost")],)), "no device called 'ghost'"),
        (QueueItem("count", ([DeviceName("det")],), {"num": {"a": 1}}), "serialize"),
    ],
)
def test_items_that_cannot_run_are_refused(RE, devices, item, match):
    client = _client(RE, devices)
    queues = []
    client.subscribe_queue(queues.append)
    with pytest.raises(QueueError, match=match):
        client.item_add(item)
    assert client.queue_get().items == () and queues == []


def test_a_failed_batch_leaves_the_queue_as_it_was(RE, devices):
    client = _client(RE, devices)
    kept = client.item_add(_count())
    before = client.queue_get()
    with pytest.raises(QueueError):
        client.item_add_batch([_count(2), QueueItem("not_a_plan")])
    with pytest.raises(QueueError):
        client.item_move_batch(uids=[kept.item_uid, "missing"], pos_dest="front")
    assert client.queue_get() == before


@pytest.mark.parametrize("replace", [False, True])
def test_update_keeps_the_uid_unless_replacing(RE, devices, replace):
    client = _client(RE, devices)
    client.item_add(_count(1))
    original = client.item_add(_count(2))
    client.item_add(_count(3))
    edited = QueueItem("count", original.args, {"num": 5}, original.item_uid)

    updated = client.item_update(edited, replace=replace)

    assert (updated.item_uid == original.item_uid) is not replace
    assert client.item_get(pos=1) == updated
    assert updated.kwargs == {"num": 5}


def test_update_needs_a_queued_uid(RE, devices):
    client = _client(RE, devices)
    with pytest.raises(QueueError, match="no UID"):
        client.item_update(_count())
    with pytest.raises(QueueError, match="not in the queue"):
        client.item_update(QueueItem("count", ([DeviceName("det")],), item_uid="Z"))


def test_the_queue_is_saved_in_the_queue_server_format_and_reloaded(
    RE, devices, tmp_path
):
    path = tmp_path / "state" / "queue.json"
    client = _client(RE, devices, path)
    added = client.item_add_batch(
        [
            _count(3),
            QueueItem("scan", ([DeviceName("det")], DeviceName("motor"), 0, 1, 5)),
        ]
    )

    saved = json.loads(path.read_text())
    assert saved["plan_queue_uid"] == client.queue_get().plan_queue_uid
    first = saved["items"][0]
    assert first["item_type"] == "plan"
    assert first["args"] == [["det"]] and first["kwargs"] == {"num": 3}
    assert first["item_uid"] == added[0].item_uid

    reloaded = _client(RE, devices, path)
    assert reloaded.queue_get() == client.queue_get()
    assert isinstance(reloaded.queue_get().items[1].args[1], DeviceName)


def test_unreadable_queue_file_is_left_alone(RE, devices, tmp_path):
    path = tmp_path / "queue.json"
    path.write_text("not json")
    client = _client(RE, devices, path)
    client.item_add(_count())
    assert path.read_text() == "not json"
    assert len(client.queue_get().items) == 1


@pytest.mark.parametrize(
    ("mock", "name"), [("NO", "plan_queue.json"), ("YES", "plan_queue_mock.json")]
)
def test_default_queue_file_is_next_to_the_history(monkeypatch, mock, name):
    from hextools.gui.__main__ import default_plan_queue_file

    monkeypatch.setenv("HEXTOOLS_RUNNING_IN_CI", mock)
    monkeypatch.setenv("XDG_STATE_HOME", "/xdg/state")
    assert default_plan_queue_file() == Path("/xdg/state/hextools") / name
