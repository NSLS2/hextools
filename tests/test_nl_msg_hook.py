from datetime import datetime

import pytest
from bluesky import plan_stubs as bps
from bluesky.utils import Msg

from hextools.utils import nl_msg_hook
from hextools.utils.nl_msg_hook import (
    GroupLabeler,
    MsgHookNarrator,
    StreamNarrator,
    describe_message,
    narrate_stream,
)

AUTO_GROUP_1 = "a1b2c3d4-e5f6-4718-a93a-4b5c6d7e8f90"
AUTO_GROUP_2 = "0f1e2d3c-4b5a-4968-8776-655443322110"


def _msg(command, obj=None, args=(), kwargs=None, time=None):
    """A serialized message, as produced by ``msg_to_json_safe_dict``."""
    msg = {"command": command, "obj": obj, "args": list(args), "kwargs": kwargs or {}, "run": None}
    if time is not None:
        msg["time"] = time
    return msg


# ----------------------------------------------------------------------------------
# describe_message
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "msg, expected",
    [
        (_msg("set", "motor", [1.5]), "Setting 'motor' to 1.5."),
        (_msg("set", "motor"), "Setting 'motor' to a new value."),
        (_msg("set", "shutter", [True]), "Setting 'shutter' to on."),
        (_msg("set", "mode", ["White"]), "Setting 'mode' to 'White'."),
        (_msg("trigger", "det"), "Triggering 'det'."),
        (_msg("read", "det"), "Reading the current value of 'det'."),
        (_msg("read"), "Reading the current value of the device."),
        (_msg("wait"), "Waiting for pending operations to complete."),
        (_msg("wait", kwargs={"group": "align"}), "Waiting for group 'align' to finish."),
        (_msg("sleep", args=[0.5]), "Pausing for 0.5 seconds."),
        (_msg("sleep"), "Pausing briefly."),
        (_msg("checkpoint"), "Marking a checkpoint (a safe point to pause or resume)."),
        (_msg("create", kwargs={"name": "primary"}), "Beginning a new data reading for stream 'primary'."),
        (_msg("save"), "Saving the collected readings as a data point."),
        (
            _msg("open_run", kwargs={"plan_name": "scan", "scan_id": 3}),
            "Starting a new run (plan 'scan', scan id 3).",
        ),
        (_msg("open_run"), "Starting a new run."),
        (_msg("close_run", kwargs={"exit_status": "success"}), "Closing the run (success)."),
        (_msg("stage", "det"), "Staging 'det' for data collection."),
        (_msg("configure", "det", kwargs={"exposure": 0.1}), "Configuring 'det' with exposure=0.1."),
        (_msg("kickoff", "flyer"), "Kicking off flyscan with 'flyer'."),
        (_msg("prepare", "det", [{"number": 5}]), "Preparing 'det' for the next step with number=5."),
        (_msg("null"), "Doing nothing (no-op)."),
    ],
)
def test_describe_message(msg, expected):
    assert describe_message(msg) == expected


def test_describe_message_unknown_command_falls_back():
    assert describe_message(_msg("frobnicate_thing", "dev")) == "Performing 'frobnicate thing' on 'dev'."
    assert describe_message(_msg(None)) == "Performing 'an operation'."


def test_describe_message_group_clause_names_auto_group_after_command():
    msg = _msg("set", "motor", [1], {"group": AUTO_GROUP_1})
    assert describe_message(msg) == "Setting 'motor' to 1, as part of group 'set'."


def test_describe_message_with_timestamp():
    t = datetime(2026, 1, 2, 3, 4, 5, 678000).timestamp()
    assert describe_message(_msg("save", time=t), show_timestamps=True) == (
        "[03:04:05.678] Saving the collected readings as a data point."
    )


def test_describe_message_timestamp_placeholder_without_time():
    assert describe_message(_msg("save"), show_timestamps=True).startswith("[--:--:--] ")


# ----------------------------------------------------------------------------------
# GroupLabeler
# ----------------------------------------------------------------------------------


def test_group_labeler_preserves_user_named_groups():
    groups = GroupLabeler()
    assert groups.label("align", "set") == "align"
    assert groups.label(None) is None


def test_group_labeler_names_auto_groups_after_command_and_numbers_concurrent_ones():
    groups = GroupLabeler()
    assert groups.label(AUTO_GROUP_1, "stage") == "stage"
    assert groups.label(AUTO_GROUP_2, "stage") == "stage 2"
    # The same id keeps its label.
    assert groups.label(AUTO_GROUP_1) == "stage"


def test_group_labeler_recognizes_hex_uuids():
    groups = GroupLabeler()
    assert groups.label(AUTO_GROUP_1.replace("-", ""), "kickoff") == "kickoff"


def test_group_labeler_release_frees_name_for_reuse():
    groups = GroupLabeler()
    groups.label(AUTO_GROUP_1, "set")
    groups.release(AUTO_GROUP_1)
    assert groups.label(AUTO_GROUP_2, "set") == "set"


def test_group_labeler_release_ignores_unknown_and_user_groups():
    groups = GroupLabeler()
    groups.label("align")
    groups.release("align")
    groups.release("never-seen")
    groups.release(None)
    assert groups.label(AUTO_GROUP_1, "set") == "set"


# ----------------------------------------------------------------------------------
# StreamNarrator / narrate_stream
# ----------------------------------------------------------------------------------


def test_set_then_wait_collapses_into_move():
    lines = narrate_stream(
        [
            _msg("set", "m1", [1], {"group": "g"}),
            _msg("set", "m2", [2.5], {"group": "g"}),
            _msg("wait", kwargs={"group": "g"}),
        ]
    )
    assert lines == ["Moving 'm1' to 1 and 'm2' to 2.5 and waiting for them to arrive, as part of group 'g'."]


def test_trigger_then_wait_collapses_into_trigger():
    lines = narrate_stream(
        [
            _msg("trigger", "det1", kwargs={"group": AUTO_GROUP_1}),
            _msg("trigger", "det2", kwargs={"group": AUTO_GROUP_1}),
            _msg("wait", kwargs={"group": AUTO_GROUP_1}),
        ]
    )
    assert lines == [
        "Triggering 'det1' and 'det2' and waiting for them to complete, as part of group 'trigger'."
    ]


def test_mixed_group_lists_each_action_then_the_wait():
    lines = narrate_stream(
        [
            _msg("set", "m1", [1], {"group": "g"}),
            _msg("trigger", "det", kwargs={"group": "g"}),
            _msg("wait", kwargs={"group": "g"}),
        ]
    )
    assert lines == [
        "Setting 'm1' to 1, as part of group 'g'.",
        "Triggering 'det', as part of group 'g'.",
        "Waiting for group 'g' to finish.",
    ]


def test_consecutive_reads_are_batched():
    lines = narrate_stream(
        [_msg("read", "det1"), _msg("read", "det2"), _msg("read", "motor"), _msg("save")]
    )
    assert lines == [
        "Reading the current values of 'det1', 'det2', and 'motor'.",
        "Saving the collected readings as a data point.",
    ]


def test_batch_deduplicates_repeated_objects():
    lines = narrate_stream([_msg("read", "det"), _msg("read", "det")])
    assert lines == ["Reading the current value of 'det'."]


def test_batches_only_join_messages_sharing_a_group():
    lines = narrate_stream(
        [
            _msg("stage", "det1", kwargs={"group": AUTO_GROUP_1}),
            _msg("stage", "det2", kwargs={"group": AUTO_GROUP_1}),
            _msg("stage", "det3"),
        ]
    )
    assert lines == [
        "Staging 'det1' and 'det2' for data collection, as part of group 'stage'.",
        "Staging 'det3' for data collection.",
    ]


def test_prepare_batches_only_join_identical_values():
    lines = narrate_stream(
        [
            _msg("prepare", "det1", [{"number": 5}]),
            _msg("prepare", "det2", [{"number": 5}]),
            _msg("prepare", "det3", [{"number": 7}]),
        ]
    )
    assert lines == [
        "Preparing 'det1' and 'det2' for the next step with number=5.",
        "Preparing 'det3' for the next step with number=7.",
    ]


def test_group_without_wait_is_described_plainly_when_interrupted():
    lines = narrate_stream([_msg("set", "m1", [1], {"group": "g"}), _msg("checkpoint")])
    assert lines == [
        "Setting 'm1' to 1, as part of group 'g'.",
        "Marking a checkpoint (a safe point to pause or resume).",
    ]


def test_unfinished_buffers_are_flushed_at_end_of_stream():
    assert narrate_stream([_msg("read", "det")]) == ["Reading the current value of 'det'."]
    assert narrate_stream([_msg("set", "m1", [1], {"group": "g"})]) == [
        "Setting 'm1' to 1, as part of group 'g'."
    ]


def test_push_returns_lines_only_once_finalized():
    narrator = StreamNarrator()
    assert narrator.push(_msg("read", "det1")) == []
    assert narrator.push(_msg("read", "det2")) == []
    assert narrator.push(_msg("save")) == [
        "Reading the current values of 'det1' and 'det2'.",
        "Saving the collected readings as a data point.",
    ]
    assert narrator.flush() == []


def test_window_splits_batches_spanning_too_long():
    narrator = StreamNarrator(window=10.0)
    out = []
    for t, det in [(0.0, "det1"), (1.0, "det2"), (100.0, "det3")]:
        out += narrator.push(_msg("read", det, time=t))
    out += narrator.flush()
    assert out == [
        "Reading the current values of 'det1' and 'det2'.",
        "Reading the current value of 'det3'.",
    ]


def test_due_flushes_idle_batches_and_groups():
    narrator = StreamNarrator(window=10.0)
    narrator.push(_msg("read", "det", time=0.0))
    assert narrator.due(now=5.0) == []
    assert narrator.due(now=20.0) == ["Reading the current value of 'det'."]

    narrator.push(_msg("set", "m1", [1], {"group": "g"}, time=30.0))
    assert narrator.due(now=35.0) == []
    assert narrator.due(now=50.0) == ["Setting 'm1' to 1, as part of group 'g'."]
    assert narrator.flush() == []


def test_stream_timestamps_use_first_message_of_collapsed_line():
    t0 = datetime(2026, 1, 2, 3, 4, 5).timestamp()
    lines = narrate_stream(
        [_msg("read", "det1", time=t0), _msg("read", "det2", time=t0 + 1)], show_timestamps=True
    )
    assert lines == ["[03:04:05.000] Reading the current values of 'det1' and 'det2'."]


def test_auto_group_names_are_reused_after_wait():
    lines = narrate_stream(
        [
            _msg("set", "m1", [1], {"group": AUTO_GROUP_1}),
            _msg("wait", kwargs={"group": AUTO_GROUP_1}),
            _msg("set", "m1", [2], {"group": AUTO_GROUP_2}),
            _msg("wait", kwargs={"group": AUTO_GROUP_2}),
        ]
    )
    assert lines == [
        "Moving 'm1' to 1 and waiting for it to arrive, as part of group 'set'.",
        "Moving 'm1' to 2 and waiting for it to arrive, as part of group 'set'.",
    ]


# ----------------------------------------------------------------------------------
# MsgHookNarrator / nl_msg_hook
# ----------------------------------------------------------------------------------


def test_nl_msg_hook_is_a_ready_made_narrator():
    assert isinstance(nl_msg_hook, MsgHookNarrator)


def test_hook_serializes_raw_msgs_and_emits_lines():
    emitted: list[str] = []
    hook = MsgHookNarrator(emit=emitted.append)
    hook(Msg("set", "motor", 1, group="g"))
    assert emitted == []
    hook(Msg("wait", None, group="g"))
    assert emitted == ["Moving 'motor' to 1 and waiting for it to arrive, as part of group 'g'."]


def test_hook_accepts_pre_serialized_messages():
    emitted: list[str] = []
    hook = MsgHookNarrator(emit=emitted.append, serialize=False)
    hook(_msg("checkpoint"))
    assert emitted == ["Marking a checkpoint (a safe point to pause or resume)."]


def test_hook_flush_and_due_emit_buffered_lines():
    emitted: list[str] = []
    hook = MsgHookNarrator(emit=emitted.append, serialize=False, window=10.0)
    hook(_msg("read", "det", time=0.0))
    hook.due(now=20.0)
    assert emitted == ["Reading the current value of 'det'."]
    hook(_msg("read", "det"))
    hook.flush()
    assert emitted[-1] == "Reading the current value of 'det'."


def test_hook_defaults_to_print(capsys):
    hook = MsgHookNarrator(serialize=False)
    hook(_msg("checkpoint"))
    assert "Marking a checkpoint" in capsys.readouterr().out


def test_hook_listeners_receive_lines_and_can_be_removed():
    emitted, heard = [], []
    hook = MsgHookNarrator(emit=emitted.append, serialize=False)
    hook.add_listener(heard.append)
    hook(_msg("checkpoint"))
    hook.remove_listener(heard.append)
    hook.remove_listener(heard.append)  # removing twice is harmless
    hook(_msg("save"))
    assert heard == ["Marking a checkpoint (a safe point to pause or resume)."]
    assert len(emitted) == 2


def test_hook_ignores_broken_listeners():
    emitted, heard = [], []

    def broken(_line):
        raise RuntimeError("listener bug")

    hook = MsgHookNarrator(emit=emitted.append, serialize=False)
    hook.add_listener(broken)
    hook.add_listener(heard.append)
    hook(_msg("checkpoint"))
    assert emitted == heard == ["Marking a checkpoint (a safe point to pause or resume)."]


def test_hook_narrates_plan_run_by_run_engine(RE):
    emitted: list[str] = []
    hook = MsgHookNarrator(emit=emitted.append)
    RE.msg_hook = hook

    def plan():
        yield from bps.checkpoint()
        yield from bps.sleep(0.01)
        yield from bps.null()

    RE(plan())
    hook.flush()
    assert emitted == [
        "Marking a checkpoint (a safe point to pause or resume).",
        "Pausing for 0.01 seconds.",
        "Doing nothing (no-op).",
    ]
