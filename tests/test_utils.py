from types import SimpleNamespace
from typing import Any

import pytest
from ophyd_async.epics.motor import Motor

from hextools.utils import (
    ProposalIDPrompt,
    auto_init_devices,
    initialize_run_engine,
    is_running_in_ci,
    merge_async_iterables,
    print_proposal_info,
    print_version_info,
    show_docs,
    start_beamtime,
)


@pytest.mark.parametrize(
    "env_value, expected",
    [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("", False),
        (None, False),
    ],
)
def test_is_running_in_ci(monkeypatch, env_value, expected):
    if env_value is None:
        monkeypatch.delenv("HEXTOOLS_RUNNING_IN_CI", raising=False)
    else:
        monkeypatch.setenv("HEXTOOLS_RUNNING_IN_CI", env_value)
    assert is_running_in_ci() == expected


async def test_auto_init_devices_preserves_dash_in_name(monkeypatch):
    # CI env makes the processor connect in mock mode.
    monkeypatch.setenv("HEXTOOLS_RUNNING_IN_CI", "yes")
    processor = auto_init_devices(timeout=1.0)
    motor = Motor("TEST:MTR")

    await processor._process_devices({"my-motor": motor})

    # Dash preserved on the device, underscore used as the child separator.
    assert motor.name == "my-motor"
    assert motor.velocity.name == "my-motor_velocity"


async def test_auto_init_devices_context_manager_printout(monkeypatch, capsys):
    # CI env makes the processor connect in mock mode.
    monkeypatch.setenv("HEXTOOLS_RUNNING_IN_CI", "yes")

    async def _fail_connect(*_args, **_kwargs):
        raise TimeoutError("simulated connection failure")

    # Async context manager names/connects devices defined in the block.
    async with auto_init_devices(timeout=1.0):
        good_motor = Motor("TEST:GOOD")
        bad_motor = Motor("TEST:BAD")
        monkeypatch.setattr(bad_motor, "connect", _fail_connect)  # force a failure

    assert good_motor.name == "good_motor"
    assert bad_motor.name == "bad_motor"

    out = capsys.readouterr().out
    assert "Initializing devices" in out
    assert "good_motor" in out
    assert "[  OK  ]" in out
    # The device that failed to connect is reported as disconnected.
    assert "bad_motor" in out
    assert "[  DC  ]" in out


async def test_merge_async_iterables_yields_all_items():
    async def _agen(items):
        for item in items:
            yield item

    merged = [
        item async for item in merge_async_iterables(_agen([1, 2, 3]), _agen([4, 5]))
    ]

    # Order across sources is not guaranteed, so compare as sets.
    assert sorted(merged) == [1, 2, 3, 4, 5]


def test_print_version_info(capsys):
    print_version_info()
    out = capsys.readouterr().out
    assert "Version Information" in out
    for package in ("bluesky", "ophyd_async", "tiled", "hextools"):
        assert package in out


def test_show_docs(capsys):
    show_docs("start", {"scan_id": 7})
    out = capsys.readouterr().out
    assert "start" in out
    assert "scan_id" in out


def test_proposal_id_prompt_tokens():
    fake_re = SimpleNamespace(md={"data_session": "pass-42"})
    fake_shell = SimpleNamespace(execution_count=7)

    tokens = ProposalIDPrompt(fake_re, fake_shell).in_prompt_tokens()  # ty: ignore[invalid-argument-type]

    text = "".join(value for _, value in tokens)
    assert "pass-42" in text
    assert "7" in text


def test_proposal_id_prompt_defaults_when_missing():
    fake_re = SimpleNamespace(md={})
    fake_shell = SimpleNamespace(execution_count=1)

    prompt = ProposalIDPrompt(fake_re, fake_shell)  # ty: ignore[invalid-argument-type]
    text = "".join(value for _, value in prompt.in_prompt_tokens())
    assert "N/A" in text


def test_initialize_run_engine_ci_metadata(monkeypatch):
    monkeypatch.setenv("HEXTOOLS_RUNNING_IN_CI", "yes")
    captured: dict[str, Any] = {}

    def fake_run_engine(md):
        captured["md"] = md
        return md

    monkeypatch.setattr("hextools.utils.RunEngine", fake_run_engine)

    initialize_run_engine()

    md = captured["md"]
    assert md["data_session"] == "pass-123456"
    assert "cycle" in md
    assert md["proposal"]["type"] == "Mock Commissioning"


def test_print_proposal_info_with_proposal(capsys):
    md = {"proposal": {"title": "My Study", "type": "PU", "pi_name": "Ada"}}
    print_proposal_info(md)
    out = capsys.readouterr().out
    assert "My Study" in out
    assert "PU" in out
    assert "Ada" in out


def test_print_proposal_info_without_proposal(capsys):
    print_proposal_info({})
    assert capsys.readouterr().out == ""


def test_start_beamtime(monkeypatch, capsys):
    md = {"proposal": {"title": "My Study", "type": "PU", "pi_name": "Ada"}}
    monkeypatch.setattr("hextools.utils.sync_experiment", lambda *a, **k: md)

    start_beamtime(123456)

    out = capsys.readouterr().out
    assert "123456" in out
    assert "My Study" in out
