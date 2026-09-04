import pytest
from ophyd_async.epics.motor import Motor

from hextools.utils import auto_init_devices, is_running_in_ci


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
        bad_motor.connect = _fail_connect  # force this one to fail

    assert good_motor.name == "good_motor"
    assert bad_motor.name == "bad_motor"

    out = capsys.readouterr().out
    assert "Initializing devices" in out
    assert "good_motor" in out
    assert "[  OK  ]" in out
    # The device that failed to connect is reported as disconnected.
    assert "bad_motor" in out
    assert "[  DC  ]" in out


