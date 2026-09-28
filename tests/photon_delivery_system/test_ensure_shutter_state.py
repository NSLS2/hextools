import asyncio

import pytest
from bluesky.run_engine import RunEngine
from ophyd_async.core import callback_on_mock_execute, init_devices, set_mock_value

from hextools.photon_delivery_system.shutter import (
    Shutter,
    ShutterStatus,
    ensure_shutter_state,
)


@pytest.fixture
def shutter() -> Shutter:
    with init_devices(mock=True):
        ps = Shutter("TEST:SHUTTER:", name="test_shutter")
    return ps


def _status(is_open: bool) -> ShutterStatus:
    return ShutterStatus.OPEN if is_open else ShutterStatus.CLOSED


@pytest.mark.parametrize("allow_actuation", (False, True))
@pytest.mark.parametrize("is_open", (True, False))
def test_ensure_shutter_state_leaves_a_shutter_already_in_state_alone(
    RE: RunEngine, shutter: Shutter, is_open: bool, allow_actuation: bool
):
    """The status readback is a ShutterStatus, so it must be compared as one.

    Compared with the bool the caller passes, it never matched: a shutter that
    was already open raised when actuation was not allowed, and was commanded
    again when it was.
    """
    set_mock_value(shutter.status, _status(is_open))
    commands: list[str] = []
    callback_on_mock_execute(shutter.open_cmd, lambda *_: commands.append("open"))
    callback_on_mock_execute(shutter.close_cmd, lambda *_: commands.append("close"))

    RE(ensure_shutter_state(shutter, is_open, allow_actuation=allow_actuation))

    assert commands == []


@pytest.mark.parametrize("desired_state", (True, False))
def test_ensure_shutter_state_actuates_a_shutter_in_the_other_state(
    RE: RunEngine, shutter: Shutter, desired_state: bool
):
    set_mock_value(shutter.status, _status(not desired_state))
    commands: list[str] = []

    def _on_execute(command: str, reaches_open: bool):
        def _callback(*_):
            commands.append(command)
            asyncio.get_running_loop().call_soon(
                set_mock_value, shutter.status, _status(reaches_open)
            )

        return _callback

    callback_on_mock_execute(shutter.open_cmd, _on_execute("open", True))
    callback_on_mock_execute(shutter.close_cmd, _on_execute("close", False))

    RE(ensure_shutter_state(shutter, desired_state, allow_actuation=True))

    assert commands == ["open" if desired_state else "close"]
