import asyncio
import time

import bluesky.plan_stubs as bps
import pytest
from bluesky.protocols import Reading
from bluesky.run_engine import RunEngine
from ophyd_async.core import (
    DeviceMock,
    callback_on_mock_execute,
    init_devices,
    set_mock_value,
)

from hextools.photon_delivery_system.shutter import (
    Shutter,
    ShutterStatus,
    ensure_shutter_closed,
    ensure_shutter_open,
)


@pytest.fixture
def shutter() -> Shutter:
    with init_devices(mock=True):
        ps = Shutter("TEST:SHUTTER:", name="test_shutter")
        ps._mock_class = DeviceMock  # bypass MockShutter's actuation behavior
    return ps


@pytest.fixture
def mock_shutter() -> Shutter:
    with init_devices(mock=True):
        ms = Shutter("TEST:MOCKSHUTTER:", name="mock_shutter")
    return ms


async def _delayed_readback(ps: Shutter, value: bool, delay: float):
    await asyncio.sleep(delay)
    set_mock_value(ps.status, ShutterStatus.OPEN if value else ShutterStatus.CLOSED)


@pytest.mark.parametrize("delay", [0.0, 0.05, 0.1, 0.15])
async def test_shutter_open_close_behavior(
    RE: RunEngine, shutter: Shutter, delay: float
):
    set_mock_value(shutter.status, ShutterStatus.CLOSED)

    callback_on_mock_execute(
        shutter.open_cmd,
        lambda *_: asyncio.ensure_future(_delayed_readback(shutter, True, delay)),
    )
    callback_on_mock_execute(
        shutter.close_cmd,
        lambda *_: asyncio.ensure_future(_delayed_readback(shutter, False, delay)),
    )

    t0 = time.monotonic()
    RE(bps.mv(shutter, True))
    open_duration = time.monotonic() - t0
    assert await shutter.status.get_value() is ShutterStatus.OPEN
    assert open_duration >= delay
    assert open_duration < delay + 0.1

    t0 = time.monotonic()
    RE(bps.mv(shutter, False))
    close_duration = time.monotonic() - t0
    assert await shutter.status.get_value() is ShutterStatus.CLOSED
    assert close_duration >= delay
    assert close_duration < delay + 0.1


async def test_mock_shutter_default_behavior(
    RE: RunEngine, mock_shutter: Shutter, monkeypatch: pytest.MonkeyPatch
):
    uniform_calls: list[tuple[float, float]] = []

    def fake_uniform(a: float, b: float) -> float:
        uniform_calls.append((a, b))
        return 0.1

    monkeypatch.setattr(
        "hextools.photon_delivery_system.shutter.random.uniform", fake_uniform
    )

    assert await mock_shutter.status.get_value() is ShutterStatus.CLOSED

    t0 = time.monotonic()
    RE(bps.mv(mock_shutter, True))
    open_duration = time.monotonic() - t0
    assert await mock_shutter.status.get_value() is ShutterStatus.OPEN
    assert open_duration >= 0.1

    t0 = time.monotonic()
    RE(bps.mv(mock_shutter, False))
    close_duration = time.monotonic() - t0
    assert await mock_shutter.status.get_value() is ShutterStatus.CLOSED
    assert close_duration >= 0.1

    assert uniform_calls == [(0.0, 3.0), (0.0, 3.0)]


@pytest.mark.parametrize("allow_actuation", [True, False])
@pytest.mark.parametrize(
    ("ensure_plan", "initial_status", "desired_status"),
    [
        (ensure_shutter_open, ShutterStatus.CLOSED, ShutterStatus.OPEN),
        (ensure_shutter_closed, ShutterStatus.OPEN, ShutterStatus.CLOSED),
    ],
)
async def test_ensure_shutter_helper_allow_actuation(
    RE: RunEngine,
    shutter: Shutter,
    ensure_plan,
    initial_status: ShutterStatus,
    desired_status: ShutterStatus,
    allow_actuation: bool,
):
    set_mock_value(shutter.status, initial_status)
    callback_on_mock_execute(
        shutter.open_cmd,
        lambda *_: set_mock_value(shutter.status, ShutterStatus.OPEN),
    )
    callback_on_mock_execute(
        shutter.close_cmd,
        lambda *_: set_mock_value(shutter.status, ShutterStatus.CLOSED),
    )

    if allow_actuation:
        RE(ensure_plan(shutter, allow_actuation=True))
        assert await shutter.status.get_value() is desired_status
    else:
        with pytest.raises(RuntimeError):
            RE(ensure_plan(shutter, allow_actuation=False))
        assert await shutter.status.get_value() is initial_status


async def test_subscribe_reading_translates_status_to_bool(
    RE: RunEngine, shutter: Shutter
):
    set_mock_value(shutter.status, ShutterStatus.CLOSED)

    received: list[dict[str, Reading[bool]]] = []
    shutter.subscribe_reading(received.append)

    # Subscribing delivers an immediate reading for the current CLOSED status.
    assert received[-1].keys() == {shutter.name}
    assert received[-1][shutter.name]["value"] is False
    assert "timestamp" in received[-1][shutter.name]

    set_mock_value(shutter.status, ShutterStatus.OPEN)
    assert received[-1][shutter.name]["value"] is True

    set_mock_value(shutter.status, ShutterStatus.CLOSED)
    assert received[-1][shutter.name]["value"] is False

    # After clearing the subscription no further updates are delivered.
    shutter.clear_sub(received.append)
    count = len(received)
    set_mock_value(shutter.status, ShutterStatus.OPEN)
    assert len(received) == count
