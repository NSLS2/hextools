"""Operator recovery contracts exercised through the complete EDXD plans."""

import pytest
from bluesky import Msg, RunEngine
from bluesky import preprocessors as bpp
from bluesky.utils import RunEngineInterrupted
from ophyd_async.core import EnableDisable, get_mock_put, init_devices, set_mock_value
from test_edxd import _fast, _run
from test_edxd import stage as stage

from hextools.edxd import (
    edxd_calib_scan,
    edxd_count,
    edxd_custom_pos_list_grid,
    edxd_grid_scan,
    edxd_scan,
)
from hextools.motors import SampleTower


def _plan(kind, germ, stage, shutters, tower, *, reset=True):
    x, y = stage
    common = {"germ": germ, "fe_shutter": shutters[0], "photon_shutter": shutters[1]}
    if kind == "count":
        return edxd_count(0.02, germ=germ)
    if kind == "scan":
        return edxd_scan(x, 0, 1, 2, 0.02, reset_position=reset, **common)
    if kind == "grid":
        return edxd_grid_scan(
            x, 0, 1, 2, y, 0, 1, 2, 0.02, reset_position=reset, **common
        )
    if kind == "custom":
        return edxd_custom_pos_list_grid(
            x, [0, 1], y, [0], [1], [2], 0.02, reset_positions=reset, **common
        )
    return edxd_calib_scan(0, 1, count_time=0.1, **common, sample_tower=tower)


@pytest.fixture
async def tower():
    with init_devices(mock=True):
        tower = SampleTower("T:", name="tower")
    _fast(tower)
    await tower.z1.velocity.set(1000)
    return tower


@pytest.mark.parametrize("kind", ["count", "scan", "grid", "custom", "calib"])
async def test_stats_reenabled_before_acquisition(
    RE, germ, stage, two_shutters, tower, kind
):
    # Autosave can leave either switch off; preparation must not trust that state.
    set_mock_value(germ.stats1.enable_callbacks, EnableDisable.DISABLE)
    set_mock_value(germ.stats1.compute_statistics, False)
    messages = []
    RE.msg_hook = messages.append
    _run(RE, _plan(kind, germ, stage, two_shutters, tower))
    assert await germ.stats1.enable_callbacks.get_value() == EnableDisable.ENABLE
    assert await germ.stats1.compute_statistics.get_value() is True
    first_trigger = next(i for i, m in enumerate(messages) if m.command == "trigger")
    for signal in (germ.stats1.enable_callbacks, germ.stats1.compute_statistics):
        assert any(
            m.command == "set" and m.obj is signal for m in messages[:first_trigger]
        )


@pytest.mark.parametrize("kind", ["scan", "grid", "custom", "calib"])
@pytest.mark.parametrize("outcome", ["success", "failure", "abort", "resume"])
async def test_complete_plan_restores_initial_positions(
    RE: RunEngine, germ, stage, two_shutters, tower, kind, outcome, monkeypatch
):
    motors = [*stage, tower.z1]
    for motor, position in zip(motors, [5.0, -3.0, 4.0], strict=True):
        set_mock_value(motor.user_setpoint, position)
        set_mock_value(motor.user_readback, position)
    initial_velocity = await tower.z1.velocity.get_value()
    interrupted = False

    def pause_then(msg):
        yield Msg("pause", None, defer=False)
        return (yield msg)

    def interrupt(msg):
        nonlocal interrupted
        if msg.command == "trigger" and not interrupted:
            interrupted = True
            if outcome in ("abort", "resume"):
                return pause_then(msg), None
        return None, None

    if outcome == "failure":

        def fail_trigger(*args, **kwargs):
            raise RuntimeError("injected acquisition failure")

        monkeypatch.setattr(type(germ), "trigger", fail_trigger)
    plan = bpp.plan_mutator(_plan(kind, germ, stage, two_shutters, tower), interrupt)
    if outcome == "failure":
        with pytest.raises(RuntimeError, match="injected acquisition failure"):
            _run(RE, plan)
    elif outcome in ("abort", "resume"):
        with pytest.raises(RunEngineInterrupted):
            _run(RE, plan)
        assert RE.state == "paused"
        # A pause must leave the motors at the acquisition point, not restore them.
        moved = tower.z1 if kind == "calib" else stage[0]
        initial = 4.0 if kind == "calib" else 5.0
        assert await moved.user_readback.get_value() != initial
        if outcome == "abort":
            RE.abort()
        else:
            RE.resume()
    else:
        _run(RE, plan)

    assert RE.state == "idle"
    assert [await m.user_readback.get_value() for m in motors] == pytest.approx(
        [5.0, -3.0, 4.0]
    )
    assert await tower.z1.velocity.get_value() == initial_velocity
    assert get_mock_put(germ.driver.acquire_time).called


@pytest.mark.parametrize("kind", ["scan", "grid", "custom"])
async def test_position_reset_opt_out_is_preserved(
    RE, germ, stage, two_shutters, tower, kind
):
    for motor, position in zip(stage, [5.0, -3.0], strict=True):
        set_mock_value(motor.user_setpoint, position)
        set_mock_value(motor.user_readback, position)
    _run(RE, _plan(kind, germ, stage, two_shutters, tower, reset=False))
    expected = [1.0, -3.0] if kind == "scan" else [1.0, 1.0]
    assert [await m.user_readback.get_value() for m in stage] == pytest.approx(expected)


async def test_repeated_scan_restores_original_not_scan_start(
    RE, germ, stage, two_shutters
):
    x, _ = stage
    set_mock_value(x.user_setpoint, 5.0)
    set_mock_value(x.user_readback, 5.0)
    docs, _ = _run(
        RE,
        edxd_scan(
            x,
            0,
            1,
            2,
            0.02,
            num_iterations=2,
            germ=germ,
            fe_shutter=two_shutters[0],
            photon_shutter=two_shutters[1],
        ),
    )
    assert len(docs["start"]) == 2
    assert await x.user_readback.get_value() == pytest.approx(5.0)
