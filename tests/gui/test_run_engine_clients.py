import enum
import logging
from collections.abc import Sequence

import bluesky.plan_stubs as bps
import pytest
from bluesky import RunEngine
from bluesky.protocols import Movable
from bluesky.utils import FailedStatus, RunEngineInterrupted
from ophyd_async.core import init_devices, soft_signal_rw
from ophyd_async.sim import SimMotor

from hextools.catalog.generate import type_expr
from hextools.catalog.model import DeviceRegistry, PlanCatalog
from hextools.gui import _ipython
from hextools.gui.run_engine_clients import (
    DeviceName,
    InProcessRunEngineClient,
    PlanFailedError,
    PlanInfo,
    PlanPausedError,
    RunEngineClient,
    RunInfo,
)


class _Mode(enum.Enum):
    FAST = "Fast"
    SLOW = "Slow"


@pytest.fixture
def RE():
    return RunEngine({"proposal": {"title": "Test"}})


def _client(RE, namespace=None) -> InProcessRunEngineClient:
    return InProcessRunEngineClient({**(namespace or {}), "RE": RE})


def test_in_process_client_needs_a_run_engine_in_the_namespace():
    with pytest.raises(ValueError):
        InProcessRunEngineClient({})
    with pytest.raises(ValueError):
        InProcessRunEngineClient({"RE": object()})


def test_in_process_client_reports_state_and_metadata(RE):
    client = _client(RE)
    assert isinstance(client, RunEngineClient)
    assert client.state == "idle"
    assert client.metadata()["proposal"] == {"title": "Test"}


def test_in_process_client_runs_plans_as_ipython_cells(RE, monkeypatch):
    ran: list[str] = []
    monkeypatch.setattr(_ipython, "run_in_ipython", ran.append)
    _client(RE).execute_plan(
        "mv", DeviceName("a_slits.vertical_gap"), 0.5, mode=_Mode.FAST, label="it's"
    )
    assert ran == ["RE(mv(a_slits.vertical_gap, 0.5, mode='Fast', label=\"it's\"))"]


def test_in_process_client_raises_the_plans_error(RE, monkeypatch):
    error = RuntimeError("motor stalled")
    monkeypatch.setattr(_ipython, "run_in_ipython", lambda code: error)
    with pytest.raises(PlanFailedError, match="^RuntimeError: motor stalled$") as info:
        _client(RE).execute_plan("mv")
    assert info.value.__cause__ is error


def test_in_process_client_reports_the_cause_of_a_failed_status(RE, monkeypatch):
    error = FailedStatus("status")
    error.__cause__ = TimeoutError("limit switch")
    monkeypatch.setattr(_ipython, "run_in_ipython", lambda code: error)
    with pytest.raises(PlanFailedError, match="^TimeoutError: limit switch$"):
        _client(RE).execute_plan("mv")


def test_in_process_client_reports_a_pause(RE, monkeypatch):
    monkeypatch.setattr(
        _ipython, "run_in_ipython", lambda code: RunEngineInterrupted("paused")
    )
    with pytest.raises(PlanPausedError, match="Resume"):
        _client(RE).execute_plan("mv")


def test_in_process_client_reports_arguments_it_cant_pass(RE):
    with pytest.raises(PlanFailedError, match="^TypeError: "):
        _client(RE).execute_plan("mv", object())


@pytest.mark.parametrize(
    "name, args",
    [
        ("mv; import os", ()),
        ("mv", (DeviceName("x; import os"),)),
        ("mv", (DeviceName("x[__import__('os')]"),)),
        ("mv", ({1: 2},)),
    ],
)
def test_in_process_client_refuses_unsafe_or_unsupported_calls(RE, name, args):
    with pytest.raises((ValueError, TypeError)):
        _client(RE).serialize_plan(name, *args)


def test_in_process_client_serializes_indexed_devices_and_lists(RE):
    assert (
        _client(RE).serialize_plan(
            "count", [DeviceName("axes[1]"), DeviceName("axes.map['x']")], num=2
        )
        == "RE(count([axes[1], axes.map['x']], num=2))"
    )


def test_in_process_client_reports_move_progress_and_restores_the_hook(RE):
    with init_devices():
        motor = SimMotor(name="motor", instant=False)
    previous_calls = []

    def previous(status_objects):
        previous_calls.append(status_objects)

    RE.waiting_hook = previous
    client = _client(RE)
    updates: list[dict] = []
    token = client.subscribe_progress(updates.append)
    RE(bps.mv(motor, 0.1))
    client.unsubscribe_progress(token)

    assert any(u.get("name") == "motor" for u in updates)
    assert updates[-1] == {"completed": True}
    assert previous_calls
    assert RE.waiting_hook is previous


def test_in_process_client_reports_state_changes_and_restores_the_hook(RE):
    previous_calls = []

    def previous(new, old):
        previous_calls.append(new)

    RE.state_hook = previous
    client = _client(RE)
    first: list[str] = []
    second: list[str] = []
    tokens = [
        client.subscribe_state(first.append),
        client.subscribe_state(second.append),
    ]
    RE(bps.null())
    client.unsubscribe_state(tokens[0])
    RE(bps.null())
    client.unsubscribe_state(tokens[1])

    assert first == ["running", "idle"]
    assert second == ["running", "idle"] * 2
    assert previous_calls == ["running", "idle"] * 2
    assert RE.state_hook is previous


def two_runs(detectors, num: int = 1, mode: _Mode = _Mode.FAST):
    for _ in range(2):
        yield from bps.open_run(md={"plan_name": "inner"})
        yield from bps.close_run()


def test_in_process_client_reports_the_current_plan_and_its_runs(RE):
    previous = RE.state_hook
    det = soft_signal_rw(float, name="det")
    client = _client(RE, {"two_runs": two_runs, "det": det})
    updates: list[PlanInfo | None] = []
    token = client.subscribe_plan(updates.append)
    RE(two_runs([det], num=3, mode=_Mode.SLOW))
    RE(two_runs([det], mode=_Mode.FAST))
    client.unsubscribe_plan(token)

    started, first_run, second_run, finished, defaults, *_ = updates
    assert started is not None
    assert started.args == ([DeviceName("det")],)
    assert started.kwargs == {"num": 3, "mode": "Slow"}
    assert started.call == "two_runs([det], num=3, mode='Slow')"
    assert started.runs == () and started.start_time is not None
    assert first_run is not None and second_run is not None
    assert [r.plan_name for r in second_run.runs] == ["inner", "inner"]
    assert [r.scan_id for r in second_run.runs] == [1, 2]
    assert all(isinstance(r, RunInfo) and r.uid for r in second_run.runs)
    assert finished is None
    assert defaults is not None and defaults.call == "two_runs([det])"
    assert client.current_plan() is None
    assert RE.state_hook is previous


def test_plan_info_reads_a_queue_server_running_item():
    item = {
        "item_type": "plan",
        "name": "count",
        "args": [["det1", "det2"]],
        "kwargs": {"num": 10},
        "item_uid": "abc",
        "properties": {"time_start": 12.5},
    }
    runs = (RunInfo.from_dict({"uid": "r1", "is_open": True, "exit_status": None}),)
    plan = PlanInfo.from_dict(item, runs)
    assert plan.call == "count(['det1', 'det2'], num=10)"
    assert plan.start_time == 12.5
    assert plan.runs == (RunInfo("r1"),)


def _move(motors: Sequence[Movable], mode: _Mode = _Mode.FAST, step: float = 1.0):
    """Move some motors."""
    yield from bps.null()


def test_in_process_client_describes_the_namespace_plans_and_devices(RE):
    with init_devices(mock=True):
        motor = SimMotor(name="motor")
    client = _client(RE, {"motor": motor, "move": _move, "not_a_plan": print})
    catalog = client.plan_catalog
    assert list(catalog) == ["move"]
    move = catalog["move"]
    assert (move.module, move.description) == (__name__, "Move some motors.")
    motors, mode, step = move.parameters
    assert motors.name == "motors" and motors.required
    assert motors.type == {"kind": "sequence", "items": type_expr(Movable)}
    assert (mode.type, mode.default) == (type_expr(_Mode), "Fast")
    assert (step.type, step.default) == ({"kind": "number"}, 1.0)
    registry = client.device_registry
    assert list(registry) == ["motor"]
    assert "motor" in registry.choices(motors.type)


def test_refresh_catalog_notifies_subscribers_until_unsubscribed(RE):
    namespace = {"RE": RE}
    client = InProcessRunEngineClient(namespace)
    assert len(client.plan_catalog) == 0 and len(client.device_registry) == 0
    updates: list[tuple[PlanCatalog, DeviceRegistry]] = []
    token = client.subscribe_catalog(lambda c, r: updates.append((c, r)))

    with init_devices(mock=True):
        motor = SimMotor(name="motor")
    namespace["motor"] = motor
    namespace["move"] = _move
    assert "move" not in client.plan_catalog
    client.refresh_catalog()
    ((catalog, registry),) = updates
    assert catalog is client.plan_catalog and registry is client.device_registry
    assert list(catalog) == ["move"] and list(registry) == ["motor"]
    assert catalog.revision == registry.revision == 1

    client.unsubscribe_catalog(token)
    client.refresh_catalog()
    assert len(updates) == 1
    assert client.plan_catalog.revision == 2


async def _noop():
    pass


async def _set(signal, value):
    await signal.set(value)


def test_in_process_client_subscribes_to_readings_and_fetches_units(RE):
    from bluesky.protocols import Reading
    from bluesky.run_engine import call_in_bluesky_event_loop

    energy = soft_signal_rw(float, initial_value=4.0, units="keV", name="energy")
    client = _client(RE, {"energy": energy})
    assert client.fetch_units("energy").result(timeout=5) == "keV"
    readings: list[Reading] = []
    token = client.subscribe_reading("energy", readings.append)
    call_in_bluesky_event_loop(_noop())
    call_in_bluesky_event_loop(_set(energy, 5.0))
    client.unsubscribe_reading(token)
    call_in_bluesky_event_loop(_set(energy, 6.0))
    assert [r["value"] for r in readings] == [4.0, 5.0]
    with pytest.raises(KeyError):
        client.subscribe_reading("missing", readings.append)


def test_in_process_client_reports_log_records_until_unsubscribed(RE):
    client = _client(RE)
    records: list[tuple] = []
    token = client.subscribe_logs(lambda *record: records.append(record))
    logger = logging.getLogger("hextools")
    logger.warning("beam %s", "lost")
    client.unsubscribe_logs(token)
    logger.warning("ignored")
    ((created, levelno, levelname, message),) = records
    assert isinstance(created, float)
    assert (levelno, levelname, message) == (logging.WARNING, "WARNING", "beam lost")
