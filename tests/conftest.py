import asyncio
import pprint
import warnings
from pathlib import Path

from ophyd_async.core import (
    StaticPathProvider,
    UUIDFilenameProvider,
    callback_on_mock_execute,
    init_devices,
    set_mock_value,
)

from hextools.detectors.germ import GeRMDetector
from hextools.photon_delivery_system.shutter import Shutter, ShutterStatus

# Pre-import websockets modules that trigger DeprecationWarnings so they are
# cached in sys.modules before uvicorn's server thread imports them.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import uvicorn.protocols.websockets.auto  # noqa: F401
    except (ImportError, Exception):
        pass

from collections.abc import Callable, Generator

import pytest
from bluesky._vendor.super_state_machine.errors import TransitionError  # noqa: PLC2701
from bluesky.run_engine import RunEngine
from pytest import FixtureRequest
from tiled.client import from_uri
from tiled.client.container import Container
from tiled.server import SimpleTiledServer

_ALLOWED_PYTEST_TASKS = {"async_finalizer", "async_setup", "async_teardown"}


@pytest.fixture(autouse=True)
def _plan_history_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep GUIs built in tests away from the user's real plan history file."""
    monkeypatch.setenv(
        "HEXTOOLS_PLAN_HISTORY_FILE", str(tmp_path / "plan_history.json")
    )


# ==================================================================================
# Copied from ophyd-async conftest.py
# ==================================================================================
def _error_and_kill_pending_tasks(
    loop: asyncio.AbstractEventLoop, test_name: str, test_passed: bool
) -> set[asyncio.Task]:
    """Cancels pending tasks in the event loop for a test. Raises an exception if
    the test hasn't already.

    Args:
        loop: The event loop to check for pending tasks.
        test_name: The name of the test.
        test_passed: Indicates whether the test passed.

    Returns:
        set[asyncio.Task]: The set of unfinished tasks that were cancelled.

    Raises:
        RuntimeError: If there are unfinished tasks and the test didn't fail.
    """
    unfinished_tasks = {
        task
        for task in asyncio.all_tasks(loop)
        if (coro := task.get_coro()) is not None
        and hasattr(coro, "__name__")
        and coro.__name__ not in _ALLOWED_PYTEST_TASKS
        and not task.done()
    }
    for task in unfinished_tasks:
        task.cancel()

    # We only raise an exception here if the test didn't fail anyway.
    # If it did then it makes sense that there's some tasks we need to cancel,
    # but an exception will already have been raised.
    if unfinished_tasks and test_passed:
        raise RuntimeError(
            f"Not all tasks closed during test {test_name}:\n"
            f"{pprint.pformat(unfinished_tasks, width=88)}"
        )

    return unfinished_tasks


@pytest.fixture(autouse=True)
def _set_ci_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HEXTOOLS_RUNNING_IN_CI", "YES")


@pytest.fixture(autouse=True)
def _preserve_detector_state(monkeypatch: pytest.MonkeyPatch):
    # The profiles set this, so trigger() uses each detector's current settings.
    monkeypatch.setenv("OPHYD_ASYNC_PRESERVE_DETECTOR_STATE", "YES")


@pytest.fixture(autouse=True, scope="function")
async def fail_test_on_unclosed_tasks(request: FixtureRequest):
    """Used on every test to ensure failure if there are pending tasks
    by the end of the test.
    """
    try:
        fail_count = request.session.testsfailed
        loop = asyncio.get_running_loop()

        loop.set_debug(True)

        request.addfinalizer(
            lambda: _error_and_kill_pending_tasks(
                loop, request.node.name, request.session.testsfailed == fail_count
            )
        )
    # Once https://github.com/bluesky/ophyd-async/issues/683
    # is finished we can remove this try, except.
    except RuntimeError as error:
        if str(error) != "no running event loop":
            raise error


@pytest.fixture(scope="function")
def RE(request: FixtureRequest) -> RunEngine:
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    RE = RunEngine(
        {"cycle": "2026-1", "data_session": "pass-123456"},
        call_returns_result=True,
        loop=loop,
    )
    fail_count = request.session.testsfailed

    async def _settle_in_flight_tasks():
        # A plan may end with a deliberate wait=False put (e.g. enable_preview_mode
        # starting Acquire); give it a moment to finish before checking for leaks.
        # On Python 3.11 asyncio.wait_for wraps it in an extra task, so it often
        # outlives the RE call. Anything still pending afterwards is a real leak.
        pending = asyncio.all_tasks() - {asyncio.current_task()}
        if pending:
            await asyncio.wait(pending, timeout=1.0)

    def clean_event_loop():
        if RE.state not in ("idle", "panicked"):
            try:
                RE.halt()
            except TransitionError:
                pass

        if loop.is_running():
            asyncio.run_coroutine_threadsafe(_settle_in_flight_tasks(), loop).result()
        loop.call_soon_threadsafe(loop.stop)
        RE._th.join()

        try:
            _error_and_kill_pending_tasks(
                loop, request.node.name, request.session.testsfailed == fail_count
            )
        finally:
            loop.close()

    request.addfinalizer(clean_event_loop)
    return RE


# ==================================================================================


@pytest.fixture
def tiled_client(tmp_path: Path) -> Generator[tuple[Path, Container], None, None]:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    server = SimpleTiledServer(
        tmp_path / "tiled_data", readable_storage=[tmp_path / "data"]
    )
    client = from_uri(server.uri)
    yield tmp_path / "data", client
    client.context.close()
    server.close()


@pytest.fixture
def shutter_factory() -> Callable[[str], Shutter]:
    def _factory(name: str) -> Shutter:
        with init_devices(mock=True):
            shutter = Shutter(name, name=name)
        # the only two arcs Shutter.set awaits: a command put flips the status readback
        callback_on_mock_execute(
            shutter.open_cmd,
            lambda *_: set_mock_value(shutter.status, ShutterStatus.OPEN),
        )
        callback_on_mock_execute(
            shutter.close_cmd,
            lambda *_: set_mock_value(shutter.status, ShutterStatus.CLOSED),
        )
        return shutter

    return _factory


@pytest.fixture
def two_shutters(shutter_factory: Callable[[str], Shutter]) -> tuple[Shutter, Shutter]:
    return shutter_factory("front_end_shutter"), shutter_factory("photon_shutter")


@pytest.fixture
def static_path_provider(tmp_path: Path) -> StaticPathProvider:
    return StaticPathProvider(UUIDFilenameProvider(), tmp_path)


@pytest.fixture
def germ(RE: RunEngine, tmp_path: Path) -> GeRMDetector:
    """A mock GeRM writing where the ``tiled_client`` fixture's server reads."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with init_devices(mock=True):
        germ = GeRMDetector(
            "GeRM:", StaticPathProvider(UUIDFilenameProvider(), data_dir), name="germ"
        )
    set_mock_value(germ.driver.acquire_time, 0.01)
    return germ
