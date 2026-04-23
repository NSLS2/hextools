import asyncio
import pprint
import warnings
from pathlib import Path

# Pre-import websockets modules that trigger DeprecationWarnings so they are
# cached in sys.modules before uvicorn's server thread imports them.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import uvicorn.protocols.websockets.auto  # noqa: F401
    except (ImportError, Exception):
        pass

from collections.abc import Generator

import pytest
from bluesky._vendor.super_state_machine.errors import TransitionError  # noqa: PLC2701
from bluesky.run_engine import RunEngine
from pytest import FixtureRequest
from tiled.client import from_uri
from tiled.client.container import Container
from tiled.server import SimpleTiledServer

_ALLOWED_PYTEST_TASKS = {"async_finalizer", "async_setup", "async_teardown"}


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

    def clean_event_loop():
        if RE.state not in ("idle", "panicked"):
            try:
                RE.halt()
            except TransitionError:
                pass

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
