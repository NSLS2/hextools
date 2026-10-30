import asyncio
import pprint
import re
import threading
import time
import uuid
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
from bluesky import Msg
from bluesky import preprocessors as bpp
from bluesky._vendor.super_state_machine.errors import TransitionError  # noqa: PLC2701
from bluesky.run_engine import RunEngine
from pytest import FixtureRequest
from tiled.client import from_uri
from tiled.client.container import Container
from tiled.server import SimpleTiledServer

_ALLOWED_PYTEST_TASKS = {"async_finalizer", "async_setup", "async_teardown"}

# ==================================================================================
# One Tiled server per test run
# ==================================================================================
# Starting a SimpleTiledServer takes seconds (it creates its catalog and storage),
# so the whole run shares one. Under pytest-xdist the controller process, which
# otherwise only waits for the workers, starts it in the background while the
# workers import and collect, and hands them its address through a file. Without
# xdist the `_tiled_uri` fixture starts one on first use. Either way the server may
# read anything under pytest's base temporary directory, which holds every test's
# ``tmp_path`` (the workers' too), and each test writes into its own container.

_TILED_URI_FILE = "hextools-tiled-uri"
_TILED_START_TIMEOUT = 120.0


def _is_xdist_controller(config: pytest.Config) -> bool:
    return bool(getattr(config.option, "numprocesses", None)) and not hasattr(
        config, "workerinput"
    )


class _BackgroundTiledServer:
    """Start a SimpleTiledServer in a thread; write its URI to a file when ready."""

    def __init__(self, base: Path):
        self._base = base
        self.uri_file = base / _TILED_URI_FILE
        self._server: SimpleTiledServer | None = None
        self._thread = threading.Thread(target=self._start, daemon=True)
        self._thread.start()

    def _start(self) -> None:
        try:
            self._server = SimpleTiledServer(
                self._base / "tiled_server", readable_storage=[self._base]
            )
            self.uri_file.write_text(self._server.uri)
        except Exception as ex:
            self.uri_file.write_text(f"ERROR: {ex!r}")
            raise

    def close(self) -> None:
        self._thread.join(timeout=_TILED_START_TIMEOUT)
        if self._server is not None:
            self._server.close()


_tiled_server_key = pytest.StashKey["_BackgroundTiledServer"]()


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    if _is_xdist_controller(config):
        # tmp_path_factory's base directory, which contains every worker's. pytest
        # offers no public way to reach it from a hook.
        base = config._tmp_path_factory.getbasetemp()  # noqa: SLF001  # pyright: ignore[reportAttributeAccessIssue]
        config.stash[_tiled_server_key] = _BackgroundTiledServer(base)


def pytest_unconfigure(config: pytest.Config) -> None:
    server = config.stash.get(_tiled_server_key, None)
    if server is not None:
        server.close()


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node) -> None:
    """Tell each xdist worker where the controller's Tiled server will be."""
    server = node.config.stash.get(_tiled_server_key, None)
    if server is not None:
        node.workerinput["hextools_tiled_uri_file"] = str(server.uri_file)


@pytest.fixture(scope="session")
def _tiled_uri(
    request: FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Generator[str, None, None]:
    workerinput = getattr(request.config, "workerinput", {})
    uri_file = workerinput.get("hextools_tiled_uri_file")
    if uri_file is not None:
        path = Path(uri_file)
        deadline = time.monotonic() + _TILED_START_TIMEOUT
        while not path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"The shared Tiled server never wrote {path}")
            time.sleep(0.05)
        uri = path.read_text()
        if uri.startswith("ERROR:"):
            raise RuntimeError(f"The shared Tiled server failed to start: {uri}")
        yield uri
        return
    base = tmp_path_factory.getbasetemp()
    server = SimpleTiledServer(
        tmp_path_factory.mktemp("tiled_server"), readable_storage=[base]
    )
    yield server.uri
    server.close()


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
def tiled_client(
    tmp_path: Path, _tiled_uri: str, request: FixtureRequest
) -> Generator[tuple[Path, Container], None, None]:
    """A data directory, and a container of the run's Tiled server for this test only.

    Each test writes its runs into its own container, so no test sees another's.
    """
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    root = from_uri(_tiled_uri)
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", request.node.name)[:80]
    client = root.create_container(key=f"{name}-{uuid.uuid4().hex[:12]}")
    yield tmp_path / "data", client
    root.context.close()


@pytest.fixture
def skip_long_sleeps(RE: RunEngine) -> Generator[list[float], None, None]:
    """Make the plan's sleeps of a second or more return at once.

    Settling waits (e.g. 4 s after a shutter moves) are real time a test doesn't
    need. Yields the requested lengths of the sleeps it shortened.
    """
    requested: list[float] = []

    def _shorten(msg: Msg) -> Msg:
        if msg.command == "sleep" and msg.args and msg.args[0] >= 1:
            requested.append(msg.args[0])
            return msg._replace(args=(0,))
        return msg

    def _preprocessor(plan):
        return bpp.msg_mutator(plan, _shorten)

    RE.preprocessors.append(_preprocessor)
    yield requested
    RE.preprocessors.remove(_preprocessor)


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
