"""The session link (D-0049): a GUI process that crashes cannot stop a plan."""

import json
import os
import stat
import subprocess
import sys
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

import numpy as np  # noqa: E402
from bluesky import RunEngine  # noqa: E402
from bluesky import plan_stubs as bps  # noqa: E402
from bluesky import plans as bp  # noqa: E402
from bluesky import preprocessors as bpp  # noqa: E402
from bluesky.callbacks.best_effort import BestEffortCallback  # noqa: E402
from ophyd_async.core import StandardReadable, init_devices  # noqa: E402
from ophyd_async.sim import SimMotor  # noqa: E402
from qtpy.QtCore import QCoreApplication  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from hextools.gui.session_client import (  # noqa: E402
    SessionClient,
    SessionDeviceSource,
    SessionError,
)
from hextools.gui.session_protocol import MAX_ARRAY_ELEMENTS, to_jsonable  # noqa: E402
from hextools.gui.session_server import SessionServer, exec_in  # noqa: E402


def _wait_until(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        QCoreApplication.processEvents()
        time.sleep(0.01)


@pytest.fixture
def link():
    QApplication.instance() or QApplication([])
    RE = RunEngine({})
    with init_devices(mock=True):
        motor = SimMotor("motor")
    namespace = {"RE": RE, "bp": bp, "motor": motor}
    server = SessionServer(RE, namespace, execute=exec_in(namespace))
    client = SessionClient(server.path)
    client.connect_to_session()
    yield RE, namespace, server, client
    client.close()
    server.close()


# Thomas Hopkins (2026-10-28): with the GUI in the RunEngine's process, a GUI
# crash ended the plan without its cleanup. The GUI is now a separate process.
def test_a_killed_gui_process_does_not_stop_the_plan_or_its_cleanup(link):
    RE, namespace, server, _ = link
    cleaned_up = []

    def plan_with_cleanup():
        def body():
            for _ in range(10):
                yield from bps.sleep(0.1)

        def cleanup():
            cleaned_up.append(RE.state)
            yield from bps.null()

        yield from bpp.finalize_wrapper(body(), cleanup())

    namespace["plan_with_cleanup"] = plan_with_cleanup
    request = json.dumps({"id": 1, "op": "exec", "code": "RE(plan_with_cleanup())"})
    gui = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import socket, sys, time\n"
            "s = socket.socket(socket.AF_UNIX)\n"
            f"s.connect({str(server.path)!r})\n"
            f"s.sendall({request!r}.encode() + b'\\n')\n"
            "print('sent', flush=True)\n"
            "time.sleep(60)\n",
        ],
        stdout=subprocess.PIPE,
    )
    assert gui.stdout is not None and gui.stdout.readline() == b"sent\n"
    state_at_kill = []

    def kill_gui():
        state_at_kill.append(RE.state)
        gui.kill()

    killer = threading.Timer(0.4, kill_gui)
    killer.start()
    try:
        # The command runs on this (the session's main) thread, as at the prompt.
        _wait_until(lambda: bool(cleaned_up))
    finally:
        killer.cancel()
        gui.kill()
        gui.wait()
        gui.stdout.close()

    assert state_at_kill == ["running"]
    assert gui.returncode == -9
    assert cleaned_up == ["running"]
    assert RE.state == "idle"
    _wait_until(lambda: server.client_count == 1)  # only the fixture's client


def test_the_socket_is_private_to_this_user(link):
    _, _, server, _ = link
    mode = stat.S_IMODE(os.stat(server.path.parent).st_mode)
    assert mode == 0o700


def test_a_plan_run_from_the_gui_reports_back(link):
    _, _, _, client = link
    started, finished, done = [], [], []
    client.plan_started.connect(lambda name, call: started.append(name))
    client.plan_finished.connect(finished.append)

    client.execute("RE(bp.count([motor], num=2))", done.append)
    _wait_until(lambda: bool(done) and bool(finished))

    assert done == [None]
    assert started == ["count"]
    assert finished[0].exit_status == "success"
    assert finished[0].rerunnable


def test_an_error_in_the_session_comes_back_with_its_type(link):
    _, _, _, client = link
    done = []

    client.execute("1/0", done.append)
    _wait_until(lambda: bool(done))

    assert isinstance(done[0], SessionError)
    assert done[0].type_name == "ZeroDivisionError"


# The session sends each device's whole class hierarchy, so a device matches a
# plan's type hint exactly when isinstance would in the session.
def test_devices_match_on_their_class_hierarchy(link):
    _, _, _, client = link
    source = SessionDeviceSource(client)

    assert source.list_devices((SimMotor,)) == ["motor"]
    assert source.list_devices((StandardReadable,)) == ["motor"]
    assert source.list_devices((BestEffortCallback,)) == []
    assert "motor" in source.list_devices()
    assert not any(client.is_signal(name) for name in source.list_devices())


def test_devices_made_at_the_prompt_reach_the_gui(link):
    _, _, _, client = link
    changed = []
    client.devices_changed.connect(lambda: changed.append(True))
    done = []

    client.execute(
        "from ophyd_async.core import init_devices\n"
        "from ophyd_async.sim import SimMotor\n"
        "with init_devices(mock=True):\n"
        "    stage = SimMotor('stage')",
        done.append,
    )
    _wait_until(lambda: bool(changed))

    assert "stage" in SessionDeviceSource(client).list_devices((SimMotor,))


# The GUI draws the live plots; the session's own BestEffortCallback stops
# plotting while one is attached, and plots again once none is.
def test_session_plots_pause_while_a_gui_is_attached(link):
    _, namespace, server, client = link
    bec = BestEffortCallback()
    namespace["bec"] = bec
    client.close()
    _wait_until(lambda: server.client_count == 0)

    second = SessionClient(server.path)
    second.connect_to_session()
    _wait_until(lambda: not bec._plots_enabled)
    second.close()
    _wait_until(lambda: bec._plots_enabled)


def test_large_arrays_are_sent_as_their_shape():
    big = np.zeros((MAX_ARRAY_ELEMENTS + 1,), dtype="uint16")

    assert to_jsonable({"image": big, "x": np.float64(1.5)}) == {
        "image": {"shape": [MAX_ARRAY_ELEMENTS + 1], "dtype": "uint16"},
        "x": 1.5,
    }
