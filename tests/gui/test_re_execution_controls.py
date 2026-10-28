import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

import bluesky.plans as bp  # noqa: E402
import matplotlib  # noqa: E402
from bluesky import RunEngine  # noqa: E402
from bluesky import plan_stubs as bps  # noqa: E402
from ophyd.sim import det, motor  # noqa: E402  # type: ignore
from qtpy.QtCore import QCoreApplication, QTimer  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from hextools.gui.re_execution_controls import QtReExecutionControls  # noqa: E402
from hextools.gui.session_client import SessionClient  # noqa: E402
from hextools.gui.session_server import SessionServer, exec_in  # noqa: E402


def _slow_step(detectors, step, pos_cache):
    yield from bps.one_nd_step(detectors, step, pos_cache)
    yield from bps.sleep(0.1)


@pytest.fixture
def session():
    # As in the session: with a Qt matplotlib backend, RE() keeps Qt events flowing.
    matplotlib.use("qtagg")
    QApplication.instance() or QApplication([])
    RE = RunEngine({})
    namespace = {
        "RE": RE,
        "bp": bp,
        "det": det,
        "motor": motor,
        "_slow_step": _slow_step,
    }
    server = SessionServer(RE, namespace, execute=exec_in(namespace))
    client = SessionClient(server.path)
    client.connect_to_session()
    yield RE, client
    client.close()
    server.close()


def _run_until(done, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not done():
        assert time.monotonic() < deadline, "timed out"
        QCoreApplication.processEvents()
        time.sleep(0.01)


# RE.resume() installs a SIGINT handler, which Python allows only on the main
# thread; resuming from a worker thread raised and the plan stayed paused. In the
# session, Resume runs as a command on its main thread.
def test_pause_and_resume_buttons_drive_the_session(session):
    RE, client = session
    controls = QtReExecutionControls(session=client)
    replies = []
    seen = []

    def resume():
        seen.append(RE.state)
        controls._pb_plan_resume_clicked()

    client.execute(
        "RE(bp.scan([det], motor, 0, 1, 10, per_step=_slow_step))", replies.append
    )
    QTimer.singleShot(300, controls._pb_plan_pause_immediate_clicked)
    QTimer.singleShot(800, resume)
    _run_until(lambda: RE.state == "idle" and len(replies) == 1 and seen)
    _run_until(lambda: not controls._command_pending)

    assert seen == ["paused"]
    assert type(replies[0]).__name__ == "RunEngineInterrupted"
    assert motor.position == pytest.approx(1)


# PR 94 review: a double-click on Resume must resume once, not queue a second
# resume() that could run through a later pause or hit an idle engine.
def test_double_click_on_resume_resumes_once(session):
    RE, client = session
    controls = QtReExecutionControls(session=client)
    resumes = []
    real_resume = RE.resume

    def counting_resume():
        resumes.append(RE.state)
        return real_resume()

    RE.resume = counting_resume  # type: ignore
    enabled_after_click = []

    def double_click():
        controls._pb_plan_resume_clicked()
        enabled_after_click.append(controls._pb_plan_resume.isEnabled())
        controls._pb_plan_resume_clicked()

    client.execute("RE(bp.scan([det], motor, 0, 1, 10, per_step=_slow_step))")
    QTimer.singleShot(300, controls._pb_plan_pause_immediate_clicked)
    _run_until(lambda: client.state == "paused")
    double_click()
    _run_until(lambda: RE.state == "idle" and not controls._command_pending)

    assert resumes == ["paused"], f"resume() called {len(resumes)} times: {resumes}"
    assert enabled_after_click == [False]
