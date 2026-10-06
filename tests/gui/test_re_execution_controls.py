import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

import bluesky.plans as bp  # noqa: E402
import matplotlib  # noqa: E402
from bluesky import RunEngine  # noqa: E402
from bluesky import plan_stubs as bps  # noqa: E402
from IPython.core.interactiveshell import InteractiveShell  # noqa: E402
from ophyd.sim import det, motor  # noqa: E402
from qtpy.QtCore import QTimer  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from hextools.gui._ipython import run_in_ipython  # noqa: E402, PLC2701
from hextools.gui.re_execution_controls import QtReExecutionControls  # noqa: E402


def _slow_step(detectors, step, pos_cache):
    yield from bps.one_nd_step(detectors, step, pos_cache)
    yield from bps.sleep(0.1)


# RE.resume() installs a SIGINT handler, which Python allows only on the main
# thread; resuming from a worker thread raised and the plan stayed paused.
# PR 94 review: Resume must act on the engine the controls hold, even when the
# shell's RE is a different one.
@pytest.mark.parametrize("given", ["shell", "re", "namespace"])
def test_resume_button_resumes_a_paused_plan(given):
    # As in the GUI: with a Qt matplotlib backend, RE() keeps Qt events flowing.
    matplotlib.use("qtagg")
    app = QApplication.instance() or QApplication([])
    shell = InteractiveShell.instance()
    engine = RunEngine({})
    shell.user_ns.update(
        RE=engine if given == "shell" else RunEngine({}),
        engine=engine,
        bp=bp,
        det=det,
        motor=motor,
        _slow_step=_slow_step,
    )
    if given == "shell":
        controls = QtReExecutionControls(local=True, namespace=shell.user_ns)
    elif given == "re":
        controls = QtReExecutionControls(local=True, re=engine)
    else:
        controls = QtReExecutionControls(local=True, namespace={"RE": engine})
    seen = []

    def resume():
        seen.append(engine.state)
        controls._pb_plan_resume_clicked()

    QTimer.singleShot(
        0,
        lambda: run_in_ipython(
            "engine(bp.scan([det], motor, 0, 1, 10, per_step=_slow_step))"
        ),
    )
    QTimer.singleShot(300, controls._pb_plan_pause_immediate_clicked)
    QTimer.singleShot(800, resume)
    QTimer.singleShot(5000, app.quit)
    app.exec()
    controls._timer.stop()

    assert seen == ["paused"]
    assert engine.state == "idle", f"plan did not resume: state={engine.state}"
    assert motor.position == pytest.approx(1)
