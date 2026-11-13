import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

import logging  # noqa: E402

import bluesky.plan_stubs as bps  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from hextools.gui.plan_status import (  # noqa: E402
    QtPlanExecutionView,
    QtPlanLogView,
    QtPlanStatus,
)
from hextools.gui.run_engine_clients import InProcessRunEngineClient  # noqa: E402

_LOGGER = "hextools.test.plan_status"


def outer():
    for _ in range(2):
        yield from bps.open_run(md={"plan_name": "inner"})
        yield from bps.close_run()


def logging_plan(label: str = "x"):
    logging.getLogger(_LOGGER).info("step %s", label)
    logging.getLogger(_LOGGER).debug("detail %s", label)
    yield from bps.null()


def _client(RE):
    logging.getLogger(_LOGGER).setLevel(logging.DEBUG)
    namespace = {"outer": outer, "logging_plan": logging_plan, "RE": RE}
    return InProcessRunEngineClient(namespace, logger_name=_LOGGER)


def test_plan_status_shows_the_plan_and_its_runs(RE):
    if QApplication.instance() is None:
        QApplication([])
    previous = RE.state_hook
    status = QtPlanStatus(_client(RE))
    try:
        RE(outer())
        QApplication.processEvents()
        assert status._plan.text() == "outer"  # noqa: SLF001
        assert status._scan_id.text() == "1, 2"  # noqa: SLF001
        assert status._inner_plan.text() == "inner"  # noqa: SLF001
        assert status._inner_scan_id.text() == "2"  # noqa: SLF001
        assert status._state.text() == "Idle"  # noqa: SLF001
    finally:
        status.close()
    assert RE.state_hook is previous


def test_plan_status_shows_the_last_info_message(RE):
    if QApplication.instance() is None:
        QApplication([])
    status = QtPlanStatus(_client(RE))
    try:
        RE(logging_plan("a"))
        QApplication.processEvents()
        assert status._last.text() == "step a"  # noqa: SLF001
    finally:
        status.close()


def test_execution_view_shows_the_call_and_this_plans_messages(RE):
    if QApplication.instance() is None:
        QApplication([])
    view = QtPlanExecutionView(_client(RE))
    try:
        RE(logging_plan("a"))
        QApplication.processEvents()
        RE(logging_plan("b"))
        QApplication.processEvents()
        assert view._call.text() == "logging_plan(label='b')"  # noqa: SLF001
        text = view._messages.toPlainText()  # noqa: SLF001
        assert "step b" in text and "detail b" in text
        assert "step a" not in text
    finally:
        view.close()


def test_log_view_keeps_every_plans_messages_under_its_call(RE):
    if QApplication.instance() is None:
        QApplication([])
    view = QtPlanLogView(_client(RE))
    try:
        RE(logging_plan("a"))
        RE(logging_plan("b"))
        QApplication.processEvents()
        text = view._log.toPlainText()  # noqa: SLF001
        assert text.index("=== logging_plan(label='a') ===") < text.index("step a")
        assert text.index("step a") < text.index("=== logging_plan(label='b') ===")
        assert "step b" in text
    finally:
        view.close()
