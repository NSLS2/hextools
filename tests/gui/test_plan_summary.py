import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from bluesky import RunEngine  # noqa: E402
from qtpy.QtCore import QCoreApplication, QEvent  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from hextools.catalog.model import PlanCatalog  # noqa: E402
from hextools.gui.plan_status import QtPlanHistory  # noqa: E402
from hextools.gui.plan_summary import (  # noqa: E402
    Argument,
    QtPlanSummary,
    plan_arguments,
)
from hextools.gui.run_engine_clients import (  # noqa: E402
    DeviceName,
    HistoryItem,
    InProcessRunEngineClient,
    PlanResult,
)


def scan_it(detectors: list, start: float, num: int = 5, *extras, note: str = ""):
    """Scan the detectors.

    Parameters
    ----------
    detectors : list
        What to read.
    start : float
        Where to start.
    num : int, optional
        How many points.
    note : str, optional
        Saved with the run.
    """
    yield from ()


def _plan():
    return PlanCatalog.from_namespace({"scan_it": scan_it})["scan_it"]


def test_arguments_are_matched_to_parameters_with_defaults_and_descriptions():
    item = HistoryItem(
        "scan_it", ([DeviceName("det")], 1.5), {"note": "hi", "extra": 2}
    )

    assert plan_arguments(item, _plan()) == [
        Argument("detectors", "[det]", "What to read.", True),
        Argument("start", "1.5", "Where to start.", True),
        Argument("num", "5", "How many points.", False),
        Argument("note", "'hi'", "Saved with the run.", True),
        Argument("extra", "2", "", True),
    ]


def test_arguments_of_a_plan_missing_from_the_catalog_are_listed_as_passed():
    item = HistoryItem("gone", (1, 2), {"x": 3})

    assert plan_arguments(item, None) == [
        Argument("argument 1", "1", "", True),
        Argument("argument 2", "2", "", True),
        Argument("x", "3", "", True),
    ]


def test_summary_lists_every_run():
    app = QApplication.instance() or QApplication([])
    uids = tuple(f"uid-{i}" for i in range(12))
    item = HistoryItem(
        "scan_it",
        ([DeviceName("det")], 0.0),
        item_uid="a",
        result=PlanResult(
            "failed", uids, tuple(range(1, 13)), msg="boom", traceback="Traceback..."
        ),
    )
    summary = QtPlanSummary(item, _plan())
    try:
        assert summary.runs.rowCount() == 12
        assert summary.runs.item(11, 0).text() == "12"
        assert summary.runs.item(11, 1).text() == "uid-11"
        assert summary.arguments.item(2, 1).text() == "5 (default)"
    finally:
        summary.close()
        app.processEvents()


def test_double_clicking_a_history_row_opens_one_summary_per_plan():
    app = QApplication.instance() or QApplication([])
    client = InProcessRunEngineClient({"RE": RunEngine({}), "scan_it": scan_it})
    history = QtPlanHistory(client)
    history._show((HistoryItem("scan_it", ([], 0.0), item_uid="a"),))
    try:
        history._table.cellDoubleClicked.emit(0, 2)
        history._table.cellDoubleClicked.emit(0, 4)
        (summary,) = history._summaries.values()
        assert summary.isVisible() and summary.item.item_uid == "a"
        deleted = []
        summary.destroyed.connect(lambda *_: deleted.append(True))

        history.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert deleted and history._summaries == {}
    finally:
        history.close()
        app.processEvents()
