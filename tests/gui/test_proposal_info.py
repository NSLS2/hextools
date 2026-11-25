import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from qtpy.QtWidgets import QApplication, QDialog  # noqa: E402

from hextools.gui import proposal_info  # noqa: E402
from hextools.gui.proposal_info import QtProposalInfo  # noqa: E402
from hextools.gui.run_engine_clients import InProcessRunEngineClient  # noqa: E402
from hextools.utils import initialize_run_engine  # noqa: E402
from hextools.utils.proposals import MOCK_PROPOSAL, MockProposalClient  # noqa: E402


class _AcceptedDialog:
    def __init__(self, number: int):
        self._number = number

    def __call__(self, *args, **kwargs):
        return self

    def exec(self):
        return QDialog.DialogCode.Accepted

    def values(self):
        return self._number, "someone", "secret"


def _wait_for(app, condition, timeout=5.0):
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
    assert condition()


def test_change_proposal_switches_the_mock_proposal_in_re_md(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("HEXTOOLS_RUNNING_IN_CI", "yes")
    RE = initialize_run_engine()
    proposals = MockProposalClient()
    target = proposals.list_proposals().user[0]
    widget = QtProposalInfo(InProcessRunEngineClient({"RE": RE}), proposals)
    monkeypatch.setattr(
        proposal_info, "QtChangeProposalDialog", _AcceptedDialog(int(target.number))
    )
    try:
        assert RE.md["data_session"] == f"pass-{MOCK_PROPOSAL.number}"
        widget.change_proposal()
        _wait_for(app, lambda: widget._switch_worker is None)

        assert RE.md["data_session"] == f"pass-{target.number}"
        assert RE.md["username"] == "someone"
        assert RE.md["proposal"]["title"] == target.title
        assert RE.md["proposal"]["pi_name"] == target.pi_name
        _wait_for(
            app,
            lambda: widget._labels["data_session"].text() == f"pass-{target.number}",
        )
    finally:
        widget.close()
        app.processEvents()


def test_a_refused_change_leaves_re_md_alone(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("HEXTOOLS_RUNNING_IN_CI", "yes")
    RE = initialize_run_engine()
    widget = QtProposalInfo(InProcessRunEngineClient({"RE": RE}), MockProposalClient())
    errors = []
    monkeypatch.setattr(proposal_info, "QtChangeProposalDialog", _AcceptedDialog(1))
    monkeypatch.setattr(widget, "_on_switch_errored", errors.append)
    try:
        widget.change_proposal()
        _wait_for(app, lambda: bool(errors))
        assert isinstance(errors[0], ValueError)
        assert RE.md["data_session"] == f"pass-{MOCK_PROPOSAL.number}"
    finally:
        widget.close()
        app.processEvents()
