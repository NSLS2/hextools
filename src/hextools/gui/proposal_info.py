"""The active proposal's details, and a dialog to switch to another proposal."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QIntValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from hextools.gui._threading import FunctionWorker
from hextools.gui.live_re_md_viewer import QtReMetadataMonitor
from hextools.gui.run_engine_clients import RunEngineClient
from hextools.utils.proposals import Proposal, ProposalClient, ProposalListing

_PLACEHOLDER = "\u2014"  # em dash


class QtProposalInfo(QtReMetadataMonitor):
    """The active proposal, from the RunEngine metadata, and a way to change it.

    Parameters
    ----------
    re_client : RunEngineClient
        Whose metadata holds the active proposal.
    proposal_client : ProposalClient
        Lists the proposals offered by **Change Proposal…**, and looks them up.
    parent : QWidget, optional
        Parent widget.
    poll_period : float, optional
        Interval between metadata requests, in seconds. Default: 1.0.
    """

    def __init__(
        self,
        re_client: RunEngineClient,
        proposal_client: ProposalClient,
        parent: QWidget | None = None,
        *,
        poll_period: float = 1.0,
    ):
        super().__init__(
            re_client,
            {
                "ID": "data_session",
                "Cycle": "cycle",
                "Type": "proposal/type",
                "PI Name": "proposal/pi_name",
                "Title": "proposal/title",
            },
            poll_period=poll_period,
            parent=parent,
            title="Proposal Info",
            columns=2,
        )
        self._proposal_client = proposal_client
        # Spare width goes to the title, which is cut short to fit.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._switch_worker = None
        self._change_button = QPushButton("Change Proposal\u2026")
        self._change_button.clicked.connect(self.change_proposal)
        self._group_layout.addStretch(1)
        self._group_layout.addWidget(self._change_button)

    def _switch(self, proposal: int, username: str, password: str):
        return

    def change_proposal(self):
        """Open the dialog to pick a proposal and sign in, then switch to it."""
        if self._switch_worker is not None:
            return
        dialog = QtChangeProposalDialog(self._proposal_client, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        proposal, username, password = dialog.values()
        self._change_button.setEnabled(False)
        self._change_button.setText("Changing proposal\u2026")
        self._switch_worker = FunctionWorker(self._switch, proposal, username, password)
        self._switch_worker.returned.connect(self._on_switch_returned)
        self._switch_worker.errored.connect(self._on_switch_errored)
        self._switch_worker.start()

    def _reset_button(self):
        self._switch_worker = None
        self._change_button.setEnabled(True)
        self._change_button.setText("Change Proposal\u2026")

    def _on_switch_returned(self, _md):
        self._reset_button()
        self._start_worker()

    def _on_switch_errored(self, error: Exception):
        self._reset_button()
        QMessageBox.critical(
            self, "Couldn't change proposal", f"{type(error).__name__}: {error}"
        )


class QtChangeProposalDialog(QDialog):
    """Pick a beamline proposal, see its details, and sign in to switch to it.

    Parameters
    ----------
    proposal_client : ProposalClient
        Lists the proposals offered, and looks up the selected one.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(self, proposal_client: ProposalClient, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Change Proposal")
        self.setMinimumWidth(420)
        self._proposal_client = proposal_client
        self._list_worker = None
        self._info_workers: set[FunctionWorker] = set()
        self._listing = ProposalListing()

        # Editable, so a proposal missing from the list can still be typed in.
        self._proposal = QComboBox()
        self._proposal.setEditable(True)
        self._proposal.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._proposal.setValidator(QIntValidator(1, 999_999_999, self))
        # Long titles widen the popup, not the dialog.
        self._proposal.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._proposal.setMinimumContentsLength(10)
        self._proposal.lineEdit().setPlaceholderText("Loading proposals\u2026")  # type: ignore
        self._proposal.currentTextChanged.connect(self._on_proposal_changed)
        self._proposal.activated.connect(self._on_activated)
        self._commissioning = QCheckBox("Commissioning")
        self._commissioning.setToolTip(
            "List the beamline's commissioning proposals instead of user proposals."
        )
        self._commissioning.toggled.connect(self._populate)

        self._title = QLabel(_PLACEHOLDER)
        self._title.setWordWrap(True)
        self._type = QLabel(_PLACEHOLDER)
        self._pi = QLabel(_PLACEHOLDER)
        details = QFormLayout()
        details.addRow("Title:", self._title)
        details.addRow("Type:", self._type)
        details.addRow("PI:", self._pi)
        details_box = QGroupBox("Proposal")
        details_box.setLayout(details)

        self._username = QLineEdit()
        self._password = QLineEdit()
        self._password.setEchoMode(QLineEdit.EchoMode.Password)
        sign_in = QFormLayout()
        sign_in.addRow("Username:", self._username)
        sign_in.addRow("Password:", self._password)
        sign_in_box = QGroupBox("Sign in")
        sign_in_box.setLayout(sign_in)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._proposal.currentTextChanged.connect(self._update_ok)
        for edit in (self._username, self._password):
            edit.textChanged.connect(self._update_ok)
        self._update_ok()

        top = QFormLayout()
        proposal_row = QHBoxLayout()
        proposal_row.addWidget(self._proposal, stretch=1)
        proposal_row.addWidget(self._commissioning)
        top.addRow("Proposal number:", proposal_row)
        vbox = QVBoxLayout(self)
        vbox.addLayout(top)
        vbox.addWidget(details_box)
        vbox.addWidget(sign_in_box)
        vbox.addWidget(self._buttons)

        self._list_worker = FunctionWorker(proposal_client.list_proposals)
        self._list_worker.returned.connect(self._on_proposals_listed)
        self._list_worker.errored.connect(self._on_proposals_errored)
        self._list_worker.start()

    def _on_proposals_listed(self, listing: ProposalListing):
        self._list_worker = None
        self._listing = listing
        self._populate()
        self._proposal.lineEdit().setPlaceholderText("Select or type a proposal number")  # type: ignore

    def _populate(self):
        """Fill the dropdown with commissioning or user proposals, per the toggle."""
        proposals = (
            self._listing.commissioning
            if self._commissioning.isChecked()
            else self._listing.user
        )
        self._proposal.blockSignals(True)
        self._proposal.clear()
        for i, proposal in enumerate(proposals):
            summary = " ".join(proposal.title.split())
            self._proposal.addItem(
                f"{proposal.number} \u2014 {summary}" if summary else proposal.number,
                proposal.number,
            )
            self._proposal.setItemData(i, proposal.title, Qt.ItemDataRole.ToolTipRole)
        view = self._proposal.view()
        view.setMinimumWidth(min(view.sizeHintForColumn(0) + 24, 900))
        self._proposal.setCurrentIndex(-1)
        self._proposal.blockSignals(False)
        self._on_proposal_changed("")
        self._update_ok()

    def _on_activated(self, index: int):
        # Items read "number — title"; keep only the number in the edit field.
        number = self._proposal.itemData(index)
        if number:
            self._proposal.setEditText(number)

    def _on_proposals_errored(self, error: Exception):
        self._list_worker = None
        self._proposal.lineEdit().setPlaceholderText(  # type: ignore
            "Couldn't load proposals; type a number"
        )

    def _on_proposal_changed(self, text: str):
        for label in (self._title, self._type, self._pi):
            label.setText(_PLACEHOLDER)
        if not text.isdigit():
            return
        self._title.setText("Loading\u2026")
        worker = FunctionWorker(self._proposal_client.get_proposal, text)
        worker.returned.connect(
            lambda proposal, number=text: self._on_proposal(number, proposal)
        )
        worker.errored.connect(
            lambda error, number=text: self._on_info_error(number, error)
        )
        # Keep each worker referenced until it finishes, even if the selection moves on.
        self._info_workers.add(worker)
        worker.finished.connect(lambda w=worker: self._info_workers.discard(w))
        worker.start()

    def _on_proposal(self, number: str, proposal: Proposal):
        # Ignore answers for a proposal that is no longer selected.
        if number != self._proposal.currentText():
            return
        self._title.setText(proposal.title or _PLACEHOLDER)
        self._type.setText(proposal.type or _PLACEHOLDER)
        self._pi.setText(proposal.pi_name or _PLACEHOLDER)
        self._fit_height()

    def _on_info_error(self, number: str, error: Exception):
        if number == self._proposal.currentText():
            self._title.setText(f"Couldn't fetch proposal: {error}")
            self._fit_height()

    def _fit_height(self):
        """Grow the dialog so a wrapped or multi-line title isn't cut off."""
        layout = self.layout()
        layout.activate()  # type: ignore
        if layout.hasHeightForWidth():  # type: ignore
            height = layout.totalHeightForWidth(self.width())  # type: ignore
        else:
            height = layout.totalSizeHint().height()  # type: ignore
        if height > self.height():
            self.resize(self.width(), height)

    def _update_ok(self):
        filled = self._proposal.currentText().isdigit() and all(
            e.text() for e in (self._username, self._password)
        )
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(filled)

    def values(self) -> tuple[int, str, str]:
        """Return the selected proposal number, username, and password."""
        return (
            int(self._proposal.currentText()),
            self._username.text().strip(),
            self._password.text(),
        )
