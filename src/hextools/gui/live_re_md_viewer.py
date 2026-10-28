"""Qt widgets that display live Run Engine metadata.

Two sources of RE metadata are supported by the same widget:

* the Queue Server's ``re_metadata`` API, when driving the beamline through the
  Queue Server;
* the IPython session's ``RE.md``, which the session sends the GUI whenever it
  changes (:class:`~hextools.gui.session_client.SessionClient`).

Display is best effort: a missing key, a closed environment, an absent
RunEngine, or a failed request renders as a placeholder rather than raising.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from bluesky_widgets.models.run_engine_client import RunEngineClient
from bluesky_widgets.qt.threading import FunctionWorker
from qtpy.QtCore import Qt, QTimer, Signal, Slot
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
    QVBoxLayout,
    QWidget,
)

from hextools.gui.session_client import SessionClient
from hextools.gui.spec import ProposalBackend

RunEngineMetadata = MutableMapping[str, Any]

_PLACEHOLDER = "\u2014"  # em dash
# Keys a proposal switch sets in RE.md.
_PROPOSAL_KEYS = ("data_session", "username", "start_datetime", "proposal")


class QtReMetadataMonitor(QWidget):
    """Poll a source of RE metadata and display selected keys (best effort).

    Subclasses implement :meth:`_fetch_metadata` to return the current metadata
    dict. The base class owns the label grid, the polling timer, and the
    off-thread refresh worker.

    Parameters
    ----------
    keys : Sequence[str]
        RE metadata keys to display. The order is preserved.
    parent : QWidget, optional
        Parent widget.
    poll_period : float, optional
        Interval between metadata requests, in seconds. Default: 1.0.
    title : str, optional
        Group box title. Default: ``"RE Metadata"``.
    """

    signal_metadata_updated = Signal(object)

    def __init__(
        self,
        re: RunEngineClient | SessionClient,
        keys: Sequence[str] | dict[str, str],
        parent: QWidget | None = None,
        *,
        poll_period: float = 1.0,
        title: str = "RE Metadata",
    ):
        super().__init__(parent)
        self._re = re
        self._keys = list(keys) if isinstance(keys, Sequence) else list(keys.values())
        self._key_labels = list(keys.keys()) if isinstance(keys, dict) else keys
        self._worker = None

        self._labels: dict[str, QLabel] = {}
        form = QFormLayout()
        form.setContentsMargins(8, 6, 8, 6)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        for i, key in enumerate(self._keys):
            label = QLabel(_PLACEHOLDER)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._labels[key] = label
            form.addRow(f"{self._key_labels[i]}:", label)

        group_box = QGroupBox(title)
        self._group_layout = QVBoxLayout(group_box)
        self._group_layout.setContentsMargins(0, 0, 0, 0)
        self._group_layout.addLayout(form)
        self._form = form

        vbox = QVBoxLayout()
        vbox.addWidget(group_box)
        self.setLayout(vbox)

        self.signal_metadata_updated.connect(self._slot_metadata_updated)

        self._timer = QTimer(self)
        self._timer.setInterval(int(poll_period * 1000))
        self._timer.timeout.connect(self._start_worker)
        self._timer.start()

        # Kick off an immediate refresh instead of waiting for the first tick.
        self._start_worker()

    def _fetch_metadata(self) -> RunEngineMetadata:
        """Return the current RE metadata dict, or an empty dict on failure."""
        if isinstance(self._re, RunEngineClient):
            try:
                # Not in bluesky-queueserver-api 0.0.13; needs a newer release.
                response = self._re._client.re_metadata()  # noqa: SLF001  # type: ignore
            except Exception:
                return {}
            if not response.get("success", False):
                return {}
            return response.get("re_metadata") or {}
        elif isinstance(self._re, SessionClient):
            return dict(self._re.md)

        raise RuntimeError(
            "RE Metadata Viewer widget must be initialized with a RunEngineClient "
            "or SessionClient."
        )

    def _start_worker(self):
        # Skip if a request is still in flight to avoid overlapping polls.
        if self._worker is not None:
            return
        self._worker = FunctionWorker(self._fetch_metadata)
        self._worker.returned.connect(self._on_worker_returned)
        self._worker.start()

    def _on_worker_returned(self, metadata):
        self._worker = None
        self.signal_metadata_updated.emit(metadata)

    @Slot(object)
    def _slot_metadata_updated(self, metadata: dict):

        def get_nested_value(metadata: dict, key: str):
            keys = key.split("/")
            value = metadata
            for k in keys:
                if isinstance(value, Mapping) and k in value:
                    value = value[k]
                else:
                    return None
            return value

        for key, label in self._labels.items():
            value = get_nested_value(metadata, key)
            if value is not None:
                label.setText(str(value))
            else:
                label.setText(_PLACEHOLDER)

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)


class QtProposalInfo(QtReMetadataMonitor):
    """Widget to display proposal information from the RunEngine metadata.

    With a ``proposals`` backend it also offers Change Proposal.
    """

    def __init__(
        self,
        re: RunEngineClient | SessionClient,
        parent: QWidget | None = None,
        *,
        proposals: ProposalBackend | None = None,
        poll_period: float = 1.0,
    ):

        super().__init__(
            re,
            {
                "Proposal ID": "data_session",
                "Cycle": "cycle",
                "Proposal Title": "proposal/title",
                "Proposal Type": "proposal/type",
                "PI Name": "proposal/pi_name",
            },
            poll_period=poll_period,
            parent=parent,
            title="Proposal Info",
        )
        self._proposals = proposals
        self._switch_worker = None
        self._change_button = QPushButton("Change Proposal…")
        self._change_button.clicked.connect(self.change_proposal)
        self._change_button.setVisible(proposals is not None)
        self._group_layout.addStretch(1)
        self._group_layout.addWidget(self._change_button)

    def _switch(self, proposal: int, username: str, password: str):
        backend = self._proposals
        if backend is None:
            raise RuntimeError("No proposal system is configured for this GUI.")
        if isinstance(self._re, SessionClient):
            # The changes reach the session's RE.md in _on_switch_returned.
            return backend.switch_proposal(
                proposal, username, password, md=dict(self._re.md)
            )
        client = self._re._client  # noqa: SLF001
        # Not in bluesky-queueserver-api 0.0.13; needs a newer release.
        response = client.re_metadata()  # type: ignore
        if not response.get("success", False):
            raise RuntimeError(
                response.get("msg") or "Couldn't read the RunEngine metadata."
            )
        md = dict(response.get("re_metadata") or {})
        updated = backend.switch_proposal(proposal, username, password, md=md)
        if not backend.md_is_shared():
            changes = {k: updated[k] for k in _PROPOSAL_KEYS if k in updated}
            client.script_upload(f"RE.md.update({changes!r})")
        return updated

    def change_proposal(self):
        """Open the dialog to pick a proposal and sign in, then switch to it."""
        if self._switch_worker is not None or self._proposals is None:
            return
        dialog = QtChangeProposalDialog(self._proposals, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        proposal, username, password = dialog.values()
        self._change_button.setEnabled(False)
        self._change_button.setText("Changing proposal…")
        self._switch_worker = FunctionWorker(self._switch, proposal, username, password)
        self._switch_worker.returned.connect(self._on_switch_returned)
        self._switch_worker.errored.connect(self._on_switch_errored)
        self._switch_worker.start()

    def _reset_button(self):
        self._switch_worker = None
        self._change_button.setEnabled(True)
        self._change_button.setText("Change Proposal…")

    def _on_switch_returned(self, updated):
        session = self._re
        backend = self._proposals
        if (
            isinstance(session, SessionClient)
            and backend is not None
            and not backend.md_is_shared()
        ):
            changes = {k: updated[k] for k in _PROPOSAL_KEYS if k in updated}
            session.execute(
                f"{session.re_name}.md.update({changes!r})", self._on_md_updated
            )
            return
        self._reset_button()
        self._start_worker()

    def _on_md_updated(self, error: BaseException | None):
        if error is not None:
            self._on_switch_errored(error)
            return
        self._reset_button()
        self._start_worker()

    def _on_switch_errored(self, error: BaseException):
        self._reset_button()
        name = getattr(error, "type_name", type(error).__name__)
        QMessageBox.critical(self, "Couldn't change proposal", f"{name}: {error}")


class QtChangeProposalDialog(QDialog):
    """Pick a beamline proposal, see its details, and sign in to switch to it."""

    def __init__(self, proposals: ProposalBackend, parent: QWidget | None = None):
        super().__init__(parent)
        self._backend = proposals
        self.setWindowTitle("Change Proposal")
        self.setMinimumWidth(420)
        self._list_worker = None
        self._info_workers: set[FunctionWorker] = set()
        self._lists: dict[str, list[tuple[str, str]]] = {}

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

        self._list_worker = FunctionWorker(self._backend.list_proposals)
        self._list_worker.returned.connect(self._on_proposals_listed)
        self._list_worker.errored.connect(self._on_proposals_errored)
        self._list_worker.start()

    def _on_proposals_listed(self, proposals: dict[str, list[tuple[str, str]]]):
        self._list_worker = None
        self._lists = proposals
        self._populate()
        self._proposal.lineEdit().setPlaceholderText("Select or type a proposal number")  # type: ignore

    def _populate(self):
        """Fill the dropdown with commissioning or user proposals, per the toggle."""
        kind = "commissioning" if self._commissioning.isChecked() else "user"
        self._proposal.blockSignals(True)
        self._proposal.clear()
        for i, (number, title) in enumerate(self._lists.get(kind, [])):
            summary = " ".join(title.split())
            self._proposal.addItem(
                f"{number} \u2014 {summary}" if summary else number, number
            )
            self._proposal.setItemData(i, title, Qt.ItemDataRole.ToolTipRole)
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
        worker = FunctionWorker(self._backend.get_proposal_info, text)
        worker.returned.connect(lambda info, number=text: self._on_info(number, info))
        worker.errored.connect(
            lambda error, number=text: self._on_info_error(number, error)
        )
        # Keep each worker referenced until it finishes, even if the selection moves on.
        self._info_workers.add(worker)
        worker.finished.connect(lambda w=worker: self._info_workers.discard(w))
        worker.start()

    def _on_info(self, number: str, info: dict):
        # Ignore answers for a proposal that is no longer selected.
        if number != self._proposal.currentText():
            return
        self._title.setText(info.get("title") or _PLACEHOLDER)
        self._type.setText(info.get("type") or _PLACEHOLDER)
        self._pi.setText(info.get("pi_name") or _PLACEHOLDER)
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
