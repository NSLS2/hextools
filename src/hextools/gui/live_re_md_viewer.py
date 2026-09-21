"""Qt widgets that display live Run Engine metadata.

Two sources of RE metadata are supported, sharing a common base class that owns
the UI and polling loop:

* :class:`QtReMetadataMonitor` polls the Queue Server ``re_metadata`` API. Use
  this when driving the beamline through the Queue Server.
* :class:`QtReMetadataMonitorLocal` reads ``RE.md`` from a RunEngine obtained
  from the IPython namespace. Use this when running acquisition in-process.

Display is best effort: a missing key, a closed environment, an absent
RunEngine, or a failed request renders as a placeholder rather than raising.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from bluesky import RunEngine
from bluesky_widgets.qt.threading import FunctionWorker
from qtpy.QtCore import Qt, QTimer, Signal, Slot
from qtpy.QtWidgets import QFormLayout, QGroupBox, QLabel, QVBoxLayout, QWidget
from bluesky_widgets.models.run_engine_client import RunEngineClient

RunEngineMetadata = MutableMapping[str, Any]

_PLACEHOLDER = "\u2014"  # em dash


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
        re: RunEngineClient | RunEngine,
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
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._labels[key] = label
            form.addRow(f"{self._key_labels[i]}:", label)

        group_box = QGroupBox(title)
        group_box.setLayout(form)

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
                response = self._re._client.re_metadata()  # noqa: SLF001
            except Exception:
                return {}
            if not response.get("success", False):
                return {}
            return response.get("re_metadata") or {}
        elif isinstance(self._re, RunEngine):
            return self._re.md

        raise RuntimeError("RE Metadata Viewer widget must be initialized with a RunEngineClient or RunEngine instance.")

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
                if isinstance(value, dict) and k in value:
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
    """Widget to display proposal information from the RunEngine metadata."""

    def __init__(
        self,
        re: RunEngineClient | RunEngine,
        parent: QWidget | None = None,
        *,
        poll_period: float = 1.0,
    ):

        super().__init__(
            re,
            {
                "Proposal ID": "data_session",
                "Cycle": "cycle",
                "Proposal Title": "proposal/title",
                "Proposal Type": "proposal/type",
                "PI Name": "proposal/pi_name"
            },
            poll_period=poll_period,
            parent=parent,
            title="Proposal Info"
        )