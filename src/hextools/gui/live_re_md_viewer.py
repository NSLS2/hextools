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

from qtpy.QtCore import Qt, QTimer, Signal, Slot
from qtpy.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from hextools.gui._elided_label import ElidedLabel
from hextools.gui._threading import FunctionWorker
from hextools.gui.run_engine_clients import RunEngineClient

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
    columns : int, optional
        Keys shown side by side in each row; the last key spans the rest of its
        row, cut short with "…" if it doesn't fit. Default: 1.
    """

    signal_metadata_updated = Signal(object)

    def __init__(
        self,
        re_client: RunEngineClient,
        keys: Sequence[str] | dict[str, str],
        parent: QWidget | None = None,
        *,
        poll_period: float = 1.0,
        title: str = "RE Metadata",
        columns: int = 1,
    ):
        super().__init__(parent)
        self._re_client = re_client
        self._keys = list(keys) if isinstance(keys, Sequence) else list(keys.values())
        self._key_labels = list(keys.keys()) if isinstance(keys, dict) else keys
        self._worker = None

        self._labels: dict[str, QLabel] = {}
        grid = QGridLayout()
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        for i, key in enumerate(self._keys):
            row, column = divmod(i, columns)
            last = i == len(self._keys) - 1
            label = (
                ElidedLabel(_PLACEHOLDER)
                if last and columns > 1
                else QLabel(_PLACEHOLDER)
            )
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._labels[key] = label
            # Each key takes a label and a value column, then a gap column.
            span = 3 * (columns - column) - 2 if last else 1
            grid.addWidget(QLabel(f"{self._key_labels[i]}:"), row, 3 * column)
            grid.addWidget(label, row, 3 * column + 1, 1, span)
        for column in range(columns):
            grid.setColumnStretch(3 * column + 1, 1)
            if column < columns - 1:
                grid.setColumnMinimumWidth(3 * column + 2, 12)

        group_box = QGroupBox(title)
        self._group_layout = QVBoxLayout(group_box)
        self._group_layout.setContentsMargins(0, 0, 0, 0)
        self._group_layout.addLayout(grid)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(group_box)
        self.setLayout(vbox)

        self.signal_metadata_updated.connect(self._slot_metadata_updated)

        self._timer = QTimer(self)
        self._timer.setInterval(int(poll_period * 1000))
        self._timer.timeout.connect(self._start_worker)
        self._timer.start()

        # Kick off an immediate refresh instead of waiting for the first tick.
        self._start_worker()

    def _start_worker(self):
        # Skip if a request is still in flight to avoid overlapping polls.
        if self._worker is not None:
            return
        self._worker = FunctionWorker(self._re_client.metadata)
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
