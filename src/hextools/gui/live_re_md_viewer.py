"""A Qt widget that displays live Run Engine metadata from the Queue Server.

The widget polls the Queue Server ``re_metadata`` API and shows the current
value of a fixed set of RE metadata keys. Display is best effort: a missing key,
a closed environment, or a failed request renders as a placeholder rather than
raising.
"""

from __future__ import annotations

from collections.abc import Sequence

from bluesky_widgets.qt.threading import FunctionWorker
from qtpy.QtCore import Qt, QTimer, Signal, Slot
from qtpy.QtWidgets import QFormLayout, QGroupBox, QLabel, QVBoxLayout, QWidget

_PLACEHOLDER = "\u2014"  # em dash


class QtReMetadataMonitor(QWidget):
    """Display current values of selected RE metadata keys (best effort).

    Parameters
    ----------
    model : bluesky_widgets.models.run_engine_client.RunEngineClient
        The Queue Server client model (e.g. ``viewer.run_engine``).
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
        model,
        keys: Sequence[str],
        parent=None,
        *,
        poll_period: float = 1.0,
        title: str = "RE Metadata",
    ):
        super().__init__(parent)
        self.model = model
        self._keys = list(keys)
        self._worker = None

        self._labels: dict[str, QLabel] = {}
        form = QFormLayout()
        form.setContentsMargins(8, 6, 8, 6)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        for key in self._keys:
            label = QLabel(_PLACEHOLDER)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._labels[key] = label
            form.addRow(f"{key}:", label)

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

    def _start_worker(self):
        # Skip if a request is still in flight to avoid overlapping polls.
        if self._worker is not None:
            return
        self._worker = FunctionWorker(self._request_metadata)
        self._worker.returned.connect(self._on_worker_returned)
        self._worker.start()

    def _request_metadata(self) -> dict:
        """Return the RE metadata dict, or an empty dict on any failure."""
        try:
            response = self.model._client.re_metadata()  # noqa: SLF001
        except Exception:
            return {}
        if not response.get("success", False):
            return {}
        return response.get("re_metadata") or {}

    def _on_worker_returned(self, metadata):
        self._worker = None
        self.signal_metadata_updated.emit(metadata)

    @Slot(object)
    def _slot_metadata_updated(self, metadata: dict):
        for key, label in self._labels.items():
            if key in metadata:
                label.setText(str(metadata[key]))
            else:
                label.setText(_PLACEHOLDER)

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)
