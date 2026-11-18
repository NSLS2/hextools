"""Qt widget listing beamline devices and their connection status.

Devices and their connection status come from a RunEngine client's device
registry, and are listed again whenever it changes.
"""

from __future__ import annotations

from qtpy.QtCore import Slot
from qtpy.QtWidgets import (
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from hextools.gui._client_signals import QtCatalog
from hextools.gui.run_engine_clients import RunEngineClient

_CONNECTED_STYLE = "color: #2e7d32; font-weight: bold;"
_DISCONNECTED_STYLE = "color: #c62828; font-weight: bold;"


class QtAvailableDevices(QWidget):
    """Table of device name, type, and connection status.

    Parameters
    ----------
    re_client : RunEngineClient
        Whose device registry is listed.
    """

    def __init__(self, re_client: RunEngineClient, *, parent=None):
        super().__init__(parent)
        self._re_client = re_client
        self._rows: list[tuple[str, str]] = []
        self._status_labels: dict[str, QLabel] = {}

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)

        vbox = QVBoxLayout()
        vbox.addWidget(self._scroll)
        self.setLayout(vbox)

        self._catalog = QtCatalog(re_client, self)
        self._catalog.changed.connect(lambda *_: self._refresh())
        self._refresh()

    def _entries(self) -> list[tuple[str, str, bool]]:
        """``(label, type name, connected)`` for each row to show."""
        entries = []
        for name, node in self._re_client.device_registry.items():
            if node.signal:
                continue
            label = f"{name} ({', '.join(node.aliases)})" if node.aliases else name
            type_name = node.classes[0].rpartition(".")[2] if node.classes else ""
            entries.append((label, type_name, bool(node.connected)))
        return sorted(entries, key=lambda entry: entry[0])

    def _rebuild(self, rows: list[tuple[str, str]]):
        grid = QGridLayout()
        for col, header in enumerate(("Name", "Type", "Status")):
            grid.addWidget(QLabel(f"<b>{header}</b>"), 0, col)
        self._status_labels = {}
        for row, (label, type_name) in enumerate(rows, start=1):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(QLabel(type_name), row, 1)
            status = QLabel()
            self._status_labels[label] = status
            grid.addWidget(status, row, 2)
        grid.setColumnStretch(3, 1)
        grid.setRowStretch(len(rows) + 1, 1)
        content = QWidget()
        content.setLayout(grid)
        self._scroll.setWidget(content)  # deletes the previous content widget
        self._rows = rows

    @Slot()
    def _refresh(self):
        entries = self._entries()
        rows = [(label, type_name) for label, type_name, _ in entries]
        if rows != self._rows:
            self._rebuild(rows)
        for label, _, connected in entries:
            status = self._status_labels[label]
            status.setText("Connected" if connected else "Disconnected")
            status.setStyleSheet(_CONNECTED_STYLE if connected else _DISCONNECTED_STYLE)

    def closeEvent(self, event):
        self._catalog.stop()
        super().closeEvent(event)
