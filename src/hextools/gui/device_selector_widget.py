"""A Qt widget for selecting any number of distinct devices from a list."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class DynamicDeviceSelector(QWidget):
    """Select any number of distinct devices via growable dropdown rows.

    Each row is a dropdown offering only the devices not already chosen in
    another row. The first row cannot be removed; later rows show a ``-`` button
    that removes the whole row. A ``+`` button on the last row appends another
    row and is hidden once every available device has been selected.
    """

    selection_changed = Signal(object)

    def __init__(self, options: Sequence[str] | None = None, parent=None):
        super().__init__(parent)
        self._options: list[str] = list(options or [])
        self._rows: list[dict] = []
        self._updating = False

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(4)
        self.setLayout(self._rows_layout)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self._add_row()
        self._refresh()

    def _make_button(self, text: str) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setFixedWidth(24)
        return button

    def _add_row(self, select: str | None = None) -> dict:
        combo = QComboBox()
        plus = self._make_button("+")
        minus = self._make_button("\u2212")

        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(4)
        hbox.addWidget(combo, 1)
        hbox.addWidget(plus)
        hbox.addWidget(minus)

        row = {"widget": container, "combo": combo, "plus": plus, "minus": minus}
        self._rows.append(row)
        self._rows_layout.addWidget(container)

        self._populate_combo(row, select)
        combo.currentIndexChanged.connect(self._on_combo_changed)
        plus.clicked.connect(self._on_add_clicked)
        minus.clicked.connect(lambda *_, r=row: self._remove_row(r))
        return row

    def _remove_row(self, row: dict):
        if len(self._rows) <= 1:
            return
        self._rows.remove(row)
        row["widget"].setParent(None)
        row["widget"].deleteLater()
        self._refresh()

    def _selected_except(self, exclude: dict | None) -> set[str]:
        chosen = set()
        for row in self._rows:
            if row is exclude:
                continue
            value = row["combo"].currentData()
            if value is not None:
                chosen.add(value)
        return chosen

    def _remaining(self) -> list[str]:
        chosen = self._selected_except(None)
        return [name for name in self._options if name not in chosen]

    def _populate_combo(self, row: dict, select: str | None = None):
        combo = row["combo"]
        current = select if select is not None else combo.currentData()
        others = self._selected_except(row)

        combo.blockSignals(True)
        combo.clear()
        for name in self._options:
            if name not in others:
                combo.addItem(name, userData=name)
        index = combo.findData(current)
        if index < 0 and combo.count():
            index = 0
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _refresh(self):
        if self._updating:
            return
        self._updating = True
        try:
            for row in self._rows:
                self._populate_combo(row)
            has_more = bool(self._remaining())
            last = len(self._rows) - 1
            for i, row in enumerate(self._rows):
                row["plus"].setVisible(i == last and has_more)
                row["minus"].setVisible(i > 0)
        finally:
            self._updating = False
        self.selection_changed.emit(self.selected_devices())

    def _on_combo_changed(self, *_):
        self._refresh()

    def _on_add_clicked(self, *_):
        remaining = self._remaining()
        if not remaining:
            return
        self._add_row(select=remaining[0])
        self._refresh()

    def selected_devices(self) -> list[str]:
        """Devices chosen across the rows, in row order (no duplicates)."""
        result: list[str] = []
        for row in self._rows:
            value = row["combo"].currentData()
            if value is not None and value not in result:
                result.append(value)
        return result

    def set_available(self, options: Sequence[str]):
        """Update the available devices, dropping rows that no longer apply."""
        self._options = list(options)
        kept = []
        for row in self._rows:
            value = row["combo"].currentData()
            if value is None or value in self._options:
                kept.append(row)
            else:
                row["widget"].setParent(None)
                row["widget"].deleteLater()
        self._rows = kept
        if not self._rows:
            self._add_row()
        self._refresh()

    def set_selected_devices(self, names: Iterable[str]):
        """Replace the rows so that exactly ``names`` (if available) are chosen."""
        wanted = [name for name in names if name in self._options]
        for row in self._rows:
            row["widget"].setParent(None)
            row["widget"].deleteLater()
        self._rows = []
        for name in wanted:
            self._add_row(select=name)
        if not self._rows:
            self._add_row()
        self._refresh()
