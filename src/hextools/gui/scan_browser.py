"""Browse finished runs in a Tiled catalog by spec and date, and act on them."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any

from bluesky import RunEngine
from qtpy.QtCore import QDate, QPoint, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDateEdit,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from tiled.queries import Key, SpecQuery
from tiled.structures.core import Spec

from hextools.gui._threading import create_worker

#: Called with the uid of the run it was invoked on.
ScanCallback = Callable[[str], Any]


@dataclass
class RunSummary:
    uid: str
    scan_id: int | None
    time: float
    specs: list[str]
    metadata: dict[str, Any]


def search_runs(
    client: Any, spec: Spec, since: float, until: float, data_session: str | None
) -> list[RunSummary]:
    """Find runs with ``spec`` started in ``[since, until)``, newest first."""
    results = client.search(SpecQuery(spec.name))
    results = results.search(Key("start.time") >= since).search(Key("start.time") < until)
    if data_session:
        results = results.search(Key("start.data_session") == data_session)
    runs = []
    for uid, run in results.items():
        metadata = dict(run.metadata)
        start = metadata.get("start", {})
        runs.append(
            RunSummary(
                uid=uid,
                scan_id=start.get("scan_id"),
                time=start.get("time", 0.0),
                specs=[s.name for s in getattr(run, "specs", [])],
                metadata=metadata,
            )
        )
    runs.sort(key=lambda r: r.time, reverse=True)
    return runs


class QtScanBrowser(QWidget):
    """Search a Tiled catalog for this proposal's runs by spec and date range.

    Selecting a run shows its metadata, plus a button for each callback attached
    (with :meth:`add_callback`) to one of the run's specs. The same callbacks are
    offered in the run's right-click menu.
    """

    _COLUMNS = ["Time", "Scan ID", "UID"]

    def __init__(self, tiled_client: Any, specs: list[Spec], re: RunEngine, parent=None):
        super().__init__(parent)
        self._client = tiled_client
        self._specs = list(specs)
        self._re = re
        self._callbacks: dict[str, list[tuple[str, ScanCallback]]] = {}
        self._runs: list[RunSummary] = []
        self._worker = None

        self._spec_combo = QComboBox()
        for spec in self._specs:
            self._spec_combo.addItem(spec.name)
        today = QDate.currentDate()
        self._since = QDateEdit(today)
        self._until = QDateEdit(today)
        for date_edit in (self._since, self._until):
            date_edit.setCalendarPopup(True)
            date_edit.setDisplayFormat("yyyy-MM-dd")
        self._search_button = QPushButton("Search")
        self._search_button.clicked.connect(self.search)
        self._status = QLabel()

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Spec:"))
        controls.addWidget(self._spec_combo)
        controls.addWidget(QLabel("From:"))
        controls.addWidget(self._since)
        controls.addWidget(QLabel("To:"))
        controls.addWidget(self._until)
        controls.addWidget(self._search_button)
        controls.addWidget(self._status, stretch=1)

        self._table = QTableWidget(0, len(self._COLUMNS))
        self._table.setHorizontalHeaderLabels(self._COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_context_menu)

        self._callback_bar = QHBoxLayout()
        self._metadata = QTreeWidget()
        self._metadata.setHeaderLabels(["Key", "Value"])
        self._metadata.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addLayout(self._callback_bar)
        right_layout.addWidget(self._metadata, stretch=1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._table)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(splitter, stretch=1)

    def add_callback(self, spec: Spec, label: str, callback: ScanCallback) -> None:
        """Offer ``callback`` (called with the run's uid) for runs with ``spec``."""
        self._callbacks.setdefault(spec.name, []).append((label, callback))
        self._on_selection_changed()

    def search(self) -> None:
        """Search the catalog in a background thread using the current filters."""
        if self._worker is not None:
            return
        spec = self._specs[self._spec_combo.currentIndex()]
        since_date = self._since.date().toPython()
        until_date = self._until.date().toPython()
        if since_date > until_date:
            self._status.setText("'From' must not be after 'To'.")
            return
        since = datetime.combine(since_date, time.min).timestamp()
        until = datetime.combine(until_date + timedelta(days=1), time.min).timestamp()
        data_session = self._re.md.get("data_session")

        self._search_button.setEnabled(False)
        self._status.setText("Searching…")
        self._worker = create_worker(
            search_runs,
            self._client,
            spec,
            since,
            until,
            data_session,
            _connect={
                "returned": self._on_results,
                "errored": self._on_search_error,
                "finished": self._on_search_finished,
            },
        )

    def _on_results(self, runs: list[RunSummary]) -> None:
        self._runs = runs
        self._table.setRowCount(0)
        for run in runs:
            row = self._table.rowCount()
            self._table.insertRow(row)
            values = [
                datetime.fromtimestamp(run.time).strftime("%Y-%m-%d %H:%M:%S"),
                "" if run.scan_id is None else str(run.scan_id),
                run.uid,
            ]
            for col, value in enumerate(values):
                self._table.setItem(row, col, QTableWidgetItem(value))
        self._status.setText(f"{len(runs)} run(s) found.")
        self._on_selection_changed()

    def _on_search_error(self, error: Exception) -> None:
        self._status.setText(f"Search failed: {error}")

    def _on_search_finished(self) -> None:
        self._worker = None
        self._search_button.setEnabled(True)

    def _selected_run(self) -> RunSummary | None:
        rows = self._table.selectionModel().selectedRows()
        return self._runs[rows[0].row()] if rows else None

    def _callbacks_for(self, run: RunSummary) -> list[tuple[str, ScanCallback]]:
        return [cb for spec in run.specs for cb in self._callbacks.get(spec, [])]

    def _on_selection_changed(self) -> None:
        while self._callback_bar.count():
            item = self._callback_bar.takeAt(0)
            if item is not None and item.widget() is not None:
                item.widget().deleteLater()
        self._metadata.clear()

        run = self._selected_run()
        if run is None:
            return
        for label, callback in self._callbacks_for(run):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, cb=callback, uid=run.uid: self._invoke(cb, uid))
            self._callback_bar.addWidget(button)
        self._callback_bar.addStretch()
        _fill_tree(self._metadata.invisibleRootItem(), run.metadata)
        for i in range(self._metadata.topLevelItemCount()):
            self._metadata.topLevelItem(i).setExpanded(True)

    def _on_context_menu(self, pos: QPoint) -> None:
        row = self._table.rowAt(pos.y())
        if row < 0:
            return
        self._table.selectRow(row)
        run = self._runs[row]
        menu = QMenu(self)
        callbacks = self._callbacks_for(run)
        for label, callback in callbacks:
            menu.addAction(label, lambda cb=callback, uid=run.uid: self._invoke(cb, uid))
        if not callbacks:
            menu.addAction("No actions for this run").setEnabled(False)
        menu.exec(self._table.viewport().mapToGlobal(pos))

    def _invoke(self, callback: ScanCallback, uid: str) -> None:
        try:
            callback(uid)
        except Exception as ex:
            QMessageBox.critical(self, "Action failed", f"{type(ex).__name__}: {ex}")


def _fill_tree(parent: QTreeWidgetItem, value: Any) -> None:
    items = value.items() if isinstance(value, dict) else enumerate(value)
    for key, child in items:
        item = QTreeWidgetItem(parent, [str(key)])
        if isinstance(child, (dict, list)) and child:
            _fill_tree(item, child)
        else:
            item.setText(1, child if isinstance(child, str) else json.dumps(child, default=str))
