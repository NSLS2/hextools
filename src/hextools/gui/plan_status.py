"""Widgets showing the running plan, its natural-language narration, and history."""

from __future__ import annotations

import enum
import html
import logging
from datetime import datetime
from typing import Any

from bluesky.callbacks.best_effort import hinted_fields

# TODO: Restore with QtPlanProgressView.
# from bluesky.callbacks.best_effort import BestEffortCallback
# from matplotlib.backends.backend_qt import NavigationToolbar2QT
# from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
# from matplotlib.figure import Figure
from qtpy.QtCore import Qt, QTimer, Slot
from qtpy.QtGui import QColor, QFont, QFontDatabase, QPainter, QPen
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hextools.gui._client_signals import (
    QtCurrentPlan,
    QtLogRecords,
    QtPlanHistoryChanges,
    QtRunEngineState,
)
from hextools.gui.log_watcher import level_color
from hextools.gui.plan_errors import show_plan_error
from hextools.gui.run_engine_clients import HistoryItem, PlanInfo, RunEngineClient

_NO_PLAN = "\u2014"  # em dash
_NOT_APPLICABLE = "N/A"

logger = logging.getLogger(__name__)


def _monospace_font() -> QFont:
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)


_STATE_COLORS = {
    "running": QColor("#2e9e44"),
    "paused": QColor("#e0a000"),
    "panicked": QColor("#d32f2f"),
}
_TRANSITION_COLOR = QColor(
    "#e0a000"
)  # pausing, stopping, aborting, halting, suspending


class _RunEngineStateIndicator(QWidget):
    """Spinning arc while the RunEngine works, pause bars when paused.

    Nothing is drawn when idle.
    """

    def __init__(self, size: int = 18, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._state = "idle"
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._advance)

    def set_state(self, state: str):
        self._state = state
        spinning = state not in ("idle", "paused", "panicked")
        if spinning and not self._timer.isActive():
            self._timer.start()
        elif not spinning:
            self._timer.stop()
        self.update()

    def _advance(self):
        self._angle = (self._angle + 12) % 360
        self.update()

    def paintEvent(self, event):
        if self._state == "idle":
            return
        color = _STATE_COLORS.get(self._state, _TRANSITION_COLOR)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(color, 2.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        rect = self.rect().adjusted(3, 3, -3, -3)
        if self._state == "paused":
            x1, x2 = rect.left() + 2, rect.right() - 2
            painter.drawLine(x1, rect.top(), x1, rect.bottom())
            painter.drawLine(x2, rect.top(), x2, rect.bottom())
        elif self._state == "panicked":
            painter.drawEllipse(rect)
        else:
            painter.drawArc(rect, -self._angle * 16, 270 * 16)


class QtPlanStatus(QWidget):
    """Compact summary: plan name, scan ids, and the latest INFO-or-above log message.

    When the plan runs another plan that opens a run under a different name, that inner
    plan and the current run's scan id are shown too, updating with each new run.
    Log messages are the RunEngine client's.
    """

    def __init__(self, re_client: RunEngineClient, parent=None):
        super().__init__(parent)
        self._plan = QLabel(_NO_PLAN)
        self._inner_plan = QLabel(_NO_PLAN)
        self._scan_id = QLabel(_NO_PLAN)
        self._scan_id.setWordWrap(True)
        self._inner_scan_id = QLabel(_NO_PLAN)
        self._last = QLabel(_NO_PLAN)
        self._last.setWordWrap(True)
        self._last.setMinimumWidth(260)
        # Wrap to the available width rather than widening the window.
        self._last.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self._outer_name = ""
        self._start_time: float | None = None

        self._indicator = _RunEngineStateIndicator()
        self._state = QLabel()
        state_row = QHBoxLayout()
        state_row.setContentsMargins(0, 0, 0, 0)
        state_row.addWidget(self._indicator)
        state_row.addWidget(self._state, stretch=1)

        form = QFormLayout()
        form.setContentsMargins(8, 6, 8, 6)
        form.addRow("RunEngine:", state_row)
        form.addRow("Plan:", self._plan)
        form.addRow("Inner plan:", self._inner_plan)
        form.addRow("Scan ID:", self._scan_id)
        form.addRow("Inner scan ID:", self._inner_scan_id)
        form.addRow(QLabel("Last step:"))
        form.addRow(self._last)
        self._form = form
        self._set_inner_visible(False)

        group_box = QGroupBox("Plan Status")
        group_box.setLayout(form)
        vbox = QVBoxLayout()
        vbox.addWidget(group_box)
        self.setLayout(vbox)

        self._re_state = QtRunEngineState(re_client, self)
        self._re_state.changed.connect(self._on_state_changed)
        self._current_plan = QtCurrentPlan(re_client, self)
        self._current_plan.changed.connect(self._on_plan_changed)
        self._on_state_changed(re_client.state)
        self._on_plan_changed(re_client.current_plan())
        self._logs = QtLogRecords(re_client, self)
        self._logs.record.connect(self._on_record)

    def _set_inner_visible(self, visible: bool):
        self._form.setRowVisible(self._inner_plan, visible)
        self._form.setRowVisible(self._inner_scan_id, visible)

    @Slot(object)
    def _on_plan_changed(self, plan: PlanInfo | None):
        # Keep showing the last plan once it finishes.
        if plan is None:
            return
        if plan.start_time != self._start_time or plan.name != self._outer_name:
            self._start_time = plan.start_time
            self._outer_name = plan.name
            self._last.setText(_NO_PLAN)
        self._plan.setText(plan.name)
        scan_ids = [
            _NO_PLAN if r.scan_id is None else str(r.scan_id) for r in plan.runs
        ]
        self._scan_id.setText(", ".join(scan_ids) or _NO_PLAN)
        inner = [r for r in plan.runs if r.plan_name and r.plan_name != plan.name]
        if inner:
            run = inner[-1]
            self._inner_plan.setText(run.plan_name or _NO_PLAN)
            self._inner_scan_id.setText(
                _NO_PLAN if run.scan_id is None else str(run.scan_id)
            )
        else:
            self._inner_plan.setText(_NO_PLAN)
            self._inner_scan_id.setText(_NO_PLAN)
        self._set_inner_visible(bool(inner))

    @Slot(str)
    def _on_state_changed(self, state: str):
        self._indicator.set_state(state)
        self._indicator.setVisible(state != "idle")
        self._state.setText(state.capitalize())
        font = self._state.font()
        font.setBold(state != "idle")
        self._state.setFont(font)

    @Slot(float, int, str, str)
    def _on_record(self, created: float, levelno: int, levelname: str, message: str):
        if levelno >= logging.INFO:
            self._last.setText(message)

    def closeEvent(self, event):
        self._re_state.stop()
        self._current_plan.stop()
        self._logs.stop()
        super().closeEvent(event)


class QtLiveTable(QWidget):
    """Table equivalent of the BestEffortCallback's LiveTable.

    Shows the same columns (seq_num, time, then the hinted dimension and data
    fields of the primary stream) with the same number formatting.
    """

    _DEFAULT_PRECISION = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self._start_doc: dict | None = None
        self._all_dim_fields: list[str] = []
        self._dim_fields: list[str] = []
        self._dim_stream = "primary"
        self._cleanup_motor_heuristic = False
        self._descriptor_uid: str | None = None
        self._fields: list[str] = []
        self._data_keys: dict = {}

        self._table = QTableWidget(0, 0)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.setFont(_monospace_font())
        self._footer = QLabel("")
        self._footer.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(self._table, stretch=1)
        vbox.addWidget(self._footer)

    @Slot(str, object)
    def on_document(self, name: str, doc: dict):
        handler = getattr(self, f"_on_{name}", None)
        if handler is not None:
            handler(doc)

    def _on_start(self, doc: dict):
        # Mirrors BestEffortCallback.start's choice of dimensions.
        self._start_doc = doc
        motors = doc.get("motors")
        guess = (
            [([m], "primary") for m in motors] if motors else [(["time"], "primary")]
        )
        dimensions = doc.get("hints", {}).get("dimensions")
        self._cleanup_motor_heuristic = dimensions is None
        if dimensions is None or len({d[1] for d in dimensions}) != 1:
            self._cleanup_motor_heuristic = True
            dimensions = guess
        self._dim_fields = [fields[0] if fields else "time" for fields, _ in dimensions]
        self._all_dim_fields = [field for fields, _ in dimensions for field in fields]
        self._dim_stream = dimensions[0][1]
        self._descriptor_uid = None
        self._table.setRowCount(0)
        self._table.setColumnCount(0)
        self._footer.setText("")

    def _on_descriptor(self, doc: dict):
        if doc.get("name", "primary") != self._dim_stream:
            return
        if self._cleanup_motor_heuristic:
            # The start doc only named objects; resolve them to fields now.
            self._cleanup_motor_heuristic = False
            fixed = []
            for obj_name in self._dim_fields:
                if obj_name == "time":
                    fixed.append("time")
                    continue
                hint = doc.get("hints", {}).get(obj_name, {})
                fixed.extend(
                    hint.get("fields", doc.get("object_keys", {}).get(obj_name, []))
                )
            self._dim_fields = fixed
        columns = [c for c in hinted_fields(doc) if c not in self._all_dim_fields]
        data_keys = doc.get("data_keys", {})
        # Like LiveTable: only fields this stream has, in a dtype it can display.
        self._fields = [
            f
            for f in list(self._all_dim_fields) + columns
            if f in data_keys
            and data_keys[f].get("dtype") in ("number", "integer", "string", "boolean")
        ]
        self._data_keys = data_keys
        self._descriptor_uid = doc["uid"]
        headers = ["seq_num", "time", *self._fields]
        self._table.setRowCount(0)
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)

    def _on_event(self, doc: dict):
        if doc.get("descriptor") != self._descriptor_uid:
            return
        data = doc.get("data", {})
        filled = doc.get("filled", {})
        cells = [
            str(doc.get("seq_num", "")),
            str(datetime.fromtimestamp(doc["time"]).time())[:10],
        ]
        for name in self._fields:
            if name in data and filled.get(name, True):
                cells.append(self._format(name, data[name]))
            else:
                cells.append("")
        scrollbar = self._table.verticalScrollBar()
        at_bottom = scrollbar.value() == scrollbar.maximum()
        row = self._table.rowCount()
        self._table.insertRow(row)
        for column, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._table.setItem(row, column, item)
        if at_bottom:
            self._table.scrollToBottom()

    def _on_stop(self, doc: dict):
        start = self._start_doc
        if start is None or doc.get("run_start") != start.get("uid"):
            return
        self._footer.setText(
            f"{start.get('plan_type', '')} {start.get('plan_name', '')} "
            f"['{start.get('uid', '')[:8]}'] (scan num: {start.get('scan_id', '')})"
        )

    def _format(self, field: str, value: Any) -> str:
        if isinstance(value, enum.Enum):
            value = value.value
        if isinstance(value, bool):
            return str(value)
        dtype = self._data_keys.get(field, {}).get("dtype")
        try:
            if dtype == "number":
                precision = self._data_keys[field].get(
                    "precision", self._DEFAULT_PRECISION
                )
                try:
                    precision = int(precision)
                except (TypeError, ValueError):
                    precision = self._DEFAULT_PRECISION
                return f"{value:.{precision}f}"
            if dtype == "integer":
                return f"{value:d}"
        except (TypeError, ValueError):
            pass
        return str(value)


# TODO: Restore once the progress view gets documents from the RunEngine client.
# class QtPlanProgressView(QWidget):
#     """Live table of the running plan, with the BestEffortCallback's plots beside it.
#
#     Any ``BestEffortCallback`` in the session namespace is redirected to draw its
#     figures here instead of in separate windows. The plot side stays hidden
#     until a figure is created.
#     """
#
#     def __init__(self, monitor: PlanMonitor, parent=None):
#         super().__init__(parent)
#         self._table = QtLiveTable()
#
#         self._plots = QWidget()
#         self._plots_layout = QVBoxLayout(self._plots)
#         self._plots_layout.setContentsMargins(0, 0, 0, 0)
#         self._plots.hide()
#
#         self._splitter = QSplitter(Qt.Orientation.Horizontal)
#         self._splitter.addWidget(self._table)
#         self._splitter.addWidget(self._plots)
#
#         vbox = QVBoxLayout(self)
#         vbox.setContentsMargins(0, 0, 0, 0)
#         vbox.addWidget(self._splitter)
#
#         monitor.document.connect(self._table.on_document)
#         monitor.plan_started.connect(self._clear_plots)
#         for obj in monitor.namespace.values():
#             if isinstance(obj, BestEffortCallback):
#                 obj._fig_factory = self._make_figure  # noqa: SLF001
#
#     def _make_figure(self, name: str) -> Figure:
#         # The BEC is a QtAwareCallback, so this is called on the GUI thread.
#         figure = Figure(layout="constrained")
#         figure.set_label(name)
#         canvas = FigureCanvasQTAgg(figure)
#         plot = QWidget()
#         plot_layout = QVBoxLayout(plot)
#         plot_layout.setContentsMargins(0, 0, 0, 0)
#         plot_layout.addWidget(NavigationToolbar2QT(canvas, plot))
#         plot_layout.addWidget(canvas, stretch=1)
#         self._plots_layout.addWidget(plot, stretch=1)
#         if not self._plots.isVisible():
#             self._plots.show()
#             half = max(1, self._splitter.width() // 2)
#             self._splitter.setSizes([half, half])
#         return figure
#
#     @Slot()
#     def _clear_plots(self, *_):
#         while self._plots_layout.count():
#             item = self._plots_layout.takeAt(0)
#             if item is not None and item.widget() is not None:
#                 item.widget().deleteLater()  # type: ignore
#         self._plots.hide()


class QtPlanExecutionView(QWidget):
    """The running plan's call, above the client's log records since it started."""

    def __init__(self, re_client: RunEngineClient, parent=None):
        super().__init__(parent)
        self._call = QLabel("No plan has run yet.")
        self._call.setFont(_monospace_font())
        self._call.setWordWrap(True)
        self._call.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        call_box = QGroupBox("Current Plan")
        call_layout = QVBoxLayout()
        call_layout.addWidget(self._call)
        call_box.setLayout(call_layout)

        self._messages = QPlainTextEdit()
        self._messages.setReadOnly(True)
        self._messages.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._messages.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        progress_tabs = QTabWidget()
        # progress_tabs.addTab(QtPlanProgressView(monitor), "Progress")
        progress_tabs.addTab(self._messages, "Debug")
        self._tabs = progress_tabs

        vbox = QVBoxLayout()
        vbox.addWidget(call_box)
        vbox.addWidget(progress_tabs, stretch=1)
        self.setLayout(vbox)

        self._item_uid: str | None = None
        self._current_plan = QtCurrentPlan(re_client, self)
        self._current_plan.changed.connect(self._on_plan_changed)
        self._on_plan_changed(re_client.current_plan())
        self._logs = QtLogRecords(re_client, self)
        self._logs.record.connect(self._on_record)

    def add_tab(self, widget: QWidget, label: str) -> None:
        """Add a tab beside Debug."""
        self._tabs.addTab(widget, label)

    @Slot(object)
    def _on_plan_changed(self, plan: PlanInfo | None):
        # Keep showing the last plan once it finishes.
        if plan is None:
            return
        self._call.setText(plan.call)
        if plan.item_uid != self._item_uid:
            self._item_uid = plan.item_uid
            self._messages.clear()

    @Slot(float, int, str, str)
    def _on_record(self, created: float, levelno: int, levelname: str, message: str):
        scrollbar = self._messages.verticalScrollBar()
        at_bottom = scrollbar.value() == scrollbar.maximum()
        self._messages.appendHtml(
            _record_html(created, levelno, levelname, message, "%H:%M:%S")
        )
        # Only follow new lines if the user hasn't scrolled up to read history.
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())


def _record_html(
    created: float, levelno: int, levelname: str, message: str, datefmt: str
) -> str:
    """One log line as HTML, with only the level name colored."""
    color = level_color(levelno).name()
    return (
        f"[{datetime.fromtimestamp(created):{datefmt}}] "
        f'<span style="color:{color}">{html.escape(levelname)}</span>: '
        f"{html.escape(message)}"
    )


class QtPlanLogView(QWidget):
    """Every record the client logs this session, color-coded by level.

    Each plan's records follow a line with its call.
    """

    _MAX_LINES = 20000

    def __init__(self, re_client: RunEngineClient, parent=None):
        super().__init__(parent)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(_monospace_font())
        self._log.setMaximumBlockCount(self._MAX_LINES)
        vbox = QVBoxLayout()
        vbox.addWidget(self._log)
        self.setLayout(vbox)

        self._item_uid: str | None = None
        self._current_plan = QtCurrentPlan(re_client, self)
        self._current_plan.changed.connect(self._on_plan_changed)
        self._logs = QtLogRecords(re_client, self)
        self._logs.record.connect(self._on_record)

    @Slot(object)
    def _on_plan_changed(self, plan: PlanInfo | None):
        if plan is None or plan.item_uid == self._item_uid:
            return
        self._item_uid = plan.item_uid
        stamp = datetime.fromtimestamp(plan.start_time or 0)
        self._log.appendHtml(
            html.escape(f"[{stamp:%Y-%m-%d %H:%M:%S}] === {plan.call} ===")
        )

    @Slot(float, int, str, str)
    def _on_record(self, created: float, levelno: int, levelname: str, message: str):
        self._log.appendHtml(
            _record_html(created, levelno, levelname, message, "%Y-%m-%d %H:%M:%S")
        )


class QtPlanHistory(QWidget):
    """Table of a RunEngine client's plan history, newest first.

    A selected plan can be re-run, and the history cleared.

    Parameters
    ----------
    re_client : RunEngineClient
        Keeps the history, and re-runs plans.
    """

    _COLUMNS = ("Start", "Stop", "Plan", "Arguments", "Status", "Scan ID", "UID")
    _ARGUMENTS_COLUMN = 3

    def __init__(self, re_client: RunEngineClient, parent=None):
        super().__init__(parent)
        self._re_client = re_client
        self._items: list[HistoryItem] = []  # same order as table rows

        self._table = QTableWidget(0, len(self._COLUMNS))
        self._table.setHorizontalHeaderLabels(self._COLUMNS)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            self._ARGUMENTS_COLUMN, QHeaderView.ResizeMode.Stretch
        )
        self._table.itemSelectionChanged.connect(self._update_button)

        self._rerun_button = QPushButton("Re-run Selected Plan")
        self._rerun_button.setEnabled(False)
        self._rerun_button.clicked.connect(self._on_rerun_clicked)
        self._clear_button = QPushButton("Clear History")
        self._clear_button.setEnabled(False)
        self._clear_button.clicked.connect(self._on_clear_clicked)
        buttons = QHBoxLayout()
        buttons.addWidget(self._clear_button)
        buttons.addStretch()
        buttons.addWidget(self._rerun_button)

        vbox = QVBoxLayout()
        vbox.addWidget(self._table, stretch=1)
        vbox.addLayout(buttons)
        self.setLayout(vbox)

        self._history = QtPlanHistoryChanges(re_client, self)
        self._history.changed.connect(self._show)
        self._show(re_client.plan_history())

    @staticmethod
    def _fmt_time(timestamp: float | None) -> str:
        if timestamp is None:
            return _NOT_APPLICABLE
        return f"{datetime.fromtimestamp(timestamp):%Y-%m-%d %H:%M:%S}"

    @Slot(object)
    def _show(self, history: tuple[HistoryItem, ...]):
        selected = self._selected_item()
        self._items = list(reversed(history))
        self._table.setRowCount(len(self._items))
        for row, item in enumerate(self._items):
            result = item.result
            scan_ids = [str(s) for s in result.scan_ids if s is not None]
            status = result.exit_status.capitalize()
            cells = (
                self._fmt_time(result.time_start),
                self._fmt_time(result.time_stop),
                item.name,
                item.arguments,
                f"{status}: {result.msg}" if result.msg else status,
                ", ".join(scan_ids) or _NOT_APPLICABLE,
                ", ".join(result.run_uids) or _NOT_APPLICABLE,
            )
            for column, text in enumerate(cells):
                cell = QTableWidgetItem(text)
                cell.setToolTip(text)
                self._table.setItem(row, column, cell)
            if selected is not None and item.item_uid == selected.item_uid:
                self._table.selectRow(row)
        self._update_button()

    def clear(self):
        """Remove every plan from the client's history."""
        self._re_client.clear_history()

    def _on_clear_clicked(self):
        answer = QMessageBox.question(
            self,
            "Clear plan history",
            "Remove all plans from the history? This can't be undone.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.clear()

    def _selected_item(self) -> HistoryItem | None:
        rows = self._table.selectionModel().selectedRows()
        return self._items[rows[0].row()] if rows else None

    def _rerun_problem(self, item: HistoryItem) -> str | None:
        """Why ``item`` can't be re-run, or None if it can."""
        if item.name not in self._re_client.plan_catalog:
            return f"'{item.name}' is not an available plan."
        try:
            self._re_client.serialize_plan(item.name, *item.args, **item.kwargs)
        except (TypeError, ValueError):
            return "Some of this plan's arguments can't be passed to it again."
        return None

    @Slot()
    def _update_button(self):
        self._clear_button.setEnabled(bool(self._items))
        item = self._selected_item()
        problem = self._rerun_problem(item) if item is not None else None
        self._rerun_button.setEnabled(item is not None and problem is None)
        self._rerun_button.setToolTip(problem or "")

    def _on_rerun_clicked(self):
        item = self._selected_item()
        if item is None or self._rerun_problem(item) is not None:
            return
        if self._re_client.state != "idle":
            QMessageBox.warning(
                self, "RunEngine busy", "Wait for the current plan to finish."
            )
            return
        # Defer so the click handler returns before the plan runs.
        QTimer.singleShot(0, lambda: self._execute(item))

    def _execute(self, item: HistoryItem):
        try:
            self._re_client.execute_plan(item.name, *item.args, **item.kwargs)
        except Exception as error:
            show_plan_error(self, "Re-run failed", error)

    def closeEvent(self, event):
        self._history.stop()
        super().closeEvent(event)
