"""Widgets showing the running plan, its natural-language narration, and history."""

from __future__ import annotations

import enum
import importlib
import inspect
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from bluesky import RunEngine
from qtpy.QtCore import QObject, Qt, QTimer, Signal, Slot
from qtpy.QtGui import QFont, QFontDatabase
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hextools.gui._ipython import run_in_ipython
from hextools.gui.device_sources import _walk_namespace
from hextools.utils.msg_hooks import MsgHookNarrator, nl_msg_hook

_NO_PLAN = "\u2014"  # em dash
_NOT_APPLICABLE = "N/A"
_EXIT_STATUS_LABELS = {"success": "Success", "fail": "Failed", "abort": "Aborted"}

logger = logging.getLogger(__name__)


class _Unrenderable(Exception):
    """A plan argument has no expression that would recreate it in the namespace."""


def _render(value: Any, names: Mapping[int, str]) -> str:
    """Return a Python expression for ``value`` that evaluates in the namespace."""
    if id(value) in names:
        return names[id(value)]
    if isinstance(value, enum.Enum):
        if isinstance(value, str):
            return repr(value.value)
        raise _Unrenderable
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return repr(value)
    if hasattr(value, "dtype") and hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        return repr(value.item())  # numpy scalar
    if isinstance(value, list):
        return "[" + ", ".join(_render(v, names) for v in value) + "]"
    if isinstance(value, tuple):
        inner = ", ".join(_render(v, names) for v in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_render(k, names)}: {_render(v, names)}" for k, v in value.items()) + "}"
    raise _Unrenderable


def _display(value: Any, names: Mapping[int, str]) -> str:
    """Like :func:`_render`, but always returns something readable."""
    try:
        return _render(value, names)
    except _Unrenderable:
        name = getattr(value, "name", None)
        return name if isinstance(name, str) else repr(value)


@dataclass
class PlanRecord:
    """A plan run through the RunEngine, as captured just before it started."""

    name: str
    arguments: str
    rerunnable: bool
    func: Callable | None = None
    scan_ids: list = field(default_factory=list)
    uids: list[str] = field(default_factory=list)
    exit_status: str | None = None
    start_time: float | None = None
    stop_time: float | None = None

    @property
    def call(self) -> str:
        return f"{self.name}({self.arguments})"

    def to_json(self) -> dict[str, Any]:
        func = None
        if self.func is not None:
            func = f"{self.func.__module__}:{self.func.__qualname__}"
        return {
            "name": self.name,
            "arguments": self.arguments,
            "rerunnable": self.rerunnable,
            "function": func,
            "scan_ids": self.scan_ids,
            "uids": self.uids,
            "exit_status": self.exit_status,
            "start_time": self.start_time,
            "stop_time": self.stop_time,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PlanRecord:
        func = _import_function(data.get("function"))
        return cls(
            name=data["name"],
            arguments=data["arguments"],
            rerunnable=bool(data.get("rerunnable")) and func is not None,
            func=func,
            scan_ids=list(data.get("scan_ids") or []),
            uids=list(data.get("uids") or []),
            exit_status=data.get("exit_status"),
            start_time=data.get("start_time"),
            stop_time=data.get("stop_time"),
        )


def _import_function(reference: str | None) -> Callable | None:
    """Resolve a ``module:qualname`` reference, or return None if it no longer exists."""
    if not reference:
        return None
    module_name, _, qualname = reference.partition(":")
    try:
        obj: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError):
        return None
    return obj if callable(obj) else None


def _record_plan(plan: Any, namespace: Mapping[str, Any]) -> PlanRecord:
    """Capture a plan's name and arguments before its first message is sent."""
    gen = getattr(plan, "_iter", plan)  # unwrap bluesky's @plan ``Plan`` object
    if not inspect.isgenerator(gen):
        name = getattr(plan, "__name__", None) or type(plan).__name__
        return PlanRecord(name=name, arguments="...", rerunnable=False)

    code = gen.gi_code
    # Before the first send, the generator's locals are exactly its arguments.
    args = inspect.getgeneratorlocals(gen)
    names = {id(obj): path for path, obj in _walk_namespace(namespace).items()}

    varnames = code.co_varnames
    n_pos, n_kwonly = code.co_argcount, code.co_kwonlyargcount
    positional = varnames[:n_pos]
    kwonly = varnames[n_pos : n_pos + n_kwonly]
    index = n_pos + n_kwonly
    varargs = varkw = None
    if code.co_flags & inspect.CO_VARARGS:
        varargs = varnames[index]
        index += 1
    if code.co_flags & inspect.CO_VARKEYWORDS:
        varkw = varnames[index]

    # (keyword or None for positional, value) in call order.
    items: list[tuple[str | None, Any]] = []
    extra_positional = args.get(varargs, ()) if varargs else ()
    for i, param in enumerate(positional):
        as_positional = i < code.co_posonlyargcount or bool(extra_positional)
        items.append((None if as_positional else param, args[param]))
    items.extend((None, v) for v in extra_positional)
    items.extend((param, args[param]) for param in kwonly)
    if varkw:
        items.extend(args.get(varkw, {}).items())

    rerunnable = True
    parts = []
    for key, value in items:
        try:
            text = _render(value, names)
        except _Unrenderable:
            rerunnable = False
            text = _display(value, names)
        parts.append(text if key is None else f"{key}={text}")

    func = gen.gi_frame.f_globals.get(code.co_name) if gen.gi_frame is not None else None
    # Only re-run via a module-level function whose (unwrapped) code is this generator's.
    if not callable(func) or getattr(inspect.unwrap(func), "__code__", None) is not code:
        func = None
    return PlanRecord(
        name=code.co_name,
        arguments=", ".join(parts),
        rerunnable=rerunnable and func is not None,
        func=func,
    )


def _monospace_font() -> QFont:
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)


class _ElidedLabel(QLabel):
    """Single-line label that elides overflow with "…" and shows the full text as a tooltip."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full_text = ""
        # Don't let long text widen the window; take whatever width the layout gives.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str):
        self._full_text = text
        self.setToolTip(text)
        self._elide()

    def text(self) -> str:
        return self._full_text

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def _elide(self):
        elided = self.fontMetrics().elidedText(
            self._full_text, Qt.TextElideMode.ElideRight, self.width()
        )
        super().setText(elided)


class PlanMonitor(QObject):
    """Bridges RunEngine activity to Qt signals on the GUI thread.

    Attaches ``narrator`` as the RunEngine ``msg_hook`` and chains onto any
    existing ``msg_hook`` and ``state_hook`` rather than replacing them.
    """

    line = Signal(float, str)
    plan_started = Signal(str, str)
    plan_finished = Signal(object)
    run_started = Signal(object, str)

    def __init__(
        self,
        re: RunEngine,
        narrator: MsgHookNarrator = nl_msg_hook,
        namespace: Mapping[str, Any] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._re = re
        self._narrator = narrator
        self._namespace = namespace if namespace is not None else {}
        self._current: PlanRecord | None = None
        narrator.add_listener(self._on_line)

        previous_msg_hook = re.msg_hook
        if previous_msg_hook is not narrator:

            def msg_hook(msg):
                if previous_msg_hook is not None:
                    previous_msg_hook(msg)
                narrator(msg)

            re.msg_hook = msg_hook

        previous_state_hook = re.state_hook

        def state_hook(new_state, old_state):
            self._on_state(str(new_state), str(old_state))
            if previous_state_hook is not None:
                previous_state_hook(new_state, old_state)

        re.state_hook = state_hook
        self._start_token = re.subscribe(self._on_start, "start")

    @property
    def namespace(self) -> Mapping[str, Any]:
        return self._namespace

    @property
    def run_engine(self) -> RunEngine:
        return self._re

    # These run on the RunEngine thread; Qt queues the signals onto the GUI thread.
    def _on_line(self, text: str):
        self.line.emit(datetime.now().timestamp(), text)

    def _on_state(self, new_state: str, old_state: str):
        if new_state == "running" and old_state == "idle":
            plan = getattr(self._re, "_plan", None)  # noqa: SLF001
            self._current = _record_plan(plan, self._namespace)
            self._current.start_time = time.time()
            self.plan_started.emit(self._current.name, self._current.call)
        elif new_state == "idle":
            self._narrator.flush()
            record, self._current = self._current, None
            if record is not None:
                record.stop_time = time.time()
                record.exit_status = getattr(self._re, "_exit_status", None)  # noqa: SLF001
                self.plan_finished.emit(record)

    def _on_start(self, name, doc):
        scan_id, uid = doc.get("scan_id"), doc.get("uid", "")
        if self._current is not None:
            self._current.scan_ids.append(scan_id)
            self._current.uids.append(uid)
        self.run_started.emit(scan_id, uid)


class QtPlanStatus(QWidget):
    """Compact summary: plan name, scan id, and latest narration."""

    def __init__(self, monitor: PlanMonitor, parent=None):
        super().__init__(parent)
        self._plan = QLabel(_NO_PLAN)
        self._scan_id = QLabel(_NO_PLAN)
        self._last = _ElidedLabel(_NO_PLAN)
        self._last.setMinimumWidth(260)

        form = QFormLayout()
        form.setContentsMargins(8, 6, 8, 6)
        form.addRow("Plan:", self._plan)
        form.addRow("Scan ID:", self._scan_id)
        form.addRow("Last step:", self._last)

        group_box = QGroupBox("Plan Status")
        group_box.setLayout(form)
        vbox = QVBoxLayout()
        vbox.addWidget(group_box)
        self.setLayout(vbox)

        monitor.plan_started.connect(self._on_plan_started)
        monitor.run_started.connect(self._on_run_started)
        monitor.line.connect(self._on_line)

    @Slot(str, str)
    def _on_plan_started(self, name: str, call: str):
        self._plan.setText(name)
        self._scan_id.setText(_NO_PLAN)
        self._last.setText(_NO_PLAN)

    @Slot(object, str)
    def _on_run_started(self, scan_id, uid: str):
        self._scan_id.setText(_NO_PLAN if scan_id is None else str(scan_id))

    @Slot(float, str)
    def _on_line(self, timestamp: float, text: str):
        self._last.setText(text)


class QtPlanExecutionView(QWidget):
    """The running plan's full call, above its narration since it started."""

    def __init__(self, monitor: PlanMonitor, parent=None):
        super().__init__(parent)
        self._call = QLabel("No plan has run yet.")
        self._call.setFont(_monospace_font())
        self._call.setWordWrap(True)
        self._call.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        call_box = QGroupBox("Current Plan")
        call_layout = QVBoxLayout()
        call_layout.addWidget(self._call)
        call_box.setLayout(call_layout)

        self._messages = QListWidget()
        self._messages.setWordWrap(True)
        messages_box = QGroupBox("Progress")
        messages_layout = QVBoxLayout()
        messages_layout.addWidget(self._messages)
        messages_box.setLayout(messages_layout)

        vbox = QVBoxLayout()
        vbox.addWidget(call_box)
        vbox.addWidget(messages_box, stretch=1)
        self.setLayout(vbox)

        monitor.plan_started.connect(self._on_plan_started)
        monitor.line.connect(self._on_line)

    @Slot(str, str)
    def _on_plan_started(self, name: str, call: str):
        self._call.setText(call)
        self._messages.clear()

    @Slot(float, str)
    def _on_line(self, timestamp: float, text: str):
        scrollbar = self._messages.verticalScrollBar()
        at_bottom = scrollbar.value() == scrollbar.maximum()
        self._messages.addItem(f"[{datetime.fromtimestamp(timestamp):%H:%M:%S %H:%M:%S}] {text}")
        # Only follow new lines if the user hasn't scrolled up to read history.
        if at_bottom:
            self._messages.scrollToBottom()


class QtPlanLogView(QWidget):
    """Every narration line received this session."""

    _MAX_LINES = 20000

    def __init__(self, monitor: PlanMonitor, parent=None):
        super().__init__(parent)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(_monospace_font())
        self._log.setMaximumBlockCount(self._MAX_LINES)
        vbox = QVBoxLayout()
        vbox.addWidget(self._log)
        self.setLayout(vbox)

        monitor.plan_started.connect(self._on_plan_started)
        monitor.line.connect(self._on_line)

    @Slot(str, str)
    def _on_plan_started(self, name: str, call: str):
        self._append(datetime.now().timestamp(), f"=== {call} ===")

    @Slot(float, str)
    def _on_line(self, timestamp: float, text: str):
        self._append(timestamp, text)

    def _append(self, timestamp: float, text: str):
        stamp = datetime.fromtimestamp(timestamp)
        self._log.appendPlainText(f"[{stamp:%Y-%m-%d %H:%M:%S}] {text}")


class QtPlanHistory(QWidget):
    """Table of completed plans, newest first, with re-run of a selected plan.

    Parameters
    ----------
    monitor : PlanMonitor
        Source of finished plans.
    history_file : str or Path, optional
        JSON file the history is saved to after every plan. If it already
        exists, its plans are loaded into the table on startup.
    """

    _COLUMNS = ("Start", "Stop", "Plan", "Arguments", "Status", "Scan ID", "UID")
    _ARGUMENTS_COLUMN = 3

    def __init__(self, monitor: PlanMonitor, history_file: str | Path | None = None, parent=None):
        super().__init__(parent)
        self._monitor = monitor
        self._records: list[PlanRecord] = []  # same order as table rows
        self._history_file = Path(history_file).expanduser() if history_file else None

        self._table = QTableWidget(0, len(self._COLUMNS))
        self._table.setHorizontalHeaderLabels(self._COLUMNS)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._ARGUMENTS_COLUMN, QHeaderView.ResizeMode.Stretch)
        self._table.itemSelectionChanged.connect(self._update_button)

        self._rerun_button = QPushButton("Re-run Selected Plan")
        self._rerun_button.setEnabled(False)
        self._rerun_button.clicked.connect(self._on_rerun_clicked)
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self._rerun_button)

        vbox = QVBoxLayout()
        vbox.addWidget(self._table, stretch=1)
        vbox.addLayout(buttons)
        self.setLayout(vbox)

        self._load()
        monitor.plan_finished.connect(self._on_plan_finished)

    def _load(self):
        if self._history_file is None or not self._history_file.exists():
            return
        try:
            entries = json.loads(self._history_file.read_text())
            records = [PlanRecord.from_json(entry) for entry in entries]
        except (OSError, ValueError, KeyError, TypeError) as ex:
            # Don't overwrite a file we couldn't read; it may hold history worth recovering.
            logger.warning("Not saving plan history: failed to load %s: %s", self._history_file, ex)
            self._history_file = None
            return
        for record in records:  # stored oldest first
            self._add_row(record)

    def _save(self):
        if self._history_file is None:
            return
        entries = [record.to_json() for record in reversed(self._records)]
        tmp = self._history_file.with_name(self._history_file.name + ".tmp")
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(entries, indent=2, default=str))
            os.replace(tmp, self._history_file)  # atomic, so a crash can't truncate the file
        except OSError as ex:
            logger.warning("Failed to save plan history to %s: %s", self._history_file, ex)

    @staticmethod
    def _fmt_time(timestamp: float | None) -> str:
        if timestamp is None:
            return _NOT_APPLICABLE
        return f"{datetime.fromtimestamp(timestamp):%Y-%m-%d %H:%M:%S}"

    @Slot(object)
    def _on_plan_finished(self, record: PlanRecord):
        self._add_row(record)
        self._save()

    def _add_row(self, record: PlanRecord):
        scan_ids = [str(s) for s in record.scan_ids if s is not None]
        cells = (
            self._fmt_time(record.start_time),
            self._fmt_time(record.stop_time),
            record.name,
            record.arguments,
            _EXIT_STATUS_LABELS.get(record.exit_status or "", record.exit_status or _NOT_APPLICABLE),
            ", ".join(scan_ids) or _NOT_APPLICABLE,
            ", ".join(record.uids) or _NOT_APPLICABLE,
        )
        self._table.insertRow(0)
        for column, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setToolTip(text)
            self._table.setItem(0, column, item)
        self._records.insert(0, record)
        self._update_button()

    def _selected_record(self) -> PlanRecord | None:
        rows = self._table.selectionModel().selectedRows()
        return self._records[rows[0].row()] if rows else None

    @Slot()
    def _update_button(self):
        record = self._selected_record()
        self._rerun_button.setEnabled(record is not None and record.rerunnable)
        if record is not None and not record.rerunnable:
            self._rerun_button.setToolTip(
                "This plan can't be re-run: its function or some of its arguments "
                "can't be found in the session namespace."
            )
        else:
            self._rerun_button.setToolTip("")

    def _on_rerun_clicked(self):
        record = self._selected_record()
        if record is None or not record.rerunnable:
            return
        if self._monitor.run_engine.state != "idle":
            QMessageBox.warning(self, "RunEngine busy", "Wait for the current plan to finish.")
            return
        namespace = self._monitor.namespace
        existing = namespace.get(record.name)
        if existing is None:
            namespace[record.name] = record.func  # type: ignore[index]
        elif existing is not record.func:
            QMessageBox.warning(
                self,
                "Name conflict",
                f"'{record.name}' in the session refers to a different object than the "
                "plan that originally ran, so it won't be re-run.",
            )
            return
        code = f"RE({record.call})"
        # Defer so the click handler returns before IPython executes the cell.
        QTimer.singleShot(0, lambda: self._execute(code))

    def _execute(self, code: str):
        error = run_in_ipython(code)
        if error is not None:
            QMessageBox.critical(self, "Re-run failed", f"{type(error).__name__}: {error}")
