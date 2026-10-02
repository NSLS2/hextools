"""Widgets showing the running plan and its natural-language narration."""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any

from bluesky import RunEngine
from qtpy.QtCore import QObject, Qt, Signal, Slot
from qtpy.QtGui import QFont, QFontDatabase
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from hextools.utils.msg_hooks import MsgHookNarrator, nl_msg_hook

_NO_PLAN = "\u2014"  # em dash


def _fmt_arg(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_fmt_arg(v) for v in value)
        return f"[{inner}]" if isinstance(value, list) else f"({inner})"
    name = getattr(value, "name", None)
    if isinstance(name, str) and not isinstance(value, str):
        return name
    return repr(value)


def _describe_plan(plan: Any) -> tuple[str, str]:
    """Return ``(name, call)`` for a plan that has not started iterating yet."""
    name = getattr(plan, "__name__", None) or type(plan).__name__
    if inspect.isgenerator(plan):
        # Before the first send, the generator's locals are exactly its arguments.
        args = inspect.getgeneratorlocals(plan)
        parts = [f"{k}={_fmt_arg(v)}" for k, v in args.items() if not k.startswith("_")]
        return name, f"{name}({', '.join(parts)})"
    return name, f"{name}(...)"


def _monospace_font() -> QFont:
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)


class PlanMonitor(QObject):
    """Bridges RunEngine activity to Qt signals on the GUI thread.

    Attaches ``narrator`` as the RunEngine ``msg_hook`` and chains onto any
    existing ``msg_hook`` and ``state_hook`` rather than replacing them.
    """

    line = Signal(float, str)
    plan_started = Signal(str, str)
    plan_finished = Signal()
    run_started = Signal(object, str)

    def __init__(self, re: RunEngine, narrator: MsgHookNarrator = nl_msg_hook, parent=None):
        super().__init__(parent)
        self._re = re
        self._narrator = narrator
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

    # These run on the RunEngine thread; Qt queues the signals onto the GUI thread.
    def _on_line(self, text: str):
        self.line.emit(datetime.now().timestamp(), text)

    def _on_state(self, new_state: str, old_state: str):
        if new_state == "running" and old_state == "idle":
            name, call = _describe_plan(getattr(self._re, "_plan", None))  # noqa: SLF001
            self.plan_started.emit(name, call)
        elif new_state == "idle":
            self._narrator.flush()
            self.plan_finished.emit()

    def _on_start(self, name, doc):
        self.run_started.emit(doc.get("scan_id"), doc.get("uid", ""))


class QtPlanStatus(QWidget):
    """Compact summary: plan name, scan id, latest narration, and run uids."""

    def __init__(self, monitor: PlanMonitor, parent=None):
        super().__init__(parent)
        self._plan = QLabel(_NO_PLAN)
        self._scan_id = QLabel(_NO_PLAN)
        self._last = QLabel(_NO_PLAN)
        self._last.setWordWrap(True)
        self._last.setMinimumWidth(260)
        self._uids = QListWidget()
        self._uids.setFont(_monospace_font())
        self._uids.setMaximumHeight(60)
        self._uids.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        form = QFormLayout()
        form.setContentsMargins(8, 6, 8, 6)
        form.addRow("Plan:", self._plan)
        form.addRow("Scan ID:", self._scan_id)
        form.addRow("Last step:", self._last)
        form.addRow("UIDs:", self._uids)

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
        self._uids.clear()

    @Slot(object, str)
    def _on_run_started(self, scan_id, uid: str):
        self._scan_id.setText(_NO_PLAN if scan_id is None else str(scan_id))
        self._uids.addItem(uid)

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
