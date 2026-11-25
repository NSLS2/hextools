"""A window summarizing one finished plan: how it went, its arguments, and its runs."""

from __future__ import annotations

from datetime import datetime
from itertools import zip_longest
from typing import NamedTuple

from qtpy.QtCore import Qt
from qtpy.QtGui import QFontDatabase
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hextools.catalog.model import PlanDescription
from hextools.gui.run_engine_clients import HistoryItem, _source

_NOT_APPLICABLE = "N/A"
_POSITIONAL = ("positional_only", "positional_or_keyword")


class Argument(NamedTuple):
    """One of a plan's parameters, and the value it was run with."""

    name: str
    value: str
    description: str
    #: Whether the value was passed, rather than the parameter's default.
    given: bool


def plan_arguments(item: HistoryItem, plan: PlanDescription | None) -> list[Argument]:
    """Return each of ``plan``'s parameters with the value ``item`` ran it with.

    Parameters that weren't passed show their default. Without ``plan``, only
    the arguments passed are returned, positional ones named by position.
    """
    args = list(item.args)
    kwargs = dict(item.kwargs)
    arguments: list[Argument] = []
    for p in plan.parameters if plan is not None else ():
        description = p.description or ""
        if p.kind in _POSITIONAL and args:
            arguments.append(Argument(p.name, _source(args.pop(0)), description, True))
        elif p.kind == "var_positional":
            if args:
                values = ", ".join(_source(a) for a in args)
                arguments.append(Argument(f"*{p.name}", values, description, True))
                args = []
        elif p.kind == "var_keyword":
            continue
        elif p.name in kwargs:
            value = _source(kwargs.pop(p.name))
            arguments.append(Argument(p.name, value, description, True))
        elif p.has_default:
            arguments.append(Argument(p.name, _source(p.default), description, False))
    arguments += [
        Argument(f"argument {i}", _source(a), "", True)
        for i, a in enumerate(args, start=len(item.args) - len(args) + 1)
    ]
    arguments += [Argument(k, _source(v), "", True) for k, v in kwargs.items()]
    return arguments


def _fmt_time(timestamp: float | None) -> str:
    if timestamp is None:
        return _NOT_APPLICABLE
    return f"{datetime.fromtimestamp(timestamp):%Y-%m-%d %H:%M:%S}"


def _fmt_duration(start: float | None, stop: float | None) -> str:
    if start is None or stop is None:
        return _NOT_APPLICABLE
    minutes, seconds = divmod(round(stop - start), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def _label(text: str, *, monospace: bool = False) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    if monospace:
        label.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
    return label


def _table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> QTableWidget:
    table = QTableWidget(len(rows), len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setWordWrap(True)
    table.setTextElideMode(Qt.TextElideMode.ElideNone)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    for row, cells in enumerate(rows):
        for column, text in enumerate(cells):
            cell = QTableWidgetItem(text)
            cell.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
            )
            table.setItem(row, column, cell)
    return table


class QtPlanSummary(QWidget):
    """Everything about one finished plan, in its own window.

    How it ended and when, the plan's description, each argument with its
    parameter's description, every run's scan ID and uid, and any traceback.

    Parameters
    ----------
    item : HistoryItem
        The finished plan.
    plan : PlanDescription, optional
        The plan's catalog entry, for its descriptions and defaults; None if the
        plan is no longer in the catalog.
    """

    def __init__(
        self,
        item: HistoryItem,
        plan: PlanDescription | None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.item = item
        self.setWindowTitle(f"Plan Summary \u2014 {item.name}")
        result = item.result

        status = result.exit_status.capitalize()
        overview = QFormLayout()
        overview.addRow("Plan:", _label(item.name))
        overview.addRow(
            "Status:", _label(f"{status}: {result.msg}" if result.msg else status)
        )
        overview.addRow("Started:", _label(_fmt_time(result.time_start)))
        overview.addRow("Stopped:", _label(_fmt_time(result.time_stop)))
        overview.addRow(
            "Duration:", _label(_fmt_duration(result.time_start, result.time_stop))
        )
        overview.addRow("Call:", _label(item.call, monospace=True))
        if plan is None:
            description = "This plan is not in the plan catalog."
        else:
            description = plan.description or "No description."
        overview.addRow("Description:", _label(description))
        overview_box = QGroupBox("Plan")
        overview_box.setLayout(overview)

        arguments = plan_arguments(item, plan)
        self.arguments = _table(
            ("Parameter", "Value", "Description"),
            [
                (a.name, a.value if a.given else f"{a.value} (default)", a.description)
                for a in arguments
            ],
        )
        for row, argument in enumerate(arguments):
            if not argument.given:
                for column in range(2):
                    cell = self.arguments.item(row, column)
                    font = cell.font()
                    font.setItalic(True)
                    cell.setFont(font)
                    cell.setToolTip("Not passed; the plan's default.")
        header = self.arguments.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        runs = list(zip_longest(result.scan_ids, result.run_uids))
        self.runs = _table(
            ("Scan ID", "UID"),
            [
                (_NOT_APPLICABLE if s is None else str(s), u or _NOT_APPLICABLE)
                for s, u in runs
            ],
        )
        header = self.runs.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        for title, widget in (
            (f"Arguments ({sum(a.given for a in arguments)} passed)", self.arguments),
            (f"Runs ({len(runs)})", self.runs),
        ):
            box = QGroupBox(title)
            QVBoxLayout(box).addWidget(widget)
            splitter.addWidget(box)
        if result.traceback:
            traceback = QPlainTextEdit(result.traceback)
            traceback.setReadOnly(True)
            traceback.setFont(
                QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
            )
            box = QGroupBox("Traceback")
            QVBoxLayout(box).addWidget(traceback)
            splitter.addWidget(box)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)

        vbox = QVBoxLayout(self)
        vbox.addWidget(overview_box)
        vbox.addWidget(splitter, stretch=1)
        vbox.addWidget(buttons)
        self.resize(900, 700)
