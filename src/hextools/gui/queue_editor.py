"""The Queue tab: a plan form to queue plans from, and the queue to edit."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hextools.gui._client_signals import QtCatalog, QtPlanQueueChanges
from hextools.gui.plan_widget import QtPlanWidget
from hextools.gui.run_engine_clients import (
    PlanQueue,
    QueueError,
    QueueItem,
    RunEngineClient,
)


class QtQueueEditor(QWidget):
    """Table of a client's plan queue, front first, below the plan running from it.

    The selected queued plan can be moved up or down, removed, or edited in its
    plan form, filled in with its arguments, and saved back to the queue in
    place. Double-clicking a queued plan edits it too.

    Parameters
    ----------
    re_client : RunEngineClient
        The client whose queue to show and edit.
    """

    _COLUMNS = ("Position", "Plan", "Arguments", "Added By")
    _ARGUMENTS_COLUMN = 2

    def __init__(self, re_client: RunEngineClient, parent: QWidget | None = None):
        super().__init__(parent)
        self._re_client = re_client
        self._items: tuple[QueueItem, ...] = ()
        # Rows above the queued plans: 1 while a plan from the queue runs.
        self._offset = 0
        self._editors: dict[str, QtPlanWidget] = {}

        self._table = QTableWidget(0, len(self._COLUMNS))
        self._table.setHorizontalHeaderLabels(self._COLUMNS)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setWordWrap(True)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(
            self._ARGUMENTS_COLUMN, QHeaderView.ResizeMode.Stretch
        )
        self._table.itemSelectionChanged.connect(self._update_buttons)
        self._table.cellDoubleClicked.connect(lambda *_: self._on_edit_clicked())

        self._up_button = self._button(
            "Move Up", "Run the plan one place earlier", self._on_up_clicked
        )
        self._down_button = self._button(
            "Move Down", "Run the plan one place later", self._on_down_clicked
        )
        self._remove_button = self._button(
            "Remove", "Take the plan out of the queue", self._on_remove_clicked
        )
        self._edit_button = self._button("Edit\u2026", "", self._on_edit_clicked)
        buttons = QHBoxLayout()
        buttons.addWidget(self._up_button)
        buttons.addWidget(self._down_button)
        buttons.addStretch()
        buttons.addWidget(self._remove_button)
        buttons.addWidget(self._edit_button)

        vbox = QVBoxLayout()
        vbox.addWidget(self._table, stretch=1)
        vbox.addLayout(buttons)
        self.setLayout(vbox)

        self._queue = QtPlanQueueChanges(re_client, self)
        self._queue.changed.connect(self._show)
        self._show(re_client.queue_get())

    def _button(
        self, text: str, tooltip: str, on_click: Callable[[], None]
    ) -> QPushButton:
        button = QPushButton(text)
        button.setToolTip(tooltip)
        button.setEnabled(False)
        button.clicked.connect(lambda *_: on_click())
        return button

    @Slot(object)
    def _show(self, queue: PlanQueue):
        selected = self._selected_item()
        self._items = queue.items
        rows: list[tuple[str, QueueItem]] = [
            (str(position), item) for position, item in enumerate(queue.items, 1)
        ]
        if queue.running_item is not None:
            rows.insert(0, ("Running", queue.running_item))
        self._offset = int(queue.running_item is not None)
        self._table.setRowCount(len(rows))
        for row, (position, item) in enumerate(rows):
            for column, text in enumerate(
                (position, item.name, item.arguments, item.user)
            ):
                cell = QTableWidgetItem(text)
                cell.setToolTip(text)
                cell.setTextAlignment(
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
                )
                if row < self._offset:
                    # The running plan is shown, but can't be moved or edited.
                    cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self._table.setItem(row, column, cell)
        self._table.clearSelection()
        if selected is not None:
            for index, item in enumerate(self._items):
                if item.item_uid == selected.item_uid:
                    self._table.selectRow(index + self._offset)
        self._update_buttons()

    def _selected_index(self) -> int | None:
        """Index in the queue of the selected plan, or None."""
        rows = self._table.selectionModel().selectedRows()
        if not rows or rows[0].row() < self._offset:
            return None
        return rows[0].row() - self._offset

    def _selected_item(self) -> QueueItem | None:
        index = self._selected_index()
        return None if index is None else self._items[index]

    def _edit_problem(self, item: QueueItem) -> str | None:
        """Why ``item`` can't be edited in its form, or None if it can."""
        plan = self._re_client.plan_catalog.get(item.name)
        if plan is None:
            return f"'{item.name}' is not an available plan."
        try:
            plan.bind(item.args, item.kwargs)
        except TypeError:
            return "This plan's positional arguments can't be edited in a form."
        return None

    @Slot()
    def _update_buttons(self):
        index = self._selected_index()
        selected = index is not None
        self._up_button.setEnabled(selected and index > 0)
        self._down_button.setEnabled(selected and index < len(self._items) - 1)
        self._remove_button.setEnabled(selected)
        item = self._selected_item()
        problem = self._edit_problem(item) if item is not None else None
        self._edit_button.setEnabled(item is not None and problem is None)
        self._edit_button.setToolTip(
            problem or "Change the plan's arguments in its form, and save them"
        )

    def _call(self, title: str, action: Callable[[], object]):
        try:
            action()
        except QueueError as error:
            QMessageBox.warning(self, title, str(error))

    def _on_up_clicked(self):
        index = self._selected_index()
        if index:
            uid, before = self._items[index].item_uid, self._items[index - 1].item_uid
            self._call(
                "Plan not moved",
                lambda: self._re_client.item_move(uid=uid, before_uid=before),
            )

    def _on_down_clicked(self):
        index = self._selected_index()
        if index is not None and index < len(self._items) - 1:
            uid, after = self._items[index].item_uid, self._items[index + 1].item_uid
            self._call(
                "Plan not moved",
                lambda: self._re_client.item_move(uid=uid, after_uid=after),
            )

    def _on_remove_clicked(self):
        item = self._selected_item()
        if item is not None:
            self._call(
                "Plan not removed",
                lambda: self._re_client.item_remove(uid=item.item_uid),
            )

    def _on_edit_clicked(self):
        item = self._selected_item()
        if item is not None and self._edit_problem(item) is None:
            self.edit(item)

    def edit(self, item: QueueItem) -> QtPlanWidget:
        """Open ``item``'s plan form, filled in, to save back to the queue.

        Raises the form if it is already open.

        Raises
        ------
        KeyError
            If the plan is no longer in the catalog.
        TypeError
            If its arguments can't all be given by name.
        """
        key = item.item_uid
        editor = self._editors.get(key)
        if editor is None:
            values = self._re_client.plan_catalog[item.name].bind(
                item.args, item.kwargs
            )

            def save(kwargs):
                self._re_client.item_update(replace(item, args=(), kwargs=kwargs))

            # No parent: a parented window is kept above the main window.
            editor = QtPlanWidget(self._re_client, item.name, on_save=save)
            editor.set_values(values)
            editor.setWindowTitle(f"Edit Queued Plan \u2014 {item.name}")
            editor.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            editor.destroyed.connect(lambda *_: self._editors.pop(key, None))
            editor.resize(max(editor.sizeHint().width(), 480), 600)
            self._editors[key] = editor
        editor.show()
        editor.raise_()
        editor.activateWindow()
        return editor

    def closeEvent(self, event):
        self._queue.stop()
        for editor in list(self._editors.values()):
            editor.close()
        super().closeEvent(event)


class QtQueuePage(QWidget):
    """Pick any plan and fill in its form, beside the queue it is added to.

    The dropdown lists every plan a form can be built for, and offers them
    again whenever the plan catalog changes.

    Parameters
    ----------
    re_client : RunEngineClient
        The client whose plans to offer, and whose queue to show and edit.
    """

    def __init__(self, re_client: RunEngineClient, parent: QWidget | None = None):
        super().__init__(parent)
        self._re_client = re_client
        self._form: QtPlanWidget | None = None

        self._plans = QComboBox()
        self._plans.setToolTip("The plan to fill in and add to the queue")
        self._plans.currentIndexChanged.connect(
            lambda *_: self._show_form(self._plans.currentData())
        )
        left = QWidget()
        self._left_layout = QVBoxLayout(left)
        self._left_layout.setContentsMargins(0, 0, 0, 0)
        self._left_layout.addWidget(self._plans)

        self.queue_editor = QtQueueEditor(re_client)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.queue_editor)
        splitter.setStretchFactor(1, 1)
        vbox = QVBoxLayout()
        vbox.addWidget(splitter)
        self.setLayout(vbox)

        self._catalog = QtCatalog(re_client, self)
        self._catalog.changed.connect(lambda *_: self._offer_plans())
        self._offer_plans()

    @Slot()
    def _offer_plans(self):
        current = self._plans.currentData()
        plans = sorted(
            self._re_client.plan_catalog.gui_plans(), key=lambda plan: plan.name
        )
        self._plans.blockSignals(True)
        self._plans.clear()
        for plan in plans:
            self._plans.addItem(plan.name, plan.name)
            self._plans.setItemData(
                self._plans.count() - 1,
                plan.description or "",
                Qt.ItemDataRole.ToolTipRole,
            )
        self._plans.setCurrentIndex(max(self._plans.findData(current), 0))
        self._plans.blockSignals(False)
        if self._plans.currentData() != current:
            self._show_form(self._plans.currentData())

    def _show_form(self, name: str | None):
        if self._form is not None:
            self._form.close()
            self._left_layout.removeWidget(self._form)
            self._form.deleteLater()
            self._form = None
        if name is not None:
            self._form = QtPlanWidget(self._re_client, name)
            self._left_layout.addWidget(self._form, stretch=1)

    def closeEvent(self, event):
        self._catalog.stop()
        if self._form is not None:
            self._form.close()
        self.queue_editor.close()
        super().closeEvent(event)
