"""Start, Stop and Clear buttons and an auto-start switch for a client's plan queue."""

from __future__ import annotations

from collections.abc import Callable

from qtpy.QtCore import Slot
from qtpy.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hextools.gui._client_signals import (
    QtPlanQueueChanges,
    QtQueueStatus,
    QtRunEngineState,
)
from hextools.gui.run_engine_clients import QueueError, RunEngineClient


class QtQueueControls(QWidget):
    """Plan queue controls, enabled to match the queue and the RunEngine.

    Start runs the queued plans, front first; Stop lets the running plan finish,
    then stops (pressed again, it cancels that); Clear empties the queue.
    With Auto-start, the queue runs whenever it has plans and the RunEngine is
    idle, so plans run as they are added.

    Parameters
    ----------
    re_client : RunEngineClient
        The client whose queue to control.
    """

    def __init__(self, re_client: RunEngineClient, parent: QWidget | None = None):
        super().__init__(parent)
        self._re_client = re_client

        self._start_button = self._button(
            "Start", "Run the queued plans, front first", self._on_start_clicked
        )
        self._stop_button = self._button("Stop", "", self._on_stop_clicked)
        self._clear_button = self._button(
            "Clear", "Remove every plan from the queue", self._on_clear_clicked
        )
        self._autostart = QCheckBox("Auto-start")
        self._autostart.setToolTip(
            "Run plans as they are added, whenever the RunEngine is idle"
        )
        self._autostart.toggled.connect(self._on_autostart_toggled)
        self._summary = QLabel()

        grid = QGridLayout()
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setSpacing(4)
        grid.addWidget(self._start_button, 0, 0)
        grid.addWidget(self._stop_button, 0, 1)
        grid.addWidget(self._clear_button, 0, 2)
        grid.addWidget(self._autostart, 1, 0)
        grid.addWidget(self._summary, 1, 1, 1, 2)

        group_box = QGroupBox("Queue")
        group_box.setLayout(grid)
        # Matches the Plan Execution controls, to fit the header's height.
        group_box.setStyleSheet("QPushButton { padding: 2px 10px; }")
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(group_box)
        self.setLayout(vbox)

        self._queue = QtPlanQueueChanges(re_client, self)
        self._queue.changed.connect(self._update)
        self._status = QtQueueStatus(re_client, self)
        self._status.changed.connect(self._update)
        self._re_state = QtRunEngineState(re_client, self)
        self._re_state.changed.connect(self._update)
        self._update()

    def _button(
        self, text: str, tooltip: str, on_click: Callable[[], None]
    ) -> QPushButton:
        button = QPushButton(text)
        button.setToolTip(tooltip)
        button.clicked.connect(lambda *_: on_click())
        return button

    @Slot()
    def _update(self, *_):
        queue = self._re_client.queue_get()
        status = self._re_client.queue_status()
        idle = self._re_client.state == "idle"
        queued = len(queue.items)

        self._start_button.setEnabled(not status.running and idle and bool(queued))
        self._stop_button.setEnabled(status.running)
        self._stop_button.setText("Cancel Stop" if status.stop_pending else "Stop")
        self._stop_button.setToolTip(
            "Keep running the queue after this plan"
            if status.stop_pending
            else "Stop the queue once the running plan finishes"
        )
        self._clear_button.setEnabled(bool(queued))
        self._autostart.blockSignals(True)
        self._autostart.setChecked(status.autostart)
        self._autostart.blockSignals(False)

        if status.stop_pending:
            state = "stopping"
        elif status.running:
            state = "running"
        else:
            state = "idle"
        self._summary.setText(f"{queued} queued, {state}")
        names = [item.name for item in queue.items]
        if queue.running_item is not None:
            names.insert(0, f"{queue.running_item.name} (running)")
        self._summary.setToolTip("\n".join(names) or "The queue is empty")

    def _call(self, title: str, action: Callable[[], None]):
        try:
            action()
        except QueueError as error:
            QMessageBox.warning(self, title, str(error))
        self._update()

    def _on_start_clicked(self):
        self._call("Queue not started", self._re_client.queue_start)

    def _on_stop_clicked(self):
        if self._re_client.queue_status().stop_pending:
            self._call("Queue stop not cancelled", self._re_client.queue_stop_cancel)
        else:
            self._call("Queue not stopped", self._re_client.queue_stop)

    def _on_clear_clicked(self):
        answer = QMessageBox.question(
            self,
            "Clear plan queue",
            "Remove all plans from the queue? This can't be undone.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._call("Queue not cleared", self._re_client.queue_clear)

    def _on_autostart_toggled(self, checked: bool):
        self._call(
            "Auto-start not changed", lambda: self._re_client.queue_autostart(checked)
        )

    def closeEvent(self, event):
        self._queue.stop()
        self._status.stop()
        self._re_state.stop()
        super().closeEvent(event)
