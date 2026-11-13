"""Pause, Resume, Stop, Abort and Halt buttons for a RunEngine."""

from __future__ import annotations

from collections.abc import Callable

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QGridLayout, QGroupBox, QPushButton, QVBoxLayout, QWidget

from hextools.gui._client_signals import QtRunEngineState
from hextools.gui.plan_errors import show_plan_error
from hextools.gui.run_engine_clients import RunEngineClient


class QtReExecutionControls(QWidget):
    """Plan execution controls, enabled to match the RunEngine's state.

    Pause is offered while a plan runs; Resume, Stop, Abort and Halt while it is
    paused.

    Parameters
    ----------
    re_client : RunEngineClient
        The RunEngine to control.
    """

    def __init__(
        self,
        re_client: RunEngineClient,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._re_client = re_client
        self._resume_pending = False

        self._pb_plan_pause_deferred = self._button(
            "Pause: Deferred",
            "Pause at the next checkpoint",
            self._pb_plan_pause_deferred_clicked,
        )
        self._pb_plan_pause_immediate = self._button(
            "Pause: Immediate",
            "Pause now",
            self._pb_plan_pause_immediate_clicked,
        )
        self._pb_plan_resume = self._button(
            "Resume", "Continue the paused plan", self._pb_plan_resume_clicked
        )
        self._pb_plan_stop = self._button(
            "Stop", "End the paused plan successfully", self._pb_plan_stop_clicked
        )
        self._pb_plan_abort = self._button(
            "Abort", "End the paused plan as failed", self._pb_plan_abort_clicked
        )
        self._pb_plan_halt = self._button(
            "Halt", "End the paused plan without cleanup", self._pb_plan_halt_clicked
        )

        grid = QGridLayout()
        grid.addWidget(self._pb_plan_pause_deferred, 0, 0)
        grid.addWidget(self._pb_plan_pause_immediate, 0, 1)
        grid.addWidget(self._pb_plan_resume, 0, 2)
        grid.addWidget(self._pb_plan_stop, 1, 0)
        grid.addWidget(self._pb_plan_abort, 1, 1)
        grid.addWidget(self._pb_plan_halt, 1, 2)
        group_box = QGroupBox("Plan Execution")
        group_box.setLayout(grid)
        vbox = QVBoxLayout()
        vbox.addWidget(group_box)
        self.setLayout(vbox)

        self._state = QtRunEngineState(re_client, self)
        self._state.changed.connect(self._update_buttons)
        self._update_buttons()

    def _button(
        self, text: str, tooltip: str, on_click: Callable[[], None]
    ) -> QPushButton:
        button = QPushButton(text)
        button.setToolTip(tooltip)
        button.clicked.connect(lambda *_: on_click())
        return button

    def _update_buttons(self, state: str | None = None):
        if state is None:
            state = self._re_client.state
        running = state == "running"
        paused = state == "paused"
        self._pb_plan_pause_deferred.setEnabled(running)
        self._pb_plan_pause_immediate.setEnabled(running)
        self._pb_plan_resume.setEnabled(paused and not self._resume_pending)
        self._pb_plan_stop.setEnabled(paused)
        self._pb_plan_abort.setEnabled(paused)
        self._pb_plan_halt.setEnabled(paused)

    def _call(self, title: str, action: Callable[[], None]):
        try:
            action()
        except Exception as error:
            show_plan_error(self, title, error)
        self._update_buttons()

    def _pb_plan_pause_deferred_clicked(self):
        self._call("Pause failed", lambda: self._re_client.pause(defer=True))

    def _pb_plan_pause_immediate_clicked(self):
        self._call("Pause failed", lambda: self._re_client.pause(defer=False))

    def _pb_plan_resume_clicked(self):
        if self._resume_pending:
            return
        # A second click before resume() returns would queue another resume().
        self._resume_pending = True
        self._pb_plan_resume.setEnabled(False)
        # Resuming installs a SIGINT handler, which only the main thread may do,
        # so run it after this click returns, never on a worker thread.
        QTimer.singleShot(0, self._resume)

    def _resume(self):
        try:
            self._re_client.resume()
        except Exception as error:
            show_plan_error(self, "Resumed plan failed", error)
        finally:
            self._resume_pending = False
            self._update_buttons()

    def _pb_plan_stop_clicked(self):
        self._call("Stop failed", self._re_client.stop)

    def _pb_plan_abort_clicked(self):
        self._call("Abort failed", self._re_client.abort)

    def _pb_plan_halt_clicked(self):
        self._call("Halt failed", self._re_client.halt)

    def closeEvent(self, event):
        self._state.stop()
        super().closeEvent(event)
