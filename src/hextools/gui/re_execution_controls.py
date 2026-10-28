"""Plan execution controls for the Queue Server or an IPython session.

:class:`QtReExecutionControls` extends the bluesky-widgets execution-controls
widget (Pause/Resume/Stop/Abort/Halt). Given a Queue Server model it behaves
exactly like the upstream widget. Given a
:class:`~hextools.gui.session_client.SessionClient` it drives the session's
RunEngine: Pause is sent straight to the RunEngine, while Resume, Stop, Abort
and Halt run as ``RE.resume()`` etc. at the session's prompt, where they show
in the terminal. Button enablement follows the session's RunEngine state, and
the Queue-Server-only ``Ctrl-C`` button is hidden.
"""

from __future__ import annotations

from bluesky_widgets.qt.run_engine_client import (
    QtReExecutionControls as _QtReExecutionControls,
)

from hextools.gui._ipython import show_plan_error
from hextools.gui.session_client import SessionClient


class _NullSignal:
    def connect(self, *args, **kwargs):
        pass


class _NullEvents:
    status_changed = _NullSignal()


class _NullModel:
    """Stand-in model so the base widget's status subscription is a no-op."""

    events = _NullEvents()


class QtReExecutionControls(_QtReExecutionControls):
    """bluesky-widgets plan execution controls, also for an IPython session.

    Parameters
    ----------
    model : object, optional
        The Queue Server client model. Ignored when ``session`` is given.
    parent : QWidget, optional
        Parent widget.
    session : SessionClient, optional
        Drive this session's RunEngine instead of the Queue Server.
    """

    def __init__(
        self, model=None, parent=None, *, session: SessionClient | None = None
    ):
        self._session = session
        self._command_pending = False
        super().__init__(_NullModel() if session is not None else model, parent)
        if session is None:
            return
        # Kernel interrupt is a Queue Server concept; hide it here.
        self._pb_kernel_interrupt.setVisible(False)
        session.state_changed.connect(self._update_buttons)
        self._update_buttons()

    def _update_buttons(self, *_):
        state = self._session.state if self._session is not None else None
        running = state == "running"
        paused = state == "paused"
        idle_for_commands = paused and not self._command_pending
        self._pb_plan_pause_deferred.setEnabled(running)
        self._pb_plan_pause_immediate.setEnabled(running)
        self._pb_plan_resume.setEnabled(idle_for_commands)
        self._pb_plan_stop.setEnabled(idle_for_commands)
        self._pb_plan_abort.setEnabled(idle_for_commands)
        self._pb_plan_halt.setEnabled(idle_for_commands)

    def _run(self, method: str):
        session = self._session
        if session is None or self._command_pending:
            return
        # A second click before the first returns would queue another command.
        self._command_pending = True
        self._update_buttons()
        session.execute(f"{session.re_name}.{method}()", self._on_done)

    def _on_done(self, error: BaseException | None):
        self._command_pending = False
        self._update_buttons()
        if error is not None:
            show_plan_error(self, "Plan failed", error)

    def _pb_plan_pause_deferred_clicked(self):
        if self._session is None:
            return super()._pb_plan_pause_deferred_clicked()
        self._session.request_pause(defer=True)

    def _pb_plan_pause_immediate_clicked(self):
        if self._session is None:
            return super()._pb_plan_pause_immediate_clicked()
        self._session.request_pause(defer=False)

    def _pb_plan_resume_clicked(self):
        if self._session is None:
            return super()._pb_plan_resume_clicked()
        self._run("resume")

    def _pb_plan_stop_clicked(self):
        if self._session is None:
            return super()._pb_plan_stop_clicked()
        self._run("stop")

    def _pb_plan_abort_clicked(self):
        if self._session is None:
            return super()._pb_plan_abort_clicked()
        self._run("abort")

    def _pb_plan_halt_clicked(self):
        if self._session is None:
            return super()._pb_plan_halt_clicked()
        self._run("halt")
