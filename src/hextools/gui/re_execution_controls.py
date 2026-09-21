"""Plan execution controls with an optional local-RunEngine mode.

:class:`QtReExecutionControls` extends the bluesky-widgets execution-controls
widget (Pause/Resume/Stop/Abort/Halt). In the default (Queue Server) mode it
behaves exactly like the upstream widget, driven by ``model.events``. When
``local`` is True it drives an in-process RunEngine directly: the buttons call
``RE.request_pause``/``resume``/``stop``/``abort``/``halt``, button enablement
tracks ``RE.state`` via a poll timer, and the Queue-Server-only ``Ctrl-C``
button is hidden.
"""

from __future__ import annotations

from collections.abc import Mapping

from bluesky_widgets.qt.run_engine_client import (
    QtReExecutionControls as _QtReExecutionControls,
)
from bluesky_widgets.qt.threading import FunctionWorker
from qtpy.QtCore import QTimer


class _NullSignal:
    def connect(self, *args, **kwargs):
        pass


class _NullEvents:
    status_changed = _NullSignal()


class _NullModel:
    """Stand-in model so the base widget's status subscription is a no-op."""

    events = _NullEvents()


class QtReExecutionControls(_QtReExecutionControls):
    """bluesky-widgets plan execution controls with an optional local-RE mode.

    Parameters
    ----------
    model : object, optional
        The Queue Server client model. Ignored when ``local`` is True.
    parent : QWidget, optional
        Parent widget.
    local : bool, optional
        Drive an in-process RunEngine instead of the Queue Server.
    re : object, optional
        A RunEngine instance to control (local mode only). If omitted, resolved
        by name from ``namespace``.
    re_name : str, optional
        Name of the RunEngine in the namespace. Default ``"RE"``.
    namespace : Mapping[str, object], optional
        Namespace to resolve ``re_name`` from. Defaults to the IPython
        ``user_ns``.
    poll_period : float, optional
        Interval between ``RE.state`` reads, in seconds. Default: 0.5.
    """

    def __init__(
        self,
        model=None,
        parent=None,
        *,
        local: bool = False,
        re=None,
        re_name: str = "RE",
        namespace: Mapping[str, object] | None = None,
        poll_period: float = 0.5,
    ):
        self._local = local
        self._re = re
        self._re_name = re_name
        self._namespace = namespace
        self._timer = None
        self._resume_worker = None

        super().__init__(_NullModel() if local else model, parent)

        if not local:
            return

        # Kernel interrupt is a Queue Server concept; hide it in local mode.
        self._pb_kernel_interrupt.setVisible(False)

        self._timer = QTimer(self)
        self._timer.setInterval(int(poll_period * 1000))
        self._timer.timeout.connect(self._poll_local_state)
        self._timer.start()
        self._poll_local_state()

    def _resolve_re(self):
        if self._re is not None:
            return self._re
        namespace = self._namespace
        if namespace is None:
            namespace = _ipython_namespace()
        return namespace.get(self._re_name)

    def _poll_local_state(self):
        run_engine = self._resolve_re()
        state = getattr(run_engine, "state", None) if run_engine is not None else None
        running = state == "running"
        paused = state == "paused"
        self._pb_plan_pause_deferred.setEnabled(running)
        self._pb_plan_pause_immediate.setEnabled(running)
        self._pb_plan_resume.setEnabled(paused)
        self._pb_plan_stop.setEnabled(paused)
        self._pb_plan_abort.setEnabled(paused)
        self._pb_plan_halt.setEnabled(paused)

    def _call_re(self, method, *args, **kwargs):
        run_engine = self._resolve_re()
        if run_engine is None:
            return
        try:
            getattr(run_engine, method)(*args, **kwargs)
        except Exception as ex:  # noqa: BLE001 - mirror upstream best-effort behavior
            print(f"Exception: {ex}")

    def _pb_plan_pause_deferred_clicked(self):
        if not self._local:
            return super()._pb_plan_pause_deferred_clicked()
        self._call_re("request_pause", True)

    def _pb_plan_pause_immediate_clicked(self):
        if not self._local:
            return super()._pb_plan_pause_immediate_clicked()
        self._call_re("request_pause", False)

    def _pb_plan_resume_clicked(self):
        if not self._local:
            return super()._pb_plan_resume_clicked()
        # resume() blocks until the plan pauses or completes; run it off-thread.
        run_engine = self._resolve_re()
        if run_engine is None or self._resume_worker is not None:
            return
        self._resume_worker = FunctionWorker(run_engine.resume)
        self._resume_worker.finished.connect(self._on_resume_finished)
        self._resume_worker.start()

    def _on_resume_finished(self):
        self._resume_worker = None

    def _pb_plan_stop_clicked(self):
        if not self._local:
            return super()._pb_plan_stop_clicked()
        self._call_re("stop")

    def _pb_plan_abort_clicked(self):
        if not self._local:
            return super()._pb_plan_abort_clicked()
        self._call_re("abort")

    def _pb_plan_halt_clicked(self):
        if not self._local:
            return super()._pb_plan_halt_clicked()
        self._call_re("halt")

    def closeEvent(self, event):
        if self._timer is not None:
            self._timer.stop()
        super().closeEvent(event)


def _ipython_namespace() -> Mapping[str, object]:
    """Return the active IPython ``user_ns``, or an empty mapping if none."""
    try:
        from IPython.core.getipython import get_ipython

        ipython = get_ipython()
    except Exception:  # pragma: no cover - IPython always present in this env
        return {}
    if ipython is None:
        return {}
    return ipython.user_ns
