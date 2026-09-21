from functools import partial
from typing import Any

from bluesky import RunEngine
from bluesky_widgets.models.run_engine_client import RunEngineClient
from qtpy.QtWidgets import QVBoxLayout, QWidget, QLabel, QProgressBar, QHBoxLayout
from qtpy.QtCore import Qt, Signal
from ._threading import FunctionWorker

class _ReDeviceProgressBar(QWidget):
    """
    A single-line progress indicator for one RunEngine status object (typically one device).

    Layout: ``name  [start]  [====== current | elapsed | pct% ======]  -> target (ETA)``.
    The start position is shown in front of the bar and the target with the estimated time
    remaining after it. The current position, elapsed time and percentage are drawn on the bar.
    """

    def __init__(self, name, parent=None):
        super().__init__(parent)

        self._lb_name = QLabel(f"{name}")
        self._lb_name.setStyleSheet("font-weight: bold;")
        self._lb_name.setMinimumWidth(120)

        self._lb_start = QLabel("")
        self._lb_start.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._lb_start.setMinimumWidth(90)

        self._lb_end = QLabel("")
        self._lb_end.setMinimumWidth(170)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setMinimumWidth(200)

        # Last known values, used to fill in fields omitted from later updates (e.g. the
        # final "done" update, which may not repeat the position/unit information).
        self._last = {
            "precision": None,
            "unit": None,
            "initial": None,
            "current": None,
            "target": None,
            "time_elapsed": None,
            "time_remaining": None,
        }

        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self._lb_name)
        hbox.addWidget(self._lb_start)
        hbox.addWidget(self._progress_bar, stretch=1)
        hbox.addWidget(self._lb_end)
        self.setLayout(hbox)

    def update_progress(self, msg: dict[str, Any]):
        # Fall back to the last known value for any field omitted from this update, so the
        # display keeps its last values (e.g. on the final update) instead of showing "N/A".
        def coalesce(key):
            value = msg.get(key)
            if value is None:
                value = self._last[key]
            else:
                self._last[key] = value
            return value

        precision = coalesce("precision")
        unit = coalesce("unit") or ""
        initial = coalesce("initial")
        current = coalesce("current")
        target = coalesce("target")
        time_elapsed = coalesce("time_elapsed")
        time_remaining = coalesce("time_remaining")
        done = bool(msg.get("done"))

        def fmt_pos(value):
            if value is None:
                return "N/A"
            if isinstance(precision, int) and isinstance(value, (int, float)):
                text = f"{value:.{precision}f}"
            else:
                text = f"{value}"
            return f"{text} {unit}".strip() if unit else text

        # Progress as fraction complete in [0, 1]. Prefer computing it from positions
        # (matches bluesky's own progress bar). ophyd reports ``fraction`` as the fraction
        # *remaining*, so convert it when positions are unavailable.
        progress = None
        if None not in (initial, current, target):
            span = abs(target - initial)
            if span:
                progress = abs(current - initial) / span
        if progress is None and msg.get("fraction") is not None:
            progress = 1.0 - msg["fraction"]
        if done:
            progress = 1.0
            if target is not None:
                current = target

        if progress is not None:
            self._progress_bar.setValue(int(max(0.0, min(1.0, progress)) * 1000))

        # Estimate the remaining time if the status object did not report it.
        if time_remaining is None and time_elapsed is not None and progress:
            time_remaining = 0.0 if progress >= 1.0 else time_elapsed * (1.0 - progress) / progress

        elapsed_str = "" if time_elapsed is None else f"{time_elapsed:.1f}s"
        remaining_str = "?" if time_remaining is None else f"{time_remaining:.1f}s"

        bar_text = fmt_pos(current)
        if elapsed_str:
            bar_text += f"  |  {elapsed_str}"
        bar_text += "  |  %p%"
        self._progress_bar.setFormat(bar_text)

        self._lb_start.setText(fmt_pos(initial))
        eta = "done" if done or (progress is not None and progress >= 1.0) else f"ETA {remaining_str}"
        self._lb_end.setText(f"\u2192 {fmt_pos(target)}  ({eta})")


class _WaitingHookDispatcher:
    """A RunEngine ``waiting_hook`` that forwards watcher updates to a monitor.

    Installed as ``RE.waiting_hook``. Any previously-installed hook (e.g. a
    terminal ``ProgressBarManager``) is preserved in ``previous_hook`` and still
    receives every call, so existing behavior keeps working alongside the GUI.
    """

    def __init__(self, monitor: "QtReWaitingHookMonitor", previous_hook=None):
        self._monitor = monitor
        self._previous_hook = previous_hook
        self._names: dict[int, str] = {}  # id(status) -> device name

    def __call__(self, status_objects):
        # Pass through to any pre-existing hook first.
        if self._previous_hook is not None:
            try:
                self._previous_hook(status_objects)
            except Exception:
                pass

        if status_objects is None:
            self._names.clear()
            self._monitor.signal_completed.emit()
            return

        for status in status_objects:
            if not hasattr(status, "watch"):
                continue
            try:
                status.watch(partial(self._on_watch, status))
            except Exception:
                pass
            try:
                status.add_callback(self._on_finished)
            except Exception:
                pass

    def _on_watch(self, status, *, name=None, **kwargs):
        if name is None:
            name = self._names.get(id(status))
        else:
            self._names[id(status)] = name
        msg = dict(kwargs)
        msg["name"] = name
        msg["done"] = bool(getattr(status, "done", False))
        self._monitor.signal_progress_update.emit(msg)

    def _on_finished(self, status=None, *args, **kwargs):
        name = self._names.get(id(status)) if status is not None else None
        if name is None and status is not None:
            name = getattr(getattr(status, "device", None), "name", None)
        if name is None:
            return
        self._monitor.signal_progress_update.emit({"name": name, "done": True})


class QtReWaitingHookMonitor(QWidget):
    """
    Displays live progress bars for RunEngine ``waiting_hook``/watcher updates. One progress
    bar is shown per device (status object), reporting the start position, target, current
    position and estimated time remaining. All progress bars are cleared when the wait
    completes.

    Accepts either a Queue Server ``RunEngineClient`` (progress is streamed from RE Manager)
    or a raw in-process ``RunEngine`` (the widget installs itself as ``RE.waiting_hook``,
    chaining to any hook that was already set).
    """

    signal_progress_update = Signal(object)
    signal_completed = Signal()

    def __init__(self, re_client: RunEngineClient | RunEngine, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._re_client = re_client
        self._hook: _WaitingHookDispatcher | None = None
        self._previous_hook = None
        self._thread = None

        # Maps device name -> _ReDeviceProgressBar
        self._progress_bars = {}

        self._lb_status = QLabel("Waiting for updates\u2026")

        self._bars_layout = QVBoxLayout()
        self._bars_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        bars_container = QWidget()
        bars_container.setLayout(self._bars_layout)

        vbox = QVBoxLayout()
        vbox.addWidget(self._lb_status)
        vbox.addWidget(bars_container)
        vbox.addStretch()
        self.setLayout(vbox)

        self.signal_progress_update.connect(self._handle_progress_msg)
        self.signal_completed.connect(self._on_completed)

        if isinstance(self._re_client, RunEngineClient):
            self._re_client.start_re_progress_monitoring()
            self._start_thread()
        else:
            self._install_waiting_hook()

    # -- Queue Server (RE Manager) path ---------------------------------------

    def _start_thread(self):
        self._thread = FunctionWorker(self._re_client.re_progress_monitoring_thread)
        if self._thread.returned is None:
            raise RuntimeError("Failed to connect to the returned signal of the thread.")
        if self._thread.finished is None:
            raise RuntimeError("Failed to connect to the finished signal of the thread.")
        self._thread.returned.connect(self._process_new_progress_update)
        self._thread.finished.connect(self._finished_receiving_progress_update)
        self._thread.start()

    def _finished_receiving_progress_update(self):
        self._start_thread()

    def _process_new_progress_update(self, result):
        if not result:
            return
        _, msg = result
        self._handle_progress_msg(msg)

    # -- In-process RunEngine path --------------------------------------------

    def _install_waiting_hook(self):
        run_engine = self._re_client
        self._previous_hook = getattr(run_engine, "waiting_hook", None)
        self._hook = _WaitingHookDispatcher(self, self._previous_hook)
        run_engine.waiting_hook = self._hook

    def _uninstall_waiting_hook(self):
        run_engine = self._re_client
        if isinstance(run_engine, RunEngineClient) or self._hook is None:
            return
        # Only restore if our hook is still the active one.
        if getattr(run_engine, "waiting_hook", None) is self._hook:
            run_engine.waiting_hook = self._previous_hook
        self._hook = None

    # -- Shared display logic --------------------------------------------------

    def _handle_progress_msg(self, msg):
        if not msg:
            return

        if msg.get("completed"):
            self._on_completed()
            return

        name = msg.get("name")
        if name is None:
            return

        bar = self._progress_bars.get(name)
        if bar is None:
            bar = _ReDeviceProgressBar(name)
            self._progress_bars[name] = bar
            self._bars_layout.addWidget(bar)

        bar.update_progress(msg)
        self._lb_status.setText("Watching device progress\u2026")

    def _on_completed(self):
        self._clear_progress_bars()
        self._lb_status.setText("Waiting for updates\u2026")

    def _clear_progress_bars(self):
        for bar in self._progress_bars.values():
            self._bars_layout.removeWidget(bar)
            bar.deleteLater()
        self._progress_bars = {}

    def closeEvent(self, event):
        self._uninstall_waiting_hook()
        super().closeEvent(event)

    def __del__(self):
        try:
            if isinstance(self._re_client, RunEngineClient):
                self._re_client.stop_re_progress_monitoring()
            else:
                self._uninstall_waiting_hook()
        except Exception:
            pass