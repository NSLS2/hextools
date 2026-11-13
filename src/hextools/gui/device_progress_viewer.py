"""Widgets showing live RunEngine device progress bars."""

from typing import Any

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from .run_engine_clients import RunEngineClient


class _ReDeviceProgressBar(QWidget):
    """Single-line progress indicator for one RunEngine status object.

    Typically one status object corresponds to one device.

    Layout:
    ``name  [start]  [====== current | elapsed | pct% ======]  -> target (ETA)``.
    The start position is shown in front of the bar and the target with the
    estimated time remaining after it. The current position, elapsed time and
    percentage are drawn on the bar.
    """

    def __init__(self, name: str, parent: QWidget | None = None):
        super().__init__(parent)

        self._lb_name = QLabel(f"{name}")
        self._lb_name.setStyleSheet("font-weight: bold;")
        self._lb_name.setMinimumWidth(120)

        self._lb_start = QLabel("")
        self._lb_start.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
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
        # Fall back to the last known value for any field omitted from this update,
        # so the display keeps its last values (e.g. on the final update) instead of
        # showing "N/A".
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
        # (matches bluesky's own progress bar). ophyd reports ``fraction`` as the
        # fraction *remaining*, so convert it when positions are unavailable.
        progress = None
        if initial is not None and current is not None and target is not None:
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
            time_remaining = (
                0.0 if progress >= 1.0 else time_elapsed * (1.0 - progress) / progress
            )

        elapsed_str = "" if time_elapsed is None else f"{time_elapsed:.1f}s"
        remaining_str = "?" if time_remaining is None else f"{time_remaining:.1f}s"

        bar_text = fmt_pos(current)
        if elapsed_str:
            bar_text += f"  |  {elapsed_str}"
        bar_text += "  |  %p%"
        self._progress_bar.setFormat(bar_text)

        self._lb_start.setText(fmt_pos(initial))
        eta = (
            "done"
            if done or (progress is not None and progress >= 1.0)
            else f"ETA {remaining_str}"
        )
        self._lb_end.setText(f"\u2192 {fmt_pos(target)}  ({eta})")


class QtReWaitingHookMonitor(QWidget):
    """Display live progress bars for RunEngine ``waiting_hook``/watcher updates.

    One progress bar is shown per device (status object), reporting the start
    position, target, current position and estimated time remaining. All progress
    bars are cleared when the wait completes.

    Parameters
    ----------
    re_client : RunEngineClient
        The RunEngine whose progress stream is shown.
    """

    # The client may call back from any thread; this queued signal hops to Qt's.
    signal_progress_update = Signal(object)

    def __init__(self, re_client: RunEngineClient, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

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
        token = re_client.subscribe_progress(self.signal_progress_update.emit)
        subscribed = [True]

        def unsubscribe(*_):
            if subscribed:
                subscribed.clear()
                re_client.unsubscribe_progress(token)

        self._unsubscribe = unsubscribe
        # Also on deletion without a close; holds no reference to this widget.
        self.destroyed.connect(unsubscribe)

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
        self._unsubscribe()
        super().closeEvent(event)
