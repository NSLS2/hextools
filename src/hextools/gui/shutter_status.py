"""Clickable open/closed indicators for the beamline shutters."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from bluesky import RunEngine
from qtpy.QtCore import QTimer, Qt, Signal, Slot
from qtpy.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hextools.gui._ipython import run_in_ipython, show_plan_error
from hextools.photon_delivery_system.shutter import (
    Shutter,
    ensure_shutter_closed,
    ensure_shutter_open,
)

# (top, bottom) gradient stops.
_OPEN_COLORS = ("#7bb08a", "#4a7a58")
_CLOSED_COLORS = ("#c47d7d", "#8f4b4b")
_UNKNOWN_COLORS = ("#a5a5a5", "#707070")
_BUTTON_SIZE = 80


class _ShutterButton(QPushButton):
    """Square colored button reflecting, and toggling, one shutter."""

    _signal_state = Signal(bool)

    def __init__(self, var_name: str, shutter: Shutter, loop: asyncio.AbstractEventLoop, parent=None):
        super().__init__(parent)
        self.var_name = var_name
        self.shutter = shutter
        self._loop = loop
        self.is_open: bool | None = None
        self.setFixedSize(_BUTTON_SIZE, _BUTTON_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._signal_state.connect(self._set_state)
        self._render()
        # ophyd-async subscriptions must be made on the bluesky event loop, and
        # callbacks arrive on that thread; marshal back to Qt via signal.
        loop.call_soon_threadsafe(shutter.subscribe_reading, self._on_reading)

    def _on_reading(self, readings: Mapping[str, Any]):
        for reading in readings.values():
            self._signal_state.emit(bool(reading["value"]))

    @Slot(bool)
    def _set_state(self, is_open: bool):
        self.is_open = is_open
        self._render()

    def _render(self):
        if self.is_open is None:
            (top, bottom), state = _UNKNOWN_COLORS, "unknown"
        elif self.is_open:
            (top, bottom), state = _OPEN_COLORS, "open"
        else:
            (top, bottom), state = _CLOSED_COLORS, "closed"
        action = "close" if self.is_open else "open"
        self.setToolTip(f"{self.shutter.name} is {state}. Click to {action}.")
        # The app theme's QPushButton min-width/min-height override setFixedSize
        # when the stylesheet is polished, so pin the size in the stylesheet too.
        inner = _BUTTON_SIZE - 2  # minus the 1px border on each side
        self.setStyleSheet(
            "QPushButton { border-radius: 6px; border: 1px solid rgba(0, 0, 0, 60); padding: 0;"
            f" min-width: {inner}px; max-width: {inner}px;"
            f" min-height: {inner}px; max-height: {inner}px;"
            f" background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {top}, stop:1 {bottom}); }}"
        )

    def unsubscribe(self):
        def _clear():
            try:
                self.shutter.clear_sub(self._on_reading)
            except KeyError:
                pass

        self._loop.call_soon_threadsafe(_clear)


# TODO: Once ophyd as a service exists, this widget should use that when
# running in remote mode.
class QtShutterStatus(QWidget):
    """Shutter indicators; click one to toggle it (local mode only).

    Parameters
    ----------
    re : RunEngine
        The in-process RunEngine.
    namespace : Mapping[str, object]
        Namespace searched for :class:`Shutter` objects, typically IPython's
        ``user_ns``. Aliases of the same shutter are shown once.
    """

    def __init__(self, re: RunEngine, namespace: Mapping[str, Any], parent=None):
        super().__init__(parent)
        self._re = re
        self._namespace = namespace

        shutters: dict[int, tuple[str, Shutter]] = {}
        for name, obj in sorted(namespace.items()):
            if not name.startswith("_") and isinstance(obj, Shutter):
                shutters.setdefault(id(obj), (name, obj))

        self._buttons: list[_ShutterButton] = []
        grid = QGridLayout()
        grid.setContentsMargins(8, 6, 8, 6)
        for row, (name, shutter) in enumerate(shutters.values()):
            button = _ShutterButton(name, shutter, re.loop)
            button.clicked.connect(lambda *_, b=button: self._toggle(b))
            label = QLabel(name)
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(label, row, 0)
            grid.addWidget(button, row, 1)
            self._buttons.append(button)

        group_box = QGroupBox("Shutters")
        group_box.setLayout(grid)
        vbox = QVBoxLayout()
        vbox.addWidget(group_box)
        self.setLayout(vbox)

    def _toggle(self, button: _ShutterButton):
        if self._re.state != "idle":
            QMessageBox.warning(
                self, "RunEngine busy", "Cannot actuate a shutter while the RunEngine is busy."
            )
            return
        plan = ensure_shutter_closed if button.is_open else ensure_shutter_open
        # Make the plan resolvable when the command is echoed and run in IPython.
        self._namespace.setdefault(plan.__name__, plan)  # type: ignore[attr-defined]
        code = f"RE({plan.__name__}({button.var_name}, allow_actuation=True))"
        # Defer so the click handler returns before IPython executes the cell.
        QTimer.singleShot(0, lambda: self._execute_cell(code))

    def _execute_cell(self, code: str):
        error = run_in_ipython(code)
        if error is not None:
            show_plan_error(self, "Shutter actuation failed", error)

    def closeEvent(self, event):
        for button in self._buttons:
            button.unsubscribe()
        super().closeEvent(event)
