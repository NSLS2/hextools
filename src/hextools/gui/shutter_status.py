"""Clickable open/closed indicators for the beamline shutters."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from bluesky.protocols import Reading
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QGroupBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hextools.gui._client_signals import QtRunEngineState
from hextools.gui.plan_errors import show_plan_error
from hextools.gui.run_engine_clients import DeviceName, RunEngineClient
from hextools.gui.signal_status import QtSignalStatus

#: Called with ``(shutter name, open?)``; returns the plan name and its keywords.
ShutterPlanBuilder = Callable[[str, bool], tuple[str, dict[str, Any]]]

# (top, bottom) gradient stops.
_OPEN_COLORS = ("#7bb08a", "#4a7a58")
_CLOSED_COLORS = ("#c47d7d", "#8f4b4b")
_UNKNOWN_COLORS = ("#a5a5a5", "#707070")
_BUTTON_SIZE = 80
_INVALID_SEVERITY = 3


class _ShutterIndicator(QtSignalStatus):
    """``label: [button]`` row whose square button reflects, and toggles, a shutter.

    An INVALID alarm shows the shutter as unknown (grey).
    """

    show_units = False

    def __init__(self, re_client: RunEngineClient, name: str, parent=None):
        super().__init__(re_client, name, parent=parent)
        self.var_name = name
        self.is_open: bool | None = None
        self._blocked_reason: str | None = None
        self.label_widget.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._render()

    @property
    def button(self) -> QPushButton:
        """The clickable indicator."""
        return self._value  # type: ignore[return-value]

    def _make_value_widget(self) -> QWidget:
        button = QPushButton()
        button.setFixedSize(_BUTTON_SIZE, _BUTTON_SIZE)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def _show_reading(self, reading: Reading[Any]):
        invalid = int(reading.get("alarm_severity", 0)) >= _INVALID_SEVERITY
        self.is_open = None if invalid else bool(reading["value"])
        self._render()

    def set_blocked(self, reason: str | None):
        """Disable clicks, giving ``reason`` as the tooltip, or enable with None."""
        if reason == self._blocked_reason:
            return
        self._blocked_reason = reason
        self.button.setEnabled(reason is None)
        self.button.setCursor(
            Qt.CursorShape.PointingHandCursor
            if reason is None
            else Qt.CursorShape.ArrowCursor
        )
        self._update_tooltip()

    def _update_tooltip(self):
        if self._blocked_reason is not None:
            self.button.setToolTip(self._blocked_reason)
        else:
            action = "close" if self.is_open else "open"
            self.button.setToolTip(f"Click to {action} the {self.var_name}.")

    def _render(self):
        if self.is_open is None:
            top, bottom = _UNKNOWN_COLORS
        elif self.is_open:
            top, bottom = _OPEN_COLORS
        else:
            top, bottom = _CLOSED_COLORS
        self._update_tooltip()
        # The app theme's QPushButton min-width/min-height override setFixedSize
        # when the stylesheet is polished, so pin the size in the stylesheet too.
        inner = _BUTTON_SIZE - 2  # minus the 1px border on each side
        self.button.setStyleSheet(
            "QPushButton { border-radius: 6px; border: 1px solid rgba(0, 0, 0, 60);"
            " padding: 0;"
            f" min-width: {inner}px; max-width: {inner}px;"
            f" min-height: {inner}px; max-height: {inner}px;"
            " background: qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            f" stop:0 {top}, stop:1 {bottom}); }}"
        )


# TODO: Once ophyd as a service exists, this widget should use that when
# running in remote mode.
class QtShutterStatus(QWidget):
    """Shutter indicators; click one to toggle it.

    Parameters
    ----------
    re_client : RunEngineClient
        Runs the shutter plans, and reads the shutters' open/closed states.
    shutters : Iterable[str]
        Names, in the client's device registry, of explicitly selected boolean
        readbacks; also the device names passed to the plan.
    plan_builder : ShutterPlanBuilder
        Return the plan, and its keyword arguments, that opens or closes a named
        shutter. Device discovery, plan choice and settling policy belong to the
        caller.
    """

    def __init__(
        self,
        re_client: RunEngineClient,
        shutters: Iterable[str],
        plan_builder: ShutterPlanBuilder,
        parent=None,
    ):
        super().__init__(parent)
        self._re_client = re_client
        self._plan_builder = plan_builder
        self._pending = False

        self._indicators: list[_ShutterIndicator] = []
        rows = QVBoxLayout()
        rows.setContentsMargins(8, 6, 8, 6)
        for name in shutters:
            indicator = _ShutterIndicator(re_client, name)
            indicator.button.clicked.connect(lambda *_, i=indicator: self._toggle(i))
            rows.addWidget(indicator)
            self._indicators.append(indicator)
        width = max(
            (i.label_widget.sizeHint().width() for i in self._indicators), default=0
        )
        for indicator in self._indicators:
            indicator.label_widget.setFixedWidth(width)

        group_box = QGroupBox("Shutters")
        group_box.setLayout(rows)
        vbox = QVBoxLayout()
        vbox.addWidget(group_box)
        self.setLayout(vbox)
        self._state = QtRunEngineState(re_client, self)
        self._state.changed.connect(lambda *_: self._update_enabled())
        self._update_enabled()

    def _update_enabled(self):
        if self._re_client.state != "idle":
            reason = "Shutters cannot be actuated while the RunEngine is not idle."
        elif self._pending:
            reason = "A shutter actuation command is running."
        else:
            reason = None
        for indicator in self._indicators:
            indicator.set_blocked(reason)

    def _toggle(self, indicator: _ShutterIndicator):
        if self._pending or self._re_client.state != "idle":
            return
        name = indicator.var_name
        plan, kwargs = self._plan_builder(name, not indicator.is_open)
        # Reserve both controls before the deferred plan can run or Qt reenters.
        self._pending = True
        self._update_enabled()
        # Defer so the click handler returns before the plan runs.
        QTimer.singleShot(0, lambda: self._execute(plan, name, kwargs))

    def _execute(self, plan: str, name: str, kwargs: dict[str, Any]):
        try:
            if self._re_client.state != "idle":
                return
            self._re_client.execute_plan(plan, DeviceName(name), **kwargs)
        except Exception as error:
            show_plan_error(self, "Shutter actuation failed", error)
        finally:
            self._pending = False
            self._update_enabled()

    def closeEvent(self, event):
        self._state.stop()
        for indicator in self._indicators:
            indicator.close()
        super().closeEvent(event)
