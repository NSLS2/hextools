"""Clickable open/closed indicators for two-state devices such as shutters."""

from __future__ import annotations

from collections.abc import Mapping
from types import FunctionType
from typing import Any

from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hextools.gui._ipython import show_plan_error
from hextools.gui.session_client import SessionClient

# (top, bottom) gradient stops.
_OPEN_COLORS = ("#7bb08a", "#4a7a58")
_CLOSED_COLORS = ("#c47d7d", "#8f4b4b")
_UNKNOWN_COLORS = ("#a5a5a5", "#707070")
_BUTTON_SIZE = 80


class _ShutterButton(QPushButton):
    """Square colored button reflecting, and toggling, one shutter."""

    def __init__(self, var_name: str, parent=None):
        super().__init__(parent)
        self.var_name = var_name
        self.is_open: bool | None = None
        self.setFixedSize(_BUTTON_SIZE, _BUTTON_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._render()

    def set_state(self, is_open: bool):
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
        self.setToolTip(f"{self.var_name} is {state}. Click to {action}.")
        # The app theme's QPushButton min-width/min-height override setFixedSize
        # when the stylesheet is polished, so pin the size in the stylesheet too.
        inner = _BUTTON_SIZE - 2  # minus the 1px border on each side
        self.setStyleSheet(
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
    """Open/closed indicators; click one to toggle it (IPython session only).

    Parameters
    ----------
    session : SessionClient
        The IPython session; it reports each device's readings.
    device_type : type
        Devices shown: those in the session's namespace of this type. Each must
        report a truthy reading when open.
    open_plan, close_plan : function
        Plans run as ``RE(plan(device, **plan_kwargs))`` in the session to
        toggle a device; imported there from their module if needed.
    plan_kwargs : mapping, optional
        Extra keyword arguments for both plans.
    title : str, optional
        Group box title. Default: ``"Shutters"``.
    """

    def __init__(
        self,
        session: SessionClient,
        *,
        device_type: type,
        open_plan: FunctionType,
        close_plan: FunctionType,
        plan_kwargs: Mapping[str, Any] | None = None,
        title: str = "Shutters",
        parent=None,
    ):
        super().__init__(parent)
        self._session = session
        self._open_plan = open_plan
        self._close_plan = close_plan
        self._plan_kwargs = dict(plan_kwargs or {})

        wanted = f"{device_type.__module__}.{device_type.__qualname__}"
        # Top-level names only; the session lists each device once.
        names = sorted(
            path
            for path in session.device_paths()
            if path.isidentifier() and wanted in (session.device_classes(path) or [])
        )

        self._buttons: dict[str, _ShutterButton] = {}
        grid = QGridLayout()
        grid.setContentsMargins(8, 6, 8, 6)
        for row, name in enumerate(names):
            button = _ShutterButton(name)
            button.clicked.connect(lambda *_, b=button: self._toggle(b))
            label = QLabel(name)
            label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            grid.addWidget(label, row, 0)
            grid.addWidget(button, row, 1)
            self._buttons[name] = button

        group_box = QGroupBox(title)
        group_box.setLayout(grid)
        vbox = QVBoxLayout()
        vbox.addWidget(group_box)
        self.setLayout(vbox)

        session.reading.connect(self._on_reading)
        if names:
            session.watch(names)

    @Slot(str, object)
    def _on_reading(self, name: str, value: Any):
        button = self._buttons.get(name)
        if button is not None:
            button.set_state(bool(value))

    def _toggle(self, button: _ShutterButton):
        session = self._session
        if session.state != "idle":
            QMessageBox.warning(
                self,
                "RunEngine busy",
                "Cannot actuate a shutter while the RunEngine is busy.",
            )
            return
        plan = self._close_plan if button.is_open else self._open_plan
        args = "".join(f", {k}={v!r}" for k, v in self._plan_kwargs.items())
        code = f"{session.re_name}({plan.__name__}({button.var_name}{args}))"
        if plan.__module__ != "__main__":
            code = f"from {plan.__module__} import {plan.__name__}; {code}"
        session.execute(code, self._on_done)

    def _on_done(self, error: BaseException | None):
        if error is not None:
            show_plan_error(self, "Shutter actuation failed", error)
