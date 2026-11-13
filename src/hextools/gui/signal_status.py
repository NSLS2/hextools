"""Label showing the live value of a signal or device."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from bluesky.protocols import Reading
from qtpy.QtCore import Qt, Signal, Slot
from qtpy.QtWidgets import QHBoxLayout, QLabel, QWidget

if TYPE_CHECKING:
    from hextools.gui.run_engine_clients import RunEngineClient


def format_value(value: Any, precision: int = 3) -> str:
    """Format a bool, int, float, or string reading value for display.

    Raises
    ------
    TypeError
        If ``value`` is of any other type.
    """
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        return f"{float(value):.{precision}f}"
    if isinstance(value, str):
        return value
    raise TypeError(f"Unsupported value type {type(value).__name__}")


# EPICS alarm severity -> (name, value style).
_ALARM_STYLES = {
    1: ("MINOR", "color: #e0a800; font-weight: bold;"),
    2: ("MAJOR", "color: #c62828; font-weight: bold;"),
    3: ("INVALID", "color: #8e24aa; font-weight: bold;"),
}


class _Forwarder:
    """Pass values to ``emit`` until cancelled, then drop them.

    It holds no reference to the widget, so the widget's ``destroyed`` signal
    can still cancel it once the widget is gone.
    """

    def __init__(self, emit: Callable[[Any], None]):
        self._emit: Callable[[Any], None] | None = emit
        self.on_cancel: Callable[[], None] | None = None

    def __call__(self, value: Any):
        emit = self._emit
        if emit is None:
            return
        try:
            emit(value)
        except RuntimeError:
            # Deleted between the check and the emit, before cancel ran.
            self._emit = None

    def cancel(self, *_: Any):
        """Stop forwarding and run :attr:`on_cancel`, once."""
        if self._emit is None and self.on_cancel is None:
            return
        self._emit = None
        on_cancel, self.on_cancel = self.on_cancel, None
        if on_cancel is not None:
            on_cancel()


class QtSignalStatus(QWidget):
    """A ``label: value units`` row that tracks a signal or device.

    The value is colored by its alarm severity: amber for minor, red for major,
    and purple for invalid. Subclasses can show the value differently by
    overriding :meth:`_make_value_widget` and :meth:`_show_reading`.

    Parameters
    ----------
    re_client : RunEngineClient
        Where the readings of ``name`` come from.
    name : str
        The signal or device, as named in the client's device registry.
    label : str, optional
        Text shown before the value. Defaults to ``name``.
    precision : int, optional
        Decimal places shown for float values.

    Raises
    ------
    KeyError
        If ``name`` is not a subscribable device or signal.
    """

    # The client may call back from any thread; these queued signals hop to Qt's.
    _signal_reading = Signal(object)
    _signal_units = Signal(object)
    #: Emitted on the Qt thread with each new raw value.
    value_changed = Signal(object)
    #: Whether to look up and show the readback's units.
    show_units = True

    def __init__(
        self,
        re_client: RunEngineClient,
        name: str,
        label: str | None = None,
        precision: int = 3,
        parent=None,
    ):
        super().__init__(parent)
        self.name = name
        self._precision = precision
        self._units = ""
        self._text = "--"

        self._label = QLabel(f"{label if label is not None else name}:")
        self._value = self._make_value_widget()
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self._label)
        hbox.addWidget(self._value, stretch=1)
        self.setLayout(hbox)

        # Connected first: the initial reading can arrive as soon as we subscribe.
        self._signal_reading.connect(self._set_reading)
        self._signal_units.connect(self._set_units)
        self._forwarders = (
            _Forwarder(self._signal_reading.emit),
            _Forwarder(self._signal_units.emit),
        )
        readings, units = self._forwarders
        for forwarder in self._forwarders:
            # Parents can delete this widget without ever closing it.
            self.destroyed.connect(forwarder.cancel)
        token = re_client.subscribe_reading(name, readings)
        readings.on_cancel = lambda: re_client.unsubscribe_reading(token)
        if self.show_units:
            re_client.fetch_units(name).add_done_callback(
                lambda f: units(None if f.cancelled() else f.result())
            )

    @property
    def label_widget(self) -> QLabel:
        """The label shown before the value, e.g. for aligning several rows."""
        return self._label

    def _make_value_widget(self) -> QWidget:
        """Create the widget that shows the value; a selectable label by default."""
        value = QLabel("--")
        value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return value

    @Slot(object)
    def _set_reading(self, reading: Reading[Any]):
        self._show_reading(reading)
        self.value_changed.emit(reading["value"])

    def _show_reading(self, reading: Reading[Any]):
        """Display a new reading: formatted text, colored by alarm severity."""
        try:
            self._text = format_value(reading["value"], self._precision)
        except TypeError as err:
            self._text = f"<{err}>"
        name, style = _ALARM_STYLES.get(int(reading.get("alarm_severity", 0)), ("", ""))
        self._value.setStyleSheet(style)
        self._value.setToolTip(f"Alarm: {name}" if name else "")
        self._render()

    @Slot(object)
    def _set_units(self, units: str | None):
        self._units = units or ""
        self._render()

    def _render(self):
        if not isinstance(self._value, QLabel):
            return
        text = f"{self._text} {self._units}" if self._units else self._text
        self._value.setText(text)

    def closeEvent(self, event):
        for forwarder in self._forwarders:
            forwarder.cancel()
        super().closeEvent(event)
