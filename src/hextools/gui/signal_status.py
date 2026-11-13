"""Label showing the live value of a signal or device, looked up by name."""

from __future__ import annotations

from typing import Any

import numpy as np
from bluesky.protocols import Reading
from qtpy.QtCore import Qt, Signal, Slot
from qtpy.QtWidgets import QHBoxLayout, QLabel, QWidget

from hextools.gui.reading_sources import ReadingSource


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


class QtSignalStatus(QWidget):
    """A ``label: value units`` row that tracks a signal or device.

    The value is colored by its alarm severity: amber for minor, red for major,
    and purple for invalid. Subclasses can show the value differently by
    overriding :meth:`_make_value_widget` and :meth:`_show_reading`.

    Parameters
    ----------
    name : str
        The signal or device, as named in ``source``.
    source : ReadingSource
        Where ``name`` is looked up and its readings come from.
    label : str, optional
        Text shown before the value. Defaults to ``name``.
    precision : int, optional
        Decimal places shown for float values.
    """

    _signal_units = Signal(str)
    # Sources may call back from any thread; this queued signal hops to Qt's.
    _signal_reading = Signal(object)
    #: Emitted on the Qt thread with each new raw value.
    value_changed = Signal(object)
    #: Whether to look up and show the readback's units.
    show_units = True

    def __init__(
        self,
        name: str,
        source: ReadingSource,
        label: str | None = None,
        precision: int = 3,
        parent=None,
    ):
        super().__init__(parent)
        self._name = name
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

        self._signal_units.connect(self._set_units)
        # Connected first: the initial reading can arrive as soon as we subscribe.
        self._signal_reading.connect(self._set_reading)
        self._source = source
        self._token = source.subscribe(name, self._signal_reading.emit)
        self._subscribed = True
        if self.show_units:
            source.fetch_units(name, self._signal_units.emit)

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

    @Slot(str)
    def _set_units(self, units: str):
        self._units = units
        self._render()

    def _render(self):
        if not isinstance(self._value, QLabel):
            return
        text = f"{self._text} {self._units}" if self._units else self._text
        self._value.setText(text)

    def closeEvent(self, event):
        # Parents and Qt teardown can each close this widget.
        if self._subscribed:
            self._subscribed = False
            self._source.unsubscribe(self._token)
        super().closeEvent(event)
