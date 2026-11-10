"""Label showing the live value of a subscribable signal or device."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from bluesky.protocols import Reading, Subscribable
from qtpy.QtCore import QObject, Qt, Signal, Slot
from qtpy.QtWidgets import QHBoxLayout, QLabel, QWidget


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


def reading_key(obj: Subscribable[Any], keys: Mapping[str, Any]) -> str | None:
    """Return the key of ``obj``'s own reading in ``keys``: its name, else the first."""
    if obj.name in keys:
        return obj.name
    return next(iter(keys), None)


class QtReadingSubscription(QObject):
    """Subscribe to ``obj`` and re-emit its readings on the Qt thread.

    Parameters
    ----------
    obj : Subscribable
        The signal or device. For a device, the reading named after the device
        is emitted, otherwise its first reading.
    loop : asyncio.AbstractEventLoop
        The bluesky event loop (e.g. ``RE.loop``) that ``obj`` lives on.
    on_reading : Callable[[Reading], None]
        Called on the Qt thread with each new reading.
    """

    reading_changed = Signal(object)

    def __init__(
        self,
        obj: Subscribable[Any],
        loop: asyncio.AbstractEventLoop,
        on_reading: Callable[[Reading[Any]], None],
        parent=None,
    ):
        super().__init__(parent)
        self._obj = obj
        self._loop = loop
        # Connected first: the initial reading can arrive as soon as we subscribe.
        self.reading_changed.connect(on_reading)
        # ophyd-async subscriptions must be made on the bluesky event loop, and
        # callbacks arrive on that thread; the queued Qt signal hops back.
        loop.call_soon_threadsafe(obj.subscribe_reading, self._on_reading)
        self._subscribed = True

    def _on_reading(self, readings: Mapping[str, Reading[Any]]):
        key = reading_key(self._obj, readings)
        if key is not None:
            self.reading_changed.emit(readings[key])

    def unsubscribe(self):
        """Remove the subscription; safe to call more than once."""
        if not self._subscribed:
            return
        self._subscribed = False

        def _clear():
            try:
                self._obj.clear_sub(self._on_reading)
            except KeyError:
                pass

        self._loop.call_soon_threadsafe(_clear)


class QtSignalStatus(QWidget):
    """A ``label: value units`` row that tracks a subscribable signal or device.

    The value is colored by its alarm severity: amber for minor, red for major,
    and purple for invalid. Subclasses can show the value differently by
    overriding :meth:`_make_value_widget` and :meth:`_show_reading`.

    Parameters
    ----------
    obj : Subscribable
        The signal or device. For a device, the reading named after the device
        is shown, otherwise its first reading.
    loop : asyncio.AbstractEventLoop
        The bluesky event loop (e.g. ``RE.loop``) that ``obj`` lives on.
    label : str, optional
        Text shown before the value. Defaults to ``obj.name``.
    precision : int, optional
        Decimal places shown for float values.
    """

    _signal_units = Signal(str)
    #: Emitted on the Qt thread with each new raw value.
    value_changed = Signal(object)
    #: Whether to look up and show the readback's units.
    show_units = True

    def __init__(
        self,
        obj: Subscribable[Any],
        loop: asyncio.AbstractEventLoop,
        label: str | None = None,
        precision: int = 3,
        parent=None,
    ):
        super().__init__(parent)
        self._obj = obj
        self._precision = precision
        self._units = ""
        self._text = "--"

        self._label = QLabel(f"{label if label is not None else obj.name}:")
        self._value = self._make_value_widget()
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self._label)
        hbox.addWidget(self._value, stretch=1)
        self.setLayout(hbox)

        self._signal_units.connect(self._set_units)
        self._subscription = QtReadingSubscription(obj, loop, self._set_reading, self)
        if self.show_units:
            asyncio.run_coroutine_threadsafe(self._fetch_units(), loop)

    @property
    def label_widget(self) -> QLabel:
        """The label shown before the value, e.g. for aligning several rows."""
        return self._label

    def _make_value_widget(self) -> QWidget:
        """Create the widget that shows the value; a selectable label by default."""
        value = QLabel("--")
        value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return value

    async def _fetch_units(self):
        describe = getattr(self._obj, "describe", None)
        if describe is None:
            return
        try:
            descriptions = describe()
            if inspect.isawaitable(descriptions):
                descriptions = await descriptions
        except Exception:
            return
        key = reading_key(self._obj, descriptions)
        units = descriptions[key].get("units") if key is not None else None
        if units:
            self._signal_units.emit(str(units))

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
        self._subscription.unsubscribe()
        super().closeEvent(event)
