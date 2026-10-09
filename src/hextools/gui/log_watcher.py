"""Forward records from a Python logger to Qt widgets."""

from __future__ import annotations

import logging

from qtpy.QtCore import QObject, Signal
from qtpy.QtGui import QColor

from hextools.log import LOGGER_NAME

#: Text color for each log level in the GUI.
LEVEL_QCOLORS: dict[int, QColor] = {
    logging.DEBUG: QColor("#3a9bd9"),
    logging.INFO: QColor("#27ae60"),
    logging.WARNING: QColor("#e0a800"),
    logging.ERROR: QColor("#e74c3c"),
    logging.CRITICAL: QColor("#c0392b"),
}


def level_color(levelno: int) -> QColor:
    """Color for ``levelno``, using the nearest standard level at or below it."""
    for level in sorted(LEVEL_QCOLORS, reverse=True):
        if levelno >= level:
            return LEVEL_QCOLORS[level]
    return LEVEL_QCOLORS[logging.DEBUG]


class _SignalHandler(logging.Handler):
    def __init__(self, watcher: LogWatcher, level: int):
        super().__init__(level)
        self._watcher = watcher

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._watcher.record.emit(record.created, record.levelno, record.levelname, record.getMessage())
        except RuntimeError:
            pass  # The watcher's Qt object was deleted.


class LogWatcher(QObject):
    """Emits ``record(created, levelno, levelname, message)`` for each record logged to ``logger_name``.

    Records may be logged from any thread; Qt delivers the signal on the watcher's thread.
    """

    record = Signal(float, int, str, str)

    def __init__(self, logger_name: str = LOGGER_NAME, level: int = logging.NOTSET, parent=None):
        super().__init__(parent)
        logger = logging.getLogger(logger_name)
        handler = _SignalHandler(self, level)
        logger.addHandler(handler)
        self.destroyed.connect(lambda *_: logger.removeHandler(handler))
