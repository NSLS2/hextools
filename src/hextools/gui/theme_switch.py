"""A dark-mode toggle that stays in step with the application-wide theme."""

from __future__ import annotations

from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication

try:  # QAction moved from QtWidgets to QtGui in Qt6
    from qtpy.QtGui import QAction
except ImportError:
    from qtpy.QtWidgets import QAction

from hextools.gui._theme import apply_bnl_theme, current_theme, save_theme


class _ThemeNotifier(QObject):
    changed = Signal(str)


def _notifier() -> _ThemeNotifier:
    # One per QApplication, so every window's toggle hears every change.
    app = QApplication.instance()
    notifier = app.findChild(_ThemeNotifier) if app is not None else None
    if notifier is None:
        notifier = _ThemeNotifier(app)
    return notifier


def set_theme(theme: str) -> None:
    """Apply ``theme`` app-wide, remember it, and update every toggle."""
    apply_bnl_theme(theme=theme)
    save_theme(theme)
    _notifier().changed.emit(theme)


class QtThemeAction(QAction):
    """Checkable "Dark Mode" menu action that sets the whole app's theme."""

    def __init__(self, parent=None):
        super().__init__("Dark Mode", parent)
        self.setCheckable(True)
        self.setToolTip("Switch between dark and light backgrounds")
        self.setChecked(current_theme() == "dark")
        self.toggled.connect(self._on_toggled)
        _notifier().changed.connect(self._follow)

    def _on_toggled(self, checked: bool):
        set_theme("dark" if checked else "light")

    def _follow(self, theme: str):
        self.blockSignals(True)
        self.setChecked(theme == "dark")
        self.blockSignals(False)
