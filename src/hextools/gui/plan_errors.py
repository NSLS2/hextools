"""Popups reporting plans that didn't complete."""

from __future__ import annotations

from qtpy.QtWidgets import QMessageBox, QWidget

from hextools.gui.run_engine_clients import PlanPausedError


def show_plan_error(parent: QWidget, title: str, error: BaseException) -> None:
    """Show ``error`` in a popup: a warning if the plan paused, else an error."""
    if isinstance(error, PlanPausedError):
        QMessageBox.warning(parent, "Plan paused", str(error))
    else:
        QMessageBox.critical(parent, title, str(error))
