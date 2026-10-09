"""Run code in the attached IPython shell on behalf of the GUI."""

from __future__ import annotations

import sys
from typing import Any

from bluesky.utils import RunEngineInterrupted
from qtpy.QtWidgets import QMessageBox, QWidget

from hextools.log import unwrap_failed_status

PAUSED_MESSAGE = (
    "The plan is paused. Press Resume to continue it, or Stop, Abort, or Halt to end it."
)


def show_plan_error(parent: QWidget, title: str, error: BaseException) -> None:
    """Report an error from :func:`run_in_ipython`, as a warning if the plan just paused."""
    if isinstance(error, RunEngineInterrupted):
        QMessageBox.warning(parent, "Plan paused", PAUSED_MESSAGE)
    else:
        cause = unwrap_failed_status(error)
        QMessageBox.critical(parent, title, f"{type(cause).__name__}: {cause}")


def run_in_ipython(code: str) -> BaseException | None:
    """Echo ``code`` to the terminal, run it in IPython, and return any error."""
    from IPython.core.getipython import get_ipython

    ipython = get_ipython()
    if ipython is None:
        return RuntimeError("No IPython shell available for in-process execution.")
    # prompt_toolkit's patch_stdout proxy buffers output until the plan finishes;
    # write to the real terminal instead so progress renders live.
    patched_stdout = sys.stdout
    real_stdout = sys.__stdout__
    if real_stdout is not None:
        sys.stdout = real_stdout
    try:
        if real_stdout is not None and real_stdout.isatty():
            # Clear the idle prompt line, then show a dim tag and bold command.
            print(f"\r\033[K\033[2m[GUI]\033[0m \033[1m{code}\033[0m")
        else:
            print(f"[GUI] {code}")
        # Trailing semicolon suppresses the Out[n] display of the RE return value.
        result = ipython.run_cell(f"{code};", store_history=True)
    finally:
        sys.stdout = patched_stdout
    _add_to_prompt_history(ipython, code)
    return result.error_in_exec or result.error_before_exec


def _add_to_prompt_history(ipython: Any, code: str) -> None:
    """Make ``code`` reachable with the up arrow at the IPython prompt that is waiting now."""
    pt_app = getattr(ipython, "pt_app", None)
    if pt_app is None:
        return
    pt_app.history.append_string(code)
    buffer = pt_app.default_buffer
    # The waiting prompt only reloads its history after a reset; keep whatever is typed.
    buffer.reset(document=buffer.document)
    pt_app.app.invalidate()
