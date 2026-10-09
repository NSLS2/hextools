"""Logging setup for hextools, with color-coded log levels."""

from __future__ import annotations

import copy
import logging
import sys
from collections.abc import Generator
from typing import Any, TextIO

from bluesky import RunEngine
from bluesky.utils import FailedPause, FailedStatus, Msg, PlanHalt, RequestAbort, RequestStop

# Name of the logger hextools logs to; child loggers (``hextools.*``) propagate to it.
LOGGER_NAME = "hextools"

_RESET = "\033[0m"
# ANSI color for each log level.
LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: "\033[36m",  # cyan
    logging.INFO: "\033[32m",  # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",  # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}

DEFAULT_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


class ColorFormatter(logging.Formatter):
    """Formatter that colors each record's level name by its severity."""

    def __init__(
        self, fmt: str = DEFAULT_FORMAT, datefmt: str | None = "%H:%M:%S", color: bool = True
    ) -> None:
        super().__init__(fmt, datefmt)
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        color = LEVEL_COLORS.get(record.levelno) if self.color else None
        if color is None:
            return super().format(record)
        # Color a copy, so other handlers of the same record see the plain level name.
        record = copy.copy(record)
        record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


class _HextoolsHandler(logging.StreamHandler):
    """Marks the handler :func:`configure_logger` installed, so reconfiguring replaces it."""


def configure_logger(
    name: str = LOGGER_NAME,
    level: int | str = logging.INFO,
    stream: TextIO | None = None,
    color: bool | None = None,
) -> logging.Logger:
    """Configure logger ``name`` to print to ``stream`` with color-coded levels.

    Calling it again replaces the handler from the previous call rather than adding
    another one.

    Parameters
    ----------
    name : str, optional
        Logger to configure, by default ``"hextools"``.
    level : int or str, optional
        Minimum level to log, by default ``INFO``.
    stream : TextIO, optional
        Where to write records, by default ``sys.stderr``.
    color : bool, optional
        Whether to color the level names. By default, only if ``stream`` is a terminal.

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    stream = stream if stream is not None else sys.stderr
    if color is None:
        color = bool(getattr(stream, "isatty", lambda: False)())

    logger = logging.getLogger(name)
    logger.setLevel(level)
    for handler in [h for h in logger.handlers if isinstance(h, _HextoolsHandler)]:
        logger.removeHandler(handler)
    handler = _HextoolsHandler(stream)
    handler.setFormatter(ColorFormatter(color=color))
    logger.addHandler(handler)
    # Our handler prints the records; don't print them again through the root logger.
    logger.propagate = False
    return logger


def unwrap_failed_status(error: BaseException) -> BaseException:
    """Return the error behind a ``FailedStatus``, or ``error`` itself otherwise.

    The RunEngine raises ``FailedStatus(status) from <original error>`` when a status fails.
    """
    while isinstance(error, FailedStatus) and error.__cause__ is not None:
        error = error.__cause__
    return error


def log_plan_exceptions(re: RunEngine, logger: logging.Logger | str = LOGGER_NAME) -> None:
    """Log exceptions raised during plans run by ``re`` as errors on ``logger``.

    Installs a RunEngine preprocessor; the exception is still raised to the caller.
    For a ``FailedStatus``, the error that caused it is logged instead.
    Aborts and halts are logged as warnings, and stops (which end a plan successfully)
    are not logged.
    """
    log = logging.getLogger(logger) if isinstance(logger, str) else logger

    def preprocessor(plan: Generator[Msg, Any, Any]) -> Generator[Msg, Any, Any]:
        try:
            return (yield from plan)
        except RequestStop:
            raise
        except (RequestAbort, PlanHalt, FailedPause) as ex:
            log.warning("Plan aborted (%s).", type(ex).__name__)
            raise
        except Exception as ex:
            cause = unwrap_failed_status(ex)
            log.error("Plan failed with %s: %s", type(cause).__name__, cause)
            raise

    re.preprocessors.append(preprocessor)


def log_pauses(re: RunEngine, logger: logging.Logger | str = LOGGER_NAME) -> None:
    """Log a warning on ``logger`` whenever ``re`` pauses, chaining onto any existing ``state_hook``."""
    log = logging.getLogger(logger) if isinstance(logger, str) else logger
    previous_hook = re.state_hook

    def state_hook(new_state, old_state):
        if str(new_state) == "paused":
            log.warning("Plan paused. Resume, stop, abort, or halt the RunEngine to continue.")
        if previous_hook is not None:
            previous_hook(new_state, old_state)

    re.state_hook = state_hook  # type: ignore (TODO: type hint upstream)
