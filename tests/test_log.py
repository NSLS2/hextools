import io
import logging

import pytest
from bluesky import RunEngine
from bluesky import plan_stubs as bps
from bluesky.utils import FailedStatus, Msg, RequestAbort, RequestStop, RunEngineInterrupted

from hextools.log import (
    LEVEL_COLORS,
    LOGGER_NAME,
    ColorFormatter,
    configure_logger,
    log_pauses,
    log_plan_exceptions,
)


@pytest.fixture
def logger_name():
    name = "hextools.test_log"
    yield name
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def _record(level: int, msg: str = "hello") -> logging.LogRecord:
    return logging.LogRecord("x", level, __file__, 1, msg, None, None)


@pytest.mark.parametrize("level", sorted(LEVEL_COLORS))
def test_color_formatter_colors_level_name(level):
    text = ColorFormatter(fmt="%(levelname)s %(message)s").format(_record(level))
    assert text == f"{LEVEL_COLORS[level]}{logging.getLevelName(level)}\033[0m hello"


def test_color_formatter_can_disable_color():
    text = ColorFormatter(fmt="%(levelname)s %(message)s", color=False).format(_record(logging.ERROR))
    assert text == "ERROR hello"


def test_color_formatter_does_not_mutate_record():
    record = _record(logging.WARNING)
    ColorFormatter().format(record)
    assert record.levelname == "WARNING"


def test_color_formatter_leaves_custom_levels_uncolored():
    text = ColorFormatter(fmt="%(levelname)s").format(_record(25))
    assert "\033[" not in text


def test_default_logger_name():
    assert LOGGER_NAME == "hextools"


def test_configure_logger_logs_at_level(logger_name):
    stream = io.StringIO()
    logger = configure_logger(logger_name, level=logging.INFO, stream=stream)
    assert logger is logging.getLogger(logger_name)
    logger.debug("hidden")
    logger.info("shown")
    logger.getChild("child").warning("from child")
    output = stream.getvalue()
    assert "hidden" not in output
    assert "INFO hextools.test_log: shown" in output
    assert "WARNING hextools.test_log.child: from child" in output
    assert not logger.propagate


def test_configure_logger_colors_only_terminals_by_default(logger_name):
    plain = io.StringIO()
    configure_logger(logger_name, stream=plain).error("plain")
    assert "\033[" not in plain.getvalue()

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    tty = Terminal()
    configure_logger(logger_name, stream=tty).error("colored")
    assert LEVEL_COLORS[logging.ERROR] in tty.getvalue()


def test_configure_logger_color_can_be_forced(logger_name):
    stream = io.StringIO()
    configure_logger(logger_name, stream=stream, color=True).info("x")
    assert LEVEL_COLORS[logging.INFO] in stream.getvalue()


def test_configure_logger_replaces_its_own_handler_only(logger_name):
    logger = logging.getLogger(logger_name)
    other = logging.NullHandler()
    logger.addHandler(other)
    first, second = io.StringIO(), io.StringIO()
    configure_logger(logger_name, stream=first)
    configure_logger(logger_name, level="DEBUG", stream=second)
    logger.debug("once")
    assert first.getvalue() == ""
    assert second.getvalue().count("once") == 1
    assert other in logger.handlers
    assert len(logger.handlers) == 2


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def plan_log(RE: RunEngine, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    handler = _ListHandler()
    logger.addHandler(handler)
    log_plan_exceptions(RE, logger_name)
    return handler.records


def _levels_and_messages(records):
    return [(r.levelno, r.getMessage()) for r in records]


def test_log_plan_exceptions_logs_plan_errors_and_reraises(RE: RunEngine, plan_log):
    def plan():
        yield from bps.null()
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        RE(plan())
    assert _levels_and_messages(plan_log) == [(logging.ERROR, "Plan failed with ValueError: boom")]


def test_log_plan_exceptions_logs_errors_from_failed_messages(RE: RunEngine, plan_log):
    def plan():
        yield Msg("not_a_real_command")

    with pytest.raises(Exception) as excinfo:
        RE(plan())
    assert len(plan_log) == 1
    assert plan_log[0].levelno == logging.ERROR
    assert plan_log[0].getMessage().startswith(f"Plan failed with {type(excinfo.value).__name__}")


def test_log_plan_exceptions_logs_the_cause_of_failed_status(RE: RunEngine, plan_log):
    def plan():
        yield from bps.null()
        try:
            raise ValueError("motor stalled")
        except ValueError as exc:
            raise FailedStatus("status") from exc

    with pytest.raises(FailedStatus):
        RE(plan())
    assert _levels_and_messages(plan_log) == [(logging.ERROR, "Plan failed with ValueError: motor stalled")]


def test_log_plan_exceptions_logs_the_cause_of_a_failed_move(RE: RunEngine, plan_log):
    status_module = pytest.importorskip("ophyd.status")

    class Stalls:
        name = "stalls"
        parent = None

        def set(self, value):
            status = status_module.Status()
            status.set_exception(RuntimeError("motor stalled"))
            return status

    with pytest.raises(FailedStatus) as excinfo:
        RE(bps.mv(Stalls(), 1))
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert _levels_and_messages(plan_log) == [(logging.ERROR, "Plan failed with RuntimeError: motor stalled")]


def test_log_plan_exceptions_logs_failed_status_without_cause(RE: RunEngine, plan_log):
    def plan():
        yield from bps.null()
        raise FailedStatus("status")

    with pytest.raises(FailedStatus):
        RE(plan())
    assert _levels_and_messages(plan_log) == [(logging.ERROR, "Plan failed with FailedStatus: status")]


def test_log_plan_exceptions_ignores_successful_plans(RE: RunEngine, plan_log):
    RE(bps.null())
    assert plan_log == []


def test_log_plan_exceptions_ignores_stops(RE: RunEngine, plan_log):
    def plan():
        yield from bps.null()
        raise RequestStop

    RE(plan())
    assert plan_log == []


def test_log_plan_exceptions_logs_aborts_as_warnings(RE: RunEngine, plan_log):
    def plan():
        yield from bps.null()
        raise RequestAbort

    try:
        RE(plan())
    except Exception:
        pass
    assert _levels_and_messages(plan_log) == [(logging.WARNING, "Plan aborted (RequestAbort).")]


def test_log_plan_exceptions_accepts_a_logger(RE: RunEngine, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    handler = _ListHandler()
    logger.addHandler(handler)
    log_plan_exceptions(RE, logger)

    def plan():
        yield from bps.null()
        raise RuntimeError("bad")

    with pytest.raises(RuntimeError):
        RE(plan())
    assert _levels_and_messages(handler.records) == [(logging.ERROR, "Plan failed with RuntimeError: bad")]


def test_log_pauses_warns_on_pause_and_chains_existing_hook(RE: RunEngine, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    handler = _ListHandler()
    logger.addHandler(handler)
    states: list[str] = []
    RE.state_hook = lambda new, old: states.append(str(new))
    log_pauses(RE, logger_name)

    def plan():
        yield from bps.checkpoint()
        yield from bps.pause()
        yield from bps.null()

    with pytest.raises(RunEngineInterrupted):
        RE(plan())
    assert [(r.levelno, r.getMessage().split(".")[0]) for r in handler.records] == [
        (logging.WARNING, "Plan paused")
    ]
    RE.resume()
    assert len(handler.records) == 1
    assert "paused" in states and states[-1] == "idle"
