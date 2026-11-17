import io
import json
import logging
from pathlib import Path

import pytest
from bluesky import RunEngine
from bluesky import plan_stubs as bps
from bluesky import plans as bp
from bluesky.run_engine import RunEngineResult
from bluesky.utils import (
    FailedStatus,
    Msg,
    RequestAbort,
    RequestStop,
    RunEngineInterrupted,
)
from ophyd_async.sim import SimMotor

from hextools.log import (
    _LEVEL_COLORS,
    _LOGGER_NAME,
    _ColorFormatter,
    configure_json_logging,
    configure_logger,
    log_plan_exceptions,
    log_state_changes,
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


@pytest.mark.parametrize("level", sorted(_LEVEL_COLORS))
def test_color_formatter_colors_level_name(level):
    text = _ColorFormatter(fmt="%(levelname)s %(message)s").format(_record(level))
    assert text == f"{_LEVEL_COLORS[level]}{logging.getLevelName(level)}\033[0m hello"


def test_color_formatter_can_disable_color():
    text = _ColorFormatter(fmt="%(levelname)s %(message)s", color=False).format(
        _record(logging.ERROR)
    )
    assert text == "ERROR hello"


def test_color_formatter_does_not_mutate_record():
    record = _record(logging.WARNING)
    _ColorFormatter().format(record)
    assert record.levelname == "WARNING"


def test_color_formatter_leaves_custom_levels_uncolored():
    text = _ColorFormatter(fmt="%(levelname)s").format(_record(25))
    assert "\033[" not in text


def test_default_logger_name():
    assert _LOGGER_NAME == "hextools"


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
    assert _LEVEL_COLORS[logging.ERROR] in tty.getvalue()


def test_configure_logger_color_can_be_forced(logger_name):
    stream = io.StringIO()
    configure_logger(logger_name, stream=stream, color=True).info("x")
    assert _LEVEL_COLORS[logging.INFO] in stream.getvalue()


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
    assert _levels_and_messages(plan_log) == [
        (logging.ERROR, "Plan failed with ValueError: boom")
    ]


def test_log_plan_exceptions_logs_errors_from_failed_messages(RE: RunEngine, plan_log):
    def plan():
        yield Msg("not_a_real_command")

    with pytest.raises(Exception) as excinfo:
        RE(plan())
    assert len(plan_log) == 1
    assert plan_log[0].levelno == logging.ERROR
    assert (
        plan_log[0]
        .getMessage()
        .startswith(f"Plan failed with {type(excinfo.value).__name__}")
    )


def test_log_plan_exceptions_logs_the_cause_of_failed_status(RE: RunEngine, plan_log):
    def plan():
        yield from bps.null()
        try:
            raise ValueError("motor stalled")
        except ValueError as exc:
            raise FailedStatus("status") from exc

    with pytest.raises(FailedStatus):
        RE(plan())
    assert _levels_and_messages(plan_log) == [
        (logging.ERROR, "Plan failed with ValueError: motor stalled")
    ]


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
    assert _levels_and_messages(plan_log) == [
        (logging.ERROR, "Plan failed with RuntimeError: motor stalled")
    ]


def test_log_plan_exceptions_logs_failed_status_without_cause(RE: RunEngine, plan_log):
    def plan():
        yield from bps.null()
        raise FailedStatus("status")

    with pytest.raises(FailedStatus):
        RE(plan())
    assert _levels_and_messages(plan_log) == [
        (logging.ERROR, "Plan failed with FailedStatus: status")
    ]


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
    assert _levels_and_messages(plan_log) == [
        (logging.WARNING, "Plan aborted (RequestAbort).")
    ]


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
    assert _levels_and_messages(handler.records) == [
        (logging.ERROR, "Plan failed with RuntimeError: bad")
    ]


def test_log_state_changes_logs_transitions_and_chains_existing_hook(
    RE: RunEngine, logger_name
):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    handler = _ListHandler()
    logger.addHandler(handler)
    states: list[str] = []
    RE.state_hook = lambda new, old: states.append(str(new))  # type: ignore
    log_state_changes(RE, logger_name)

    def plan():
        yield from bps.checkpoint()
        yield from bps.pause()
        yield from bps.null()

    with pytest.raises(RunEngineInterrupted):
        RE(plan())
    RE.resume()
    messages = [(r.levelno, r.getMessage().split(".")[0]) for r in handler.records]
    assert messages[0] == (logging.INFO, "Run engine transitioned from idle to running")
    assert (
        logging.WARNING,
        "Run engine transitioned from pausing to paused",
    ) in messages
    assert messages[-1] == (
        logging.INFO,
        "Run engine transitioned from running to idle",
    )
    assert len(messages) == len(states)
    assert "paused" in states and states[-1] == "idle"


_JSON_LOGGERS = ("bluesky", "ophyd", "ophyd_async", "nslsii", "nslsii.documents")


@pytest.fixture
def json_log_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the JSON logs at ``tmp_path``, then close and remove their handlers."""
    monkeypatch.setenv("BLUESKY_JSON_LOG_DIR", str(tmp_path))
    names = (*_JSON_LOGGERS, _LOGGER_NAME)
    saved = {
        name: (logging.getLogger(name).level, logging.getLogger(name).propagate)
        for name in names
    }
    yield tmp_path
    for name in names:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            if str(getattr(handler, "baseFilename", "")).startswith(str(tmp_path)):
                logger.removeHandler(handler)
                handler.close()
        logger.setLevel(saved[name][0])
        logger.propagate = saved[name][1]
    configure_logger()


def _json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_configure_json_logging_writes_hextools_and_bluesky_records(
    RE: RunEngine, json_log_dir: Path
):
    logger = configure_logger(stream=io.StringIO())
    paths = configure_json_logging(RE)
    log_file, documents_file = paths
    assert log_file.parent == json_log_dir and log_file.name.endswith(".log.jsonl")
    assert documents_file.name.endswith(".documents.jsonl")

    def plan():
        yield from bps.open_run()
        logger.info("narrating inside the run")
        logger.debug("too detailed for the file")
        yield from bps.close_run()

    runs = RE(plan())
    assert isinstance(runs, RunEngineResult)
    uid = runs.run_start_uids[0]
    logging.getLogger("bluesky").warning("a bluesky warning")

    records = _json_lines(log_file)
    hextools_records = [r for r in records if r["logger"] == _LOGGER_NAME]
    assert [r["message"] for r in hextools_records] == ["narrating inside the run"]
    assert hextools_records[0]["level"] == "INFO"
    assert uid in hextools_records[0]["run_uids"]
    assert any(
        r["logger"] == "bluesky" and r["message"] == "a bluesky warning"
        for r in records
    )


def test_configure_json_logging_summarizes_every_document(
    RE: RunEngine, json_log_dir: Path
):
    _, documents_file = configure_json_logging(RE)
    runs = RE(bp.count([SimMotor(name="m")], num=2))
    assert isinstance(runs, RunEngineResult)
    uid = runs.run_start_uids[0]

    documents = _json_lines(documents_file)
    names = [d["document"] for d in documents]
    assert names[0] == "start" and names[-1] == "stop"
    assert "descriptor" in names
    assert names.count("event") + names.count("event_page") >= 1
    assert documents[0]["uid"] == uid
    assert documents[-1]["run_start"] == uid
    assert documents[-1]["exit_status"] == "success"


def test_configure_json_logging_keeps_the_console_handler(
    RE: RunEngine, json_log_dir: Path
):
    stream = io.StringIO()
    logger = configure_logger(stream=stream)
    configure_json_logging(RE)
    logger.info("both places")
    assert "both places" in stream.getvalue()
    assert logger.propagate is False
