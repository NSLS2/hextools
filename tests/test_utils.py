from types import SimpleNamespace
from typing import Any

import pytest

from hextools.utils import (
    ensure_available,
    get_obj_from_ipython_ns,
    initialize_run_engine,
    is_running_in_ci,
    merge_async_iterables,
    show_docs,
    start_beamtime,
)


@pytest.mark.parametrize(
    "env_value, expected",
    [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("", False),
        (None, False),
    ],
)
def test_is_running_in_ci(monkeypatch, env_value, expected):
    if env_value is None:
        monkeypatch.delenv("HEXTOOLS_RUNNING_IN_CI", raising=False)
    else:
        monkeypatch.setenv("HEXTOOLS_RUNNING_IN_CI", env_value)
    assert is_running_in_ci() == expected


async def test_merge_async_iterables_yields_all_items():
    async def _agen(items):
        for item in items:
            yield item

    merged = [
        item async for item in merge_async_iterables(_agen([1, 2, 3]), _agen([4, 5]))
    ]

    # Order across sources is not guaranteed, so compare as sets.
    assert sorted(merged) == [1, 2, 3, 4, 5]


def test_show_docs(capsys):
    show_docs("start", {"scan_id": 7})
    out = capsys.readouterr().out
    assert "start" in out
    assert "scan_id" in out


def test_initialize_run_engine_ci_metadata(monkeypatch):
    monkeypatch.setenv("HEXTOOLS_RUNNING_IN_CI", "yes")
    captured: dict[str, Any] = {}

    def fake_run_engine(md):
        captured["md"] = md
        return md

    monkeypatch.setattr("hextools.utils.RunEngine", fake_run_engine)

    initialize_run_engine()

    md = captured["md"]
    assert md["data_session"] == "pass-123456"
    assert "cycle" in md
    assert md["proposal"]["type"] == "Mock Commissioning"


def test_start_beamtime(monkeypatch, capsys):
    md = {"proposal": {"title": "My Study", "type": "PU", "pi_name": "Ada"}}
    monkeypatch.setattr("hextools.utils.sync_experiment", lambda *a, **k: md)

    start_beamtime(123456)

    out = capsys.readouterr().out
    assert "123456" in out
    assert "My Study" in out


def test_get_obj_from_ipython_ns(monkeypatch):
    fake_ns = {"my_var": 42}
    monkeypatch.setattr(
        "hextools.utils.get_ipython", lambda: SimpleNamespace(user_ns=fake_ns)
    )

    result = get_obj_from_ipython_ns("my_var", int)
    assert result == 42

    result_none = get_obj_from_ipython_ns("non_existent_var", int)
    assert result_none is None


@pytest.fixture
def mock_namespace(monkeypatch):
    fake_ns = {
        "my_var": 42,
        "other_var": 99.5,
        "another_var": "hello",
        "a_fourth_var": "world",
    }
    monkeypatch.setattr(
        "hextools.utils.get_ipython", lambda: SimpleNamespace(user_ns=fake_ns)
    )
    return fake_ns


@pytest.mark.parametrize("kwargs", [{}, {"my_var": 42, "other_var": 99.5}])
def test_ensure_available_only_takes_one_var(mock_namespace, kwargs):
    with pytest.raises(
        ValueError, match="Can only check availability of a single object at a time."
    ):
        ensure_available(int, **kwargs)


@pytest.mark.parametrize(
    "name, value, type, expected",
    [
        ("my_var", None, int, 42),
        ("other_var", None, float, 99.5),
        ("another_var", None, str, "hello"),
        ("a_fourth_var", None, str, "world"),
        ("provided_var", 123.4, float, 123.4),
    ],
)
def test_ensure_available_success(mock_namespace, name, value, type, expected):
    result = ensure_available(type, **{name: value})
    assert result == expected


def test_ensure_available_fails_if_val_invalid_type(mock_namespace):
    with pytest.raises(
        TypeError,
        match="Value for my_var must be of type <class 'int'> or None",
    ):
        ensure_available(int, my_var="not an int")


def test_ensure_available_fails_if_not_provided_and_not_in_ns(mock_namespace):
    with pytest.raises(
        ValueError,
        match="Object non_existent_var of type <class 'int'>",
    ):
        ensure_available(int, non_existent_var=None)
