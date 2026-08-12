import pytest

from hextools.utils import is_running_in_ci


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
