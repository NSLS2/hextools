"""Names the startup profile gives operators at the IPython prompt."""

import ast
from pathlib import Path

import pytest

PROFILE = Path(__file__).parents[1] / "src/hextools/profiles/collection.py"

# Imported for the prompt, not used in the file, so ruff --fix (F401) deletes them.
OPERATOR_NAMES = [
    "reset_scan_id",
    "show_docs",
    "bp",
    "bps",
    "bpp",
    "move_motor",
    "move_slits",
    "edxd_grid_scan",
    "GeRMStatsDataLogic",
    "ProgressBarManager",
    "call_in_bluesky_event_loop",
    "KinetixDetector",
    "Path",
]


def _profile_import_namespace() -> dict:
    """Run only the profile's import statements, without building any devices."""
    namespace: dict = {}
    for node in ast.parse(PROFILE.read_text()).body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            code = compile(ast.Module([node], type_ignores=[]), str(PROFILE), "exec")
            exec(code, namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize("name", OPERATOR_NAMES)
def test_profile_imports_operator_name(name: str):
    assert name in _profile_import_namespace()
