"""Blocking boundary for the generic widgets changed in this follow-up."""

import ast
from pathlib import Path

import pytest

# Packages with no HEX beamline policy, which generic widgets may use.
_GENERIC = ("hextools.gui.", "hextools.log", "hextools.catalog")


@pytest.mark.parametrize(
    "module", ["plan_widget", "shutter_status", "run_engine_clients"]
)
def test_generic_widgets_do_not_import_hex_policy(module):
    source = (
        Path(__file__).parents[2] / "src" / "hextools" / "gui" / f"{module}.py"
    ).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            name = node.module or ""
            assert name != "hextools.gui.hex_config"
            assert not name.startswith("hextools.") or name.startswith(_GENERIC), (
                f"{module} imports beamline policy from {name}"
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "hextools.gui.hex_config"
                assert not alias.name.startswith("hextools.") or alias.name.startswith(
                    _GENERIC
                )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not any(
                identity in node.value
                for identity in (
                    "move_slits",
                    "edxd_scan",
                    "edxd_2theta_tilt",
                    "Two theta",
                    "sleep_time=0",
                    "XF:27",
                )
            ), f"{module} embeds HEX-specific identity or policy"
