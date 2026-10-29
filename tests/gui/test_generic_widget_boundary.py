"""Blocking boundary for the generic widgets changed in this follow-up."""

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", ["plan_widget", "shutter_status", "device_sources"])
def test_generic_widgets_do_not_import_hex_policy(module):
    source = (
        Path(__file__).parents[2] / "src" / "hextools" / "gui" / f"{module}.py"
    ).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            name = node.module or ""
            assert name != "hextools.gui.hex_config"
            assert not name.startswith("hextools.") or name.startswith(
                ("hextools.gui.", "hextools.log")
            ), f"{module} imports beamline policy from {name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "hextools.gui.hex_config"
                assert not alias.name.startswith("hextools.") or alias.name.startswith(
                    ("hextools.gui.", "hextools.log")
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
