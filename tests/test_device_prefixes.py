"""HEX devices use the PV prefixes their IOCs serve.

Expected values are each IOC's PREFIX in NSLS2/ioc_host_vars
(xf/hex/<host>/<ioc>.yml), read at deefd59.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parents[1] / "src/hextools"

IOC_PREFIXES = {
    "phantom": "XF:27ID1-ES{Phantom-Det:1}",
    "diamond_window_cam": "XF:27IDA-BI{FAM:1-Cam:1}",
    "fs_window_cam": "XF:27IDA-BI{FS:1-Cam:1}",
    "sample_cam": "XF:27ID1-ES{Sample-Cam:1}",
    "f_hutch_cam": "XF:27ID1-ES{GigE-Cam:5}",
}

GUI_CAMERAS = ["fs_window_cam", "diamond_window_cam", "sample_cam", "f_hutch_cam"]


def _string_first_args(path: Path) -> dict[str, str]:
    """``name = Device("prefix", ...)`` assignments, as ``{name: prefix}``."""
    found = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and node.value.args
            and isinstance(node.value.args[0], ast.Constant)
            and isinstance(node.value.args[0].value, str)
        ):
            found[node.targets[0].id] = node.value.args[0].value
    return found


def _gui_camera_prefixes() -> dict[str, str]:
    """The ``(name, prefix)`` pairs the GUI builds its camera viewers from."""
    found = {}
    for node in ast.walk(ast.parse((SRC / "gui/__main__.py").read_text())):
        if (
            isinstance(node, ast.Tuple)
            and len(node.elts) == 2
            and all(
                isinstance(e, ast.Constant) and isinstance(e.value, str)
                for e in node.elts
            )
        ):
            name, prefix = (e.value for e in node.elts)  # type: ignore[attr-defined]
            if name in GUI_CAMERAS:
                found[name] = prefix
    return found


@pytest.mark.parametrize("name", IOC_PREFIXES)
def test_profile_device_uses_its_ioc_prefix(name: str):
    profile = _string_first_args(SRC / "profiles/collection.py")
    assert profile[name] == IOC_PREFIXES[name]


@pytest.mark.parametrize("name", GUI_CAMERAS)
def test_gui_camera_uses_its_ioc_prefix(name: str):
    assert _gui_camera_prefixes()[name] == IOC_PREFIXES[name]
