"""The GUI package is generic: beamline specifics come from a GuiSpec."""

import ast
from pathlib import Path

import pytest

import hextools.gui
from hextools.gui.spec import GuiSpec, load_spec

GUI_DIR = Path(hextools.gui.__file__).parent
# The launcher names the beamline's spec; everything else must stay generic.
LAUNCHER = GUI_DIR / "__main__.py"


def _hextools_imports(path: Path) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
        elif isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
    return [n for n in names if n == "hextools" or n.startswith("hextools.")]


@pytest.mark.parametrize(
    "path",
    [p for p in sorted(GUI_DIR.rglob("*.py")) if p != LAUNCHER],
    ids=lambda p: str(p.relative_to(GUI_DIR)),
)
def test_gui_module_imports_nothing_from_hextools_outside_the_gui(path: Path):
    outside = [
        name for name in _hextools_imports(path) if not name.startswith("hextools.gui")
    ]
    assert outside == []


def test_hex_spec_loads_and_names_callable_plans():
    spec = load_spec("hextools.gui_config:HEX_GUI")
    assert isinstance(spec, GuiSpec)
    assert spec.title == "HEX Data Acquisition"
    assert [t.title for t in spec.techniques] == [
        "Beamline",
        "Tomography",
        "EDXD",
        "XRD",
    ]
    for technique in spec.techniques:
        assert all(callable(plan) for _, plan in technique.plans)


def test_load_spec_refuses_something_that_is_not_a_spec():
    with pytest.raises(TypeError, match="not a GuiSpec"):
        load_spec("hextools.gui.spec:Site")
