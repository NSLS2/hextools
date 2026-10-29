"""The GUI's expected devices name the classes the profile builds."""

import ast
from pathlib import Path

import hextools

SRC = Path(hextools.__file__).parent


def _profile_constructors() -> dict[str, str]:
    tree = ast.parse((SRC / "profiles" / "collection.py").read_text())
    built = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    built[target.id] = node.value.func.id
    return built


def _expected_devices() -> list[tuple[str, str]]:
    # HEX device expectations live with its spec, not the generic launcher.
    tree = ast.parse((SRC / "gui_config.py").read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "HEX_GUI"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
        ):
            expected_devices = next(
                (
                    keyword.value
                    for keyword in node.value.keywords
                    if keyword.arg == "expected_devices"
                ),
                None,
            )
            if not isinstance(expected_devices, ast.List):
                break
            return [
                (ast.literal_eval(entry.elts[0]), ast.unparse(entry.elts[1]))
                for entry in expected_devices.elts
                if isinstance(entry, ast.Tuple)
            ]
    raise AssertionError("GuiSpec.expected_devices not found in gui_config.py")


# Queue Server mode matches class names exactly, so fe_shutter built as a
# PPSShutter but expected as a Shutter showed as Disconnected (#120 review).
def test_expected_devices_name_the_class_the_profile_builds():
    built = _profile_constructors()
    expected = _expected_devices()
    assert expected
    for name, class_name in expected:
        constructor = built.get(name)
        assert constructor is not None, f"{name} is not built in the profile"
        if constructor[0].isupper():  # a class, not a factory such as kinetix_factory
            assert constructor == class_name, name
