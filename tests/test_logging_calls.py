import ast
from pathlib import Path

import hextools

_LEVELS = {"debug", "info", "warning", "error", "critical", "exception"}


# An f-string message plus an extra positional argument makes logging fail to format
# and drop the message; ruff's PLE1205 does not check f-string messages.
def test_no_logging_call_passes_extra_args_after_an_f_string():
    offenders = []
    for path in sorted(Path(hextools.__file__).parent.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _LEVELS
                and len(node.args) > 1
                and isinstance(node.args[0], ast.JoinedStr)
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []
