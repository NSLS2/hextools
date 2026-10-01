"""Run code in the attached IPython shell on behalf of the GUI."""

from __future__ import annotations

import sys


def run_in_ipython(code: str) -> BaseException | None:
    """Echo ``code`` to the terminal, run it in IPython, and return any error."""
    from IPython.core.getipython import get_ipython

    ipython = get_ipython()
    if ipython is None:
        return RuntimeError("No IPython shell available for in-process execution.")
    # prompt_toolkit's patch_stdout proxy buffers output until the plan finishes;
    # write to the real terminal instead so progress renders live.
    patched_stdout = sys.stdout
    real_stdout = sys.__stdout__
    if real_stdout is not None:
        sys.stdout = real_stdout
    try:
        if real_stdout is not None and real_stdout.isatty():
            # Clear the idle prompt line, then show a dim tag and bold command.
            print(f"\r\033[K\033[2m[GUI]\033[0m \033[1m{code}\033[0m")
        else:
            print(f"[GUI] {code}")
        # Trailing semicolon suppresses the Out[n] display of the RE return value.
        result = ipython.run_cell(f"{code};", store_history=True)
    finally:
        sys.stdout = patched_stdout
    return result.error_in_exec or result.error_before_exec
