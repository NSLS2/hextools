"""%mov and %movr leave the RunEngine's waiting hook as they found it."""

import pytest
from bluesky import RunEngine
from IPython.core.interactiveshell import InteractiveShell
from ophyd.sim import motor  # type: ignore

from hextools.utils.magics import HEXBlueskyMagics


# Copilot on #109: they set RE.waiting_hook to None afterwards, so the GUI's
# progress bars stopped showing for later scans.
@pytest.mark.parametrize("magic", ["mov", "movr"])
def test_mov_magics_restore_the_waiting_hook(RE: RunEngine, magic: str):
    def gui_hook(statuses):
        return None

    shell = InteractiveShell.instance()
    try:
        shell.user_ns["motor"] = motor
        HEXBlueskyMagics.RE = RE
        shell.register_magics(HEXBlueskyMagics)
        RE.waiting_hook = gui_hook

        shell.run_line_magic(magic, "motor 1")

        assert RE.waiting_hook is gui_hook
    finally:
        InteractiveShell.clear_instance()
