"""IPython magics for the HEX session."""

from bluesky.magics import BlueskyMagics
from IPython.core.magic import line_magic, magics_class


@magics_class
class HEXBlueskyMagics(BlueskyMagics):
    """Bluesky magics whose ``%mov`` and ``%movr`` put back the waiting hook.

    bluesky's versions set ``RE.waiting_hook`` to None afterwards, which drops
    the GUI's progress bars for every later scan.
    """

    @line_magic
    def mov(self, line):
        return self._keeping_waiting_hook(BlueskyMagics.mov, line)

    @line_magic
    def movr(self, line):
        return self._keeping_waiting_hook(BlueskyMagics.movr, line)

    def _keeping_waiting_hook(self, magic, line):
        hook = self.RE.waiting_hook
        try:
            return magic(self, line)
        finally:
            self.RE.waiting_hook = hook
