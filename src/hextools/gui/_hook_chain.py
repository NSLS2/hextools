"""RunEngine hooks that windows sharing one RE can remove in any order."""

from collections.abc import Callable
from typing import Any


class ChainedHook:
    """Calls ``previous`` then ``own`` until detached; afterwards only ``previous``."""

    def __init__(self, own: Callable[..., Any], previous: Callable[..., Any] | None):
        self.own = own
        self.previous = previous
        self.detached = False

    def __call__(self, *args):
        previous = live_hook(self.previous)
        if previous is not None:
            previous(*args)
        if not self.detached:
            self.own(*args)


def live_hook(hook):
    """Return ``hook``, or the first hook beneath it that is not detached."""
    while getattr(hook, "detached", False):
        hook = hook.previous
    return hook


def install_hook(re, attr: str, own: Callable[..., Any]) -> ChainedHook:
    """Chain ``own`` onto ``re.<attr>`` and return the installed hook."""
    hook = ChainedHook(own, getattr(re, attr))
    setattr(re, attr, hook)
    return hook


def uninstall_hook(re, attr: str, hook) -> None:
    """Detach ``hook``; if it is still ``re.<attr>``, restore the live hook beneath."""
    hook.detached = True
    if getattr(re, attr) is hook:
        setattr(re, attr, live_hook(hook.previous))
