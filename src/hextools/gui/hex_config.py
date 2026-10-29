"""HEX-specific policy adapters for generic GUI extension points."""

from collections.abc import Callable, MutableMapping
from typing import Any

from hextools.photon_delivery_system.shutter import (
    Shutter,
    ensure_shutter_closed,
    ensure_shutter_open,
)


def shutter_controls(
    namespace: MutableMapping[str, Any],
) -> tuple[dict[str, Shutter], Callable[[str, bool], str]]:
    """Select HEX shutters and supply the manual, zero-settling command policy."""
    shutters = {}
    seen = set()
    for name, obj in sorted(namespace.items()):
        if (
            not name.startswith("_")
            and isinstance(obj, Shutter)
            and id(obj) not in seen
        ):
            shutters[name] = obj
            seen.add(id(obj))

    def command(name: str, open_shutter: bool) -> str:
        plan = ensure_shutter_open if open_shutter else ensure_shutter_closed
        namespace.setdefault(plan.__name__, plan)
        return f"RE({plan.__name__}({name}, allow_actuation=True, sleep_time=0.0))"

    return shutters, command
