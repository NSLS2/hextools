"""Compatibility exports for the RunEngine message narrator."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from hextools.gui.narrator import (
    GroupLabeler,
    StreamNarrator,
    describe_message,
    narrate_stream,
)
from hextools.gui.narrator import MsgHookNarrator as _GenericMsgHookNarrator


class MsgHookNarrator(_GenericMsgHookNarrator):
    """Keep the original hextools logger default for utility-module callers."""

    def __init__(
        self,
        *,
        emit: Callable[[str], Any] | None = None,
        logger: logging.Logger | str = "hextools",
        level: int = logging.INFO,
        show_timestamps: bool = False,
        window: float = 60.0,
    ) -> None:
        super().__init__(
            emit=emit,
            logger=logger,
            level=level,
            show_timestamps=show_timestamps,
            window=window,
        )


nl_msg_hook = MsgHookNarrator()

__all__ = [
    "GroupLabeler",
    "MsgHookNarrator",
    "StreamNarrator",
    "describe_message",
    "narrate_stream",
    "nl_msg_hook",
]
