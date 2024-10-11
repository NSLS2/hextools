from __future__ import annotations

from enum import Enum, IntEnum


class AcqStatuses(Enum):
    """Enum class for acquisition statuses."""

    IDLE = "Done"
    ACQUIRING = "Count"


class StageStates(Enum):
    """Enum class for stage states."""

    UNSTAGED = "unstaged"
    STAGED = "staged"


class TrueFalse(IntEnum):
    """Enum class for bool states."""

    false = 0
    true = 1
