from __future__ import annotations

from enum import Enum


class AcqStatuses(Enum):
    """Enum class for acquisition statuses."""

    IDLE = "Done"
    ACQUIRING = "Count"


class StageStates(Enum):
    """Enum class for stage states."""

    UNSTAGED = "unstaged"
    STAGED = "staged"


class TrueFalse(Enum):
    """Enum class for bool states."""

    FALSE = "FALSE"
    TRUE = "TRUE"
