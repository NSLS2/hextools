from __future__ import annotations

from enum import Enum


class AcqStatuses(Enum):
    """Enum class for acquisition statuses."""

    IDLE = "Done"
    ACQUIRING = "Count"
