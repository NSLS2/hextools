from __future__ import annotations

import datetime


def now():
    """A helper function to return ISO 8601 formatted datetime string."""
    return datetime.datetime.now().isoformat()
