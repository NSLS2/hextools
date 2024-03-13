from __future__ import annotations

import datetime


def now():
    """A helper function to return ISO 8601 formatted datetime string."""
    return datetime.datetime.now().isoformat()


def replace_curlies(string, how_many=2):
    """A helper function to replace multiple curly braces with one."""
    return string.replace("{" * how_many, "{").replace("}" * how_many, "}")
