"""Utility functions for exporters."""

from collections.abc import Mapping

import numpy as np

__all__ = ["_json_default"]


def _json_default(obj):
    if isinstance(obj, Mapping):
        return dict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)
