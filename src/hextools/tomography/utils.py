"""Utilities shared by the tomography plans."""

from collections.abc import MutableMapping
from typing import Any

from hextools.utils import DARK_FLAT_MD_KEYS


class LastDarkFlatUIDTracker:
    """RunEngine subscription that tracks the last dark-flat scan UID and scan ID."""

    def __init__(self, _metadata_dict: MutableMapping[str, Any]):
        self._metadata_dict = _metadata_dict

    def __call__(self, name: str, doc: dict[str, Any]):
        if (
            name == "start"
            and doc.get("uid", None)
            and doc.get("plan_name", None) == "tomo_dark_flat"
        ):
            uid_key, scan_id_key = DARK_FLAT_MD_KEYS
            self._metadata_dict[uid_key] = doc.get("uid", None)
            self._metadata_dict[scan_id_key] = doc.get("scan_id", None)
