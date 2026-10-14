from typing import Any, MutableMapping, Sequence
from ophyd_async.epics.adcore import NDPluginBaseIO, AreaDetector
from bluesky import plan_stubs as bps

class LastDarkFlatUIDTracker:
    """RunEngine subscription compatible class that tracks the last dark-flat scan UID and scan ID."""

    def __init__(self, _metadata_dict: MutableMapping[str, Any]):
        self._metadata_dict = _metadata_dict

    def __call__(self, name: str, doc: dict[str, Any]):
        if name == "start" and doc.get("uid", None) and doc.get("plan_name", None) == "tomo_dark_flat":
            last_uid = doc.get("uid", None)
            last_scan_id = doc.get("scan_id", None)
            self._metadata_dict["last_dark_flat_uid"] = last_uid
            self._metadata_dict["last_dark_flat_scan_id"] = last_scan_id


