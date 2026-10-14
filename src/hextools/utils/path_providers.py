from collections import Counter
from collections.abc import Sequence
from typing import Any

from nslsii.ophyd_async.providers import TimestampFilenameProvider
import logging

logger = logging.getLogger("hextools")

class StartDocAwareFilenameProvider(TimestampFilenameProvider):
    """Prefix filenames with prefix depending on the current run's plan name.

    Special handling is provided for the ``tomo_dark_flat`` plan, where the first
    file is prefixed with ``dark`` and the second with ``flat``. Other plans use
    a default prefix based on the plan name.
    """

    def __init__(
        self,
        timestamp_format: str = "%Y%m%d_%H%M%S",
    ):
        super().__init__(timestamp_format)
        self._start_doc: dict[str, Any] | None = None
        self._files_per_datakey: Counter[str] = Counter()

    def doc_callback(self, name: str, doc: dict[str, Any]):
        if name == "start":
            self._start_doc = doc
            self._files_per_datakey.clear()
        elif name == "stop":
            self._start_doc = None

    def __call__(self, datakey_name: str | None = None) -> str:
        filename = super().__call__(datakey_name)
        # Without a datakey we can't count files per detector, so don't prefix.
        if datakey_name is None or self._start_doc is None:
            return filename
        plan_name = self._start_doc.get("plan_name")
        if plan_name is None or not isinstance(plan_name, str):
            return filename

        file_index = self._files_per_datakey[datakey_name]
        if plan_name == "tomo_dark_flat":
            if file_index == 0:
                prefix = "dark"
            elif file_index == 1:
                prefix = "flat"
            else:
                logger.warning(f"Unexpected file index for tomo_dark_flat plan: {file_index}. Should be 0 or 1.")
                prefix = plan_name + "_" + str(file_index)

        elif plan_name.startswith("tomo"):
            prefix = "proj"

        elif plan_name.startswith("edxd"):
            prefix = "edxd"
        else:
            prefix = plan_name

        self._files_per_datakey[datakey_name] += 1

        return f"{prefix}_{filename}"
