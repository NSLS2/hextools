"""HEX-specific path and filename providers."""

import logging
from collections import Counter
from datetime import date
from pathlib import Path, PurePath
from typing import Any

from nslsii.ophyd_async.providers import NSLS2PathProvider, TimestampFilenameProvider
from ophyd_async.core import UUIDFilenameProvider, YMDPathProvider

logger = logging.getLogger("hextools")


class HEXPathProvider(NSLS2PathProvider):
    """NSLS2 provider that builds the read and write paths with their own separators.

    The base class joins the date directories with a single separator for both
    paths, so a Windows write dir puts invalid separator into the Linux read path.
    """

    def _generate_directory_path(
        self, base_path: PurePath, datakey_name: str
    ) -> PurePath:
        ymd = date.today().strftime("%Y %m %d").split()[: int(self._granularity)]
        directory_path = (
            base_path
            / str(self._metadata_dict["cycle"])
            / str(self._metadata_dict["data_session"])
            / "assets"
            / datakey_name
        ).joinpath(*ymd)
        if self._include_scan_id_dir:
            directory_path /= (
                f"scan_{self._metadata_dict['scan_id']:0{self._scan_id_padding}}"
            )
        return directory_path


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
                logger.warning(
                    f"Unexpected file index for tomo_dark_flat plan: {file_index}. "
                    "Should be 0 or 1."
                )
                prefix = plan_name + "_" + str(file_index)

        elif plan_name.startswith("tomo"):
            prefix = "proj"

        elif plan_name.startswith("edxd"):
            prefix = "edxd"
        else:
            prefix = plan_name

        self._files_per_datakey[datakey_name] += 1

        return f"{prefix}_{filename}"


def mock_path_provider(base_directory: str | Path) -> YMDPathProvider:
    """Path provider for mock detectors: dated folders, a new file name per call.

    Timestamp names repeat within a second, so two short mock runs could write
    to the same file.
    """
    return YMDPathProvider(UUIDFilenameProvider(), base_directory)
