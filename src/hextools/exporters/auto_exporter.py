"""Run metadata exporters when a Bluesky run in Tiled closes.

Subscribes to the run catalog over Tiled's websocket API, so the Tiled server must
have its streaming cache enabled.
"""

import argparse
import csv
import logging
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from bluesky_tiled_plugins import CatalogOfBlueskyRuns
from bluesky_tiled_plugins.clients.bluesky_run import BlueskyRunV3
from tiled.client import from_uri
from tiled.client.stream import LiveChildMetadataUpdated

from hextools.exporters.edxd import export_edxd_run_to_h5
from hextools.exporters.tomography import (
    export_tomo_dark_flat_to_h5,
    export_tomo_flyscan_to_h5,
)
from hextools.log import configure_logger
from hextools.specs import EDXD_SCAN_V1, TOMO_DARK_FLAT_V1, TOMO_FLYSCAN_V1

logger = logging.getLogger("hextools.exporters.auto_exporter")

EXPORTERS: dict[tuple[str, str | None], Callable[..., Any]] = {
    (TOMO_FLYSCAN_V1.name, TOMO_FLYSCAN_V1.version): export_tomo_flyscan_to_h5,
    (TOMO_DARK_FLAT_V1.name, TOMO_DARK_FLAT_V1.version): export_tomo_dark_flat_to_h5,
    (EDXD_SCAN_V1.name, EDXD_SCAN_V1.version): export_edxd_run_to_h5,
}

# Run metadata updates carry the full start document, which can exceed the 1 MB default
MAX_MESSAGE_BYTES = 10 * 1024 * 1024

CSV_COLUMNS = ("scan_id", "plan_name", "uid", "status", "export_status")


def _record_run(csv_path: Path, run: BlueskyRunV3, export_status: str) -> None:
    uid = run.start["uid"]
    try:
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(CSV_COLUMNS)
            writer.writerow(
                [
                    run.start.get("scan_id"),
                    run.start.get("plan_name"),
                    uid,
                    run.stop.get("exit_status"),
                    export_status,
                ]
            )
    except OSError:
        logger.exception(f"Failed to record run {uid} in {csv_path}")


def auto_export_run(
    run: BlueskyRunV3, catalog: CatalogOfBlueskyRuns, csv_path: str | Path | None = None
) -> None:
    """Run every exporter registered for ``run``'s specs, raising if any of them failed.

    A run that did not end in success is skipped; export it by hand if needed.
    If ``csv_path`` is given, append the run's scan_id, plan_name, uid and export
    status to it.
    """
    uid = run.start["uid"]
    exit_status = run.stop.get("exit_status")
    if exit_status != "success":
        logger.info(f"Run {uid} ended with exit status {exit_status}; not exporting")
        if csv_path is not None:
            _record_run(Path(csv_path), run, "skipped")
        return
    specs = {(spec.name, spec.version) for spec in run.specs}
    exporters = [EXPORTERS[spec] for spec in specs if spec in EXPORTERS]
    if not exporters:
        logger.info(f"Run {uid} has specs {sorted(specs)}; no exporter registered")

    failed = []
    for exporter in exporters:
        logger.info(f"Running {exporter.__name__} for run {uid}")  # type: ignore
        try:
            exporter(run, tiled_client=catalog)
        except Exception:
            logger.exception(f"{exporter.__name__} failed for run {uid}")  # type: ignore
            failed.append(exporter.__name__)  # type: ignore

    if csv_path is not None:
        _record_run(Path(csv_path), run, "failure" if failed else "success")
    if failed:
        raise RuntimeError(f"Exporters failed for run {uid}: {failed}")


class RunEndSubscription:
    """Ctx manager that creates an end-of-run ws subscription to tiled.

    Use as a context manager; the websocket is closed on exit.
    """

    def __init__(
        self, catalog: CatalogOfBlueskyRuns, callback: Callable[[BlueskyRunV3], None]
    ):
        self._catalog = catalog
        self._callback = callback
        # partial objects have no __name__, so fall back to the wrapped function's
        self._callback_name = getattr(callback, "__name__", None) or getattr(
            getattr(callback, "func", None), "__name__", repr(callback)
        )
        self._handled: set[str] = set()
        self._subscription = catalog.subscribe()
        self._subscription.child_metadata_updated.add_callback(
            self._on_metadata_updated
        )
        self._subscription.stream_closed.add_callback(self._on_stream_closed)
        logger.debug(
            f"Created run-end subscription on {catalog.uri} "
            f"calling {self._callback_name}"
        )

    def __enter__(self) -> "RunEndSubscription":
        """Enter the run-end subscription context."""
        logger.info(
            f"Entering run-end subscription context for catalog {self._catalog.uri}"
        )
        return self

    def __exit__(self, *exc_info) -> None:
        """Close the subscription and wait for any in-progress callback to finish."""
        logger.info(
            "Closing the Tiled subscription."
            "Waiting for any in-progress callback to finish"
        )
        self._subscription.disconnect()
        logger.info(
            f"Closed the Tiled subscription after handling {len(self._handled)} run(s)"
        )

    def run(self) -> None:
        """Block and handle run close events until stopped."""
        logger.info(f"Listening for closed runs on {self._catalog.uri}")
        # Only new updates; the subscription reconnects on its own
        # after dropped connections
        self._subscription.start(start=None, max_size=MAX_MESSAGE_BYTES)
        logger.info("Tiled subscription stopped")

    def _on_metadata_updated(self, update: LiveChildMetadataUpdated) -> None:
        # TiledWriter adds the stop document to the run's metadata when the run closes
        if "stop" not in update.metadata:
            logger.debug(
                f"Ignoring metadata update for {update.key}: run has not closed"
            )
            return

        start = update.metadata.get("start", {})
        uid = start.get("uid") or update.key
        if uid in self._handled:
            logger.debug(f"Run {uid} was already handled, ignoring repeat update")
            return
        self._handled.add(uid)

        exit_status = update.metadata["stop"].get("exit_status")
        logger.info(
            f"Run {uid} (scan {start.get('scan_id')}, {start.get('plan_name')}) closed "
            f"with exit status {exit_status}"
        )
        t0 = time.monotonic()
        try:
            run = self._catalog[uid]
            if not isinstance(run, BlueskyRunV3):
                raise TypeError(
                    f"Run {uid} is a {type(run).__name__}, not a BlueskyRunV3"
                )
            logger.info(f"Calling {self._callback_name} for run {uid}")
            self._callback(run)
        except Exception:
            logger.exception(f"{self._callback_name} failed for run {uid}")
        else:
            logger.info(
                f"{self._callback_name} finished for run {uid} in "
                f"{time.monotonic() - t0:.1f} s"
            )

    def _on_stream_closed(self, sub) -> None:
        logger.warning(f"Tiled closed the stream for {sub}")


def main() -> None:
    """Entry point for the auto exporter script."""
    parser = argparse.ArgumentParser(
        description="Run metadata exporters when Bluesky runs close in Tiled."
    )
    parser.add_argument("--tiled-uri", default="https://tiled.nsls2.bnl.gov")
    parser.add_argument(
        "--catalog",
        default="hex/raw",
        help="path of the run catalog on the Tiled server",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="CSV file to append each closed run's export status to",
    )
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    args = parser.parse_args()

    configure_logger(level=logging.DEBUG if args.debug else logging.INFO)

    catalog = from_uri(args.tiled_uri)
    for segment in args.catalog.strip("/").split("/"):
        catalog = catalog[segment]

    try:
        callback = partial(auto_export_run, catalog=catalog, csv_path=args.csv)
        with RunEndSubscription(catalog, callback) as subscription:
            subscription.run()
    except KeyboardInterrupt:
        logger.info("Interrupted; closed the Tiled subscription and exiting")


if __name__ == "__main__":
    main()
