
from collections.abc import Mapping
import logging
from pathlib import Path

import numpy as np

from bluesky_tiled_plugins import CatalogOfBlueskyRuns
from tiled.client.array import ArrayClient
import h5py
import xarray as xr
from bluesky_tiled_plugins.clients.bluesky_run import BlueskyRunV3
import json

from hextools.utils import ensure_available

logger = logging.getLogger("hextools")


def export_tomo_flyscan_to_h5(
    tomo_flyscan: BlueskyRunV3 | str | int,
    tiled_client: CatalogOfBlueskyRuns | None = None
):

    tiled_client = ensure_available(CatalogOfBlueskyRuns, c=tiled_client)

    if isinstance(tomo_flyscan, (str, int)):
        run = tiled_client[tomo_flyscan]
    else:
        run = tomo_flyscan

    if not isinstance(run, BlueskyRunV3):
        raise ValueError("Failed to resolve tomography flyscan to a BlueskyRunV3 instance")

    run_start = run.start
    dark_flat_uid = run_start.get("last_dark_flat_uid")
    proposal = run_start.get('data_session')
    cycle = run_start.get('cycle')
    scan_id = run_start.get('scan_id')

    if dark_flat_uid is None:
        logger.warning("No last_dark_flat_uid found in the run start document. Omitting dark/flat entries")

    baseline_data = run["baseline"].read()
    scan_id_dir_path = Path(f"/nsls2/data/hex/proposals/{cycle}/{proposal}/tomography/metadata/scan_{scan_id:05}")

    def link_hdf_dataset(det: ArrayClient, h5file: h5py.File):

    for det in run["primary"]:
        if not isinstance(det, ArrayClient):
            continue
        det.data_sources()