
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from bluesky_tiled_plugins import CatalogOfBlueskyRuns
import h5py
import xarray as xr
from bluesky_tiled_plugins.clients.bluesky_run import BlueskyRunV3
import json

from hextools.utils import ensure_available

from .utils import _json_default

def export_edxd_run_to_h5(
    edxd_scan: BlueskyRunV3 | str | int,
    tiled_client: CatalogOfBlueskyRuns | None = None
):

    # TODO: add validation to check that it is in fact an EDXD run
    if isinstance(edxd_scan, (str, int)):
        if tiled_client is None:
            tiled_client = ensure_available(CatalogOfBlueskyRuns, c=tiled_client)
        run = tiled_client[edxd_scan]
    else:
        run = edxd_scan

    if not isinstance(run, BlueskyRunV3):
        raise ValueError("Failed to resolve EDXD run to a BlueskyRunV3 instance")

    primary_data = run["primary"].read()
    baseline_data = run["baseline"].read()
    proposal = run.start.get('data_session')
    cycle = run.start.get('cycle')
    scan_id = run.start.get('scan_id')
    if any(v is None for v in (proposal, cycle, scan_id)):
        raise ValueError("Missing required run metadata: proposal, cycle, or scan_id")
    scan_id_dir_path = Path(f"/nsls2/data/hex/proposals/{cycle}/{proposal}/edxd/metadata/scan_{scan_id:05}")
    if not scan_id_dir_path.exists():
        scan_id_dir_path.mkdir(parents=True, exist_ok=True)

    h5_file_path = scan_id_dir_path / f"scan_{scan_id:05}.nxs"

    with h5py.File(h5_file_path, "w") as f:
        primary_group = f.create_group("primary")
        for key, value in primary_data.items():
            primary_group.create_dataset(key, data=value)

        baseline_group = f.create_group("baseline")
        for key, value in baseline_data.items():
            baseline_group.create_dataset(key, data=value)

    all_metadata = {
        "run": run.metadata,
        "primary": run["primary"].metadata,
        "baseline": run["baseline"].metadata,
    }

    json_file_path = scan_id_dir_path / f"scan_{scan_id:05}.json"
    with open(json_file_path, "w") as fp:
        json.dump(all_metadata, fp, indent=4, default=_json_default)