"""Export EDXD runs from Tiled to HDF5 and JSON files."""

from pathlib import Path

import h5py
from bluesky_tiled_plugins import CatalogOfBlueskyRuns
from bluesky_tiled_plugins.clients.bluesky_run import BlueskyRunV3

from hextools.exporters.utils import confirm_required_start_fields
from hextools.utils import ensure_available

from .utils import run_metadata_to_json


def export_edxd_run_to_h5(
    edxd_scan: BlueskyRunV3 | str | int,
    tiled_client: CatalogOfBlueskyRuns | None = None,
):
    """Export an EDXD run's data to a NeXus/HDF5 file and its metadata to JSON.

    Files are written to the proposal's ``edxd/metadata/scan_<scan_id>`` directory.

    Parameters
    ----------
    edxd_scan : BlueskyRunV3 | str | int
        The run to export, or its uid/scan id to look up in ``tiled_client``.
    tiled_client : CatalogOfBlueskyRuns | None, optional
        The catalog to look up the run in. If None, will pull from global namespace.
    """
    # TODO: add validation to check that it is in fact an EDXD run
    if isinstance(edxd_scan, (str, int)):
        if tiled_client is None:
            tiled_client = ensure_available(CatalogOfBlueskyRuns, c=tiled_client)
        run = tiled_client[edxd_scan]
    else:
        run = edxd_scan

    if not isinstance(run, BlueskyRunV3):
        raise ValueError("Failed to resolve EDXD run to a BlueskyRunV3 instance")

    primary_data = run["primary"].read(
        variables=[key for key in run["primary"] if key != "germ"]
    )
    baseline_data = run["baseline"].read()
    proposal, cycle, scan_id = confirm_required_start_fields(run.start)

    metadata_dir_path = Path(
        f"/nsls2/data/hex/proposals/{cycle}/{proposal}/edxd/metadata/scan_{scan_id:05}"
    )
    raw_export_dir_path = Path(
        f"/nsls2/data/hex/proposals/{cycle}/{proposal}/edxd/raw_data/scan_{scan_id:05}"
    )
    for dir_path in (metadata_dir_path, raw_export_dir_path):
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)

    metadata_nxs_path = metadata_dir_path / f"scan_{scan_id:05}.nxs"

    with h5py.File(metadata_nxs_path, "w") as f:
        primary_group = f.create_group("primary")
        for key, value in primary_data.items():
            primary_group.create_dataset(key, data=value)

        baseline_group = f.create_group("baseline")
        for key, value in baseline_data.items():
            baseline_group.create_dataset(key, data=value)

    run_metadata_to_json(run, scan_id, metadata_dir_path)

    for data_source in run["primary"]["germ"].data_sources():
        for asset in data_source.assets:
            asset_path = Path(asset.data_uri.split("localhost")[1])
            raw_path = raw_export_dir_path / asset_path.name
            raw_path.unlink(missing_ok=True)
            raw_path.symlink_to(asset_path)
