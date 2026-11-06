"""Exporters for XRD data."""

import logging
from pathlib import Path

import h5py
from bluesky_tiled_plugins import CatalogOfBlueskyRuns
from bluesky_tiled_plugins.clients.bluesky_event_stream import BlueskyEventStreamV3
from bluesky_tiled_plugins.clients.bluesky_run import BlueskyRunV3
from tiled.utils import path_from_uri

from hextools.exporters.utils import (
    confirm_required_start_fields,
    create_baseline_dataset,
    find_hdf_datasets,
    run_metadata_to_json,
)
from hextools.utils import ensure_available

logger = logging.getLogger("hextools.exporters.xrd")


def export_xrd_dark_reference(
    dark_ref_scan: BlueskyRunV3 | str | int,
    tiled_client: CatalogOfBlueskyRuns | None = None,
):
    """Symlink an XRD dark reference's TIFF file(s) and write its metadata to JSON.

    Files go in the proposal's ``xrd/metadata/dark_ref/scan_<scan_id>`` directory.

    Parameters
    ----------
    dark_ref_scan : BlueskyRunV3 | str | int
        The run to export, or its uid/scan id to look up in ``tiled_client``.
    tiled_client : CatalogOfBlueskyRuns | None, optional
        The catalog to look up the run in. If None, will pull from global namespace.
    """
    if isinstance(dark_ref_scan, (str, int)):
        tiled_client = ensure_available(CatalogOfBlueskyRuns, c=tiled_client)
        run = tiled_client[dark_ref_scan]
    else:
        run = dark_ref_scan

    if not isinstance(run, BlueskyRunV3):
        raise ValueError("Failed to resolve dark reference to a BlueskyRunV3 instance")

    proposal, cycle, scan_id = confirm_required_start_fields(run.start)
    detectors = run.start.get("detectors")
    if not isinstance(detectors, list) or len(detectors) != 1:
        raise ValueError(
            f"Expected one detector in the start document, got {detectors!r}"
        )
    det_name = detectors[0]

    scan_id_dir_path = Path(
        f"/nsls2/data/hex/proposals/{cycle}/{proposal}/xrd/dark_ref/scan_{scan_id:05}"
    )
    scan_id_dir_path.mkdir(parents=True, exist_ok=True)

    tiff_paths = [
        path_from_uri(asset.data_uri)
        for data_source in run["primary"][det_name].data_sources() or []
        if "image/tiff" in (data_source.mimetype or "")
        for asset in data_source.assets
    ]
    if not tiff_paths:
        raise ValueError(f"No TIFF data for det {det_name!r} in the primary stream")

    for tiff_path in tiff_paths:
        if not tiff_path.is_file():
            raise FileNotFoundError(f"{tiff_path} does not exist")
        link_path = scan_id_dir_path / tiff_path.name
        link_path.unlink(missing_ok=True)
        link_path.symlink_to(tiff_path)
        logger.info(f"Linked {link_path} -> {tiff_path}")

    run_metadata_to_json(run, scan_id, scan_id_dir_path)
    logger.info(
        f"Finished exporting XRD dark reference {scan_id} to {scan_id_dir_path}"
    )


def export_xrd_run_to_h5(
    xrd_scan: BlueskyRunV3 | str | int,
    tiled_client: CatalogOfBlueskyRuns | None = None,
):
    """Export an XRD run to a NeXus/HDF5 file and its metadata to JSON.

    The detector image stack is linked at ``/entry/data/data``; every other
    primary stream field is read from Tiled and written to
    ``/entry/data/<field>``. Files are written to the proposal's
    ``xrd/metadata/scan_<scan_id>`` directory.

    Parameters
    ----------
    xrd_scan : BlueskyRunV3 | str | int
        The run to export, or its uid/scan id to look up in ``tiled_client``.
    tiled_client : CatalogOfBlueskyRuns | None, optional
        The catalog to look up the run in. If None, will pull from global namespace.
    """
    if isinstance(xrd_scan, (str, int)):
        tiled_client = ensure_available(CatalogOfBlueskyRuns, c=tiled_client)
        run = tiled_client[xrd_scan]
    else:
        run = xrd_scan

    if not isinstance(run, BlueskyRunV3):
        raise ValueError("Failed to resolve XRD run to a BlueskyRunV3 instance")

    proposal, cycle, scan_id = confirm_required_start_fields(run.start)
    is_calib = run.start.get("plan_name", "") == "xrd_calibration"
    logger.info(
        f"Exporting XRD scan {scan_id} (uid {run.start.get('uid')})"
        f" for proposal {proposal}, cycle {cycle}"
    )

    scan_id_dir_path = Path(
        f"/nsls2/data/hex/proposals/{cycle}/{proposal}/xrd/{'metadata' if not is_calib else 'calibration'}/scan_{scan_id:05}"  # noqa: E501
    )
    scan_id_dir_path.mkdir(parents=True, exist_ok=True)

    detectors = run.start.get("detectors")
    if not isinstance(detectors, list) or len(detectors) != 1:
        raise ValueError(
            f"Expected one detector in the start document, got {detectors!r}"
        )
    det_name = detectors[0]
    hdf_datasets = find_hdf_datasets(run, "primary", scan_id_dir_path)
    if det_name not in hdf_datasets:
        raise ValueError(
            f"No HDF5 dataset for detector {det_name!r} in the primary stream"
        )
    det_relpath, det_dataset, _ = hdf_datasets[det_name]  # type: ignore

    primary = run["primary"]
    if not isinstance(primary, BlueskyEventStreamV3):
        raise TypeError(
            f"Expected a BlueskyEventStreamV3, got {type(primary).__name__}"
        )
    other_fields = [key for key in primary if key != det_name]
    primary_data = primary.read(variables=other_fields) if other_fields else {}

    nxs_path = scan_id_dir_path / f"scan_{scan_id:05}.nxs"
    logger.info(f"Writing {nxs_path}")
    with h5py.File(nxs_path, "w") as h5file:
        data_grp = h5file.require_group("/entry/data")
        data_grp["data"] = h5py.ExternalLink(det_relpath, det_dataset)
        logger.info(f"Linked /entry/data/data -> {det_relpath}:{det_dataset}")
        for name, values in primary_data.items():
            data_grp.create_dataset(name, data=values)
            logger.info(f"Wrote /entry/data/{name}")

        logger.info("Writing baseline data")
        create_baseline_dataset(run, h5file)

    logger.info(f"Writing run metadata to {scan_id_dir_path}")
    run_metadata_to_json(run, scan_id, scan_id_dir_path)
    logger.info(f"Finished exporting XRD scan {scan_id} to {nxs_path}")
