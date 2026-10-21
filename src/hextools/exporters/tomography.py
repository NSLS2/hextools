"""Exporters for tomography data."""

import logging
from pathlib import Path

import h5py
from bluesky_tiled_plugins import CatalogOfBlueskyRuns
from bluesky_tiled_plugins.clients.bluesky_run import BlueskyRunV3

from hextools.exporters.utils import (
    average_per_projection,
    confirm_required_start_fields,
    create_baseline_dataset,
    find_hdf_datasets,
    run_metadata_to_json,
)
from hextools.utils import ensure_available

logger = logging.getLogger("hextools.exporters.tomography")


def export_tomo_flyscan_to_h5(
    tomo_flyscan: BlueskyRunV3 | str | int,
    tiled_client: CatalogOfBlueskyRuns | None = None,
    dark_flat_id: str | int | None = None,
):
    """Export a tomography flyscan to an HDF5 file."""
    tiled_client = ensure_available(CatalogOfBlueskyRuns, c=tiled_client)

    if isinstance(tomo_flyscan, (str, int)):
        run = tiled_client[tomo_flyscan]
    else:
        run = tomo_flyscan

    if not isinstance(run, BlueskyRunV3):
        raise ValueError(
            "Failed to resolve tomography flyscan to a BlueskyRunV3 instance"
        )

    run_start = run.start
    proposal, cycle, scan_id = confirm_required_start_fields(run_start)
    logger.info(
        f"Exporting tomography scan {scan_id} (uid {run_start.get('uid')})"
        f" for proposal {proposal}, cycle {cycle}"
    )

    dark_flat_uid = dark_flat_id or run_start.get("last_dark_flat_uid")

    if dark_flat_uid is None:
        logger.warning(
            "No last_dark_flat_uid found in the run start document. "
            "Omitting dark/flat entries"
        )
        dark_flat_run = None
    else:
        logger.info(f"Using dark/flat run {dark_flat_uid}")
        dark_flat_run = tiled_client[dark_flat_uid]
        if not isinstance(dark_flat_run, BlueskyRunV3):
            raise ValueError(
                f"Failed to resolve dark/flat run {dark_flat_uid} to a BlueskyRunV3"
            )

    scan_id_dir_path = Path(
        f"/nsls2/data/hex/proposals/{cycle}/{proposal}/tomography/metadata/scan_{scan_id:05}"
    )
    scan_id_dir_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Export directory: {scan_id_dir_path}")

    primary = find_hdf_datasets(run, "primary", scan_id_dir_path)
    # Image stacks are (frame, y, x); 1D datasets are PandA
    # captures like the rotation angle
    det_names = [name for name, (_, _, ndim) in primary.items() if ndim >= 3]
    single_det = len(det_names) == 1
    logger.info(f"Detector datasets: {det_names}")

    links = {}
    for name, (relpath, dataset, _) in primary.items():
        links["data" if single_det and name in det_names else name] = (relpath, dataset)

    # PandA captures with one row per projection (e.g. rotation_angle when
    # averaging) are written as per-projection means; the raw link is kept.
    images_to_average = run_start.get("images_to_average", 1)
    if not isinstance(images_to_average, int):
        raise ValueError(f"images_to_average is not an int: {images_to_average!r}")
    averaged = {}
    for name, (_, _, ndim) in primary.items():
        if name in det_names or ndim != 2:
            continue
        averaged[name] = average_per_projection(
            run["primary"][name].read(), images_to_average
        )
        links[f"{name}_raw"] = links.pop(name)
        logger.info(f"Averaging {name} over {images_to_average} images per projection")

    if dark_flat_run is not None:
        for stream_name in ("dark", "flat"):
            found = find_hdf_datasets(dark_flat_run, stream_name, scan_id_dir_path)
            missing = set(det_names) - found.keys()
            if missing:
                logger.warning(
                    f"Dark/flat run {dark_flat_uid} has no {stream_name} data for "
                    f"{sorted(missing)}"
                )
            for name in set(det_names) & found.keys():
                relpath, dataset, _ = found[name]
                links[stream_name if single_det else f"{stream_name}-{name}"] = (
                    relpath,
                    dataset,
                )

    nxs_path = scan_id_dir_path / f"scan_{scan_id:05}.nxs"
    logger.info(f"Writing {nxs_path}")
    with h5py.File(nxs_path, "w") as h5file:
        data_grp = h5file.require_group("/entry/data")
        for link_name, (relpath, dataset) in links.items():
            data_grp[link_name] = h5py.ExternalLink(relpath, dataset)
            logger.info(f"Linked /entry/data/{link_name} -> {relpath}:{dataset}")
        for name, values in averaged.items():
            data_grp.create_dataset(name, data=values)
            logger.info(f"Wrote /entry/data/{name} ({len(values)} projections)")

        logger.info("Writing baseline data")
        create_baseline_dataset(run, h5file)

    logger.info(f"Writing run metadata to {scan_id_dir_path}")
    run_metadata_to_json(run, scan_id, scan_id_dir_path)
    logger.info(f"Finished exporting tomography scan {scan_id} to {nxs_path}")


def export_tomo_dark_flat_to_h5(
    tomo_dark_flat_scan: BlueskyRunV3 | str | int,
    tiled_client: CatalogOfBlueskyRuns | None,
):
    """Export a tomography dark/flat scan to an HDF5 file."""
    if isinstance(tomo_dark_flat_scan, (str, int)):
        tiled_client = ensure_available(CatalogOfBlueskyRuns, c=tiled_client)
        run = tiled_client[tomo_dark_flat_scan]
    else:
        run = tomo_dark_flat_scan

    if not isinstance(run, BlueskyRunV3):
        raise ValueError(f"Failed to resolve run {run} to a BlueskyRunV3 instance")

    run_start = run.start
    det_names = run_start.get("detectors", [])
    if not isinstance(det_names, list):
        raise ValueError(f"Run {run} has an invalid 'detectors' field: {det_names}")

    data_session, cycle, scan_id = confirm_required_start_fields(run_start)

    scan_id_dir_path = Path(
        f"/nsls2/data/hex/proposals/{cycle}/{data_session}/tomography/metadata/scan_{scan_id:05}"
    )
    scan_id_dir_path.mkdir(parents=True, exist_ok=True)

    links = {}
    for stream_name in ("dark", "flat"):
        found = find_hdf_datasets(run, stream_name, scan_id_dir_path)
        for name in det_names:
            relpath, dataset, _ = found[name]
            single_det = len(det_names) == 1
            links[stream_name if single_det else f"{stream_name}-{name}"] = (
                relpath,
                dataset,
            )

    nxs_path = scan_id_dir_path / f"scan_{scan_id:05}.nxs"
    logger.info(f"Writing {nxs_path}")
    with h5py.File(nxs_path, "w") as h5file:
        data_grp = h5file.require_group("/entry/data")
        for link_name, (relpath, dataset) in links.items():
            data_grp[link_name] = h5py.ExternalLink(relpath, dataset)
            logger.info(f"Linked /entry/data/{link_name} -> {relpath}:{dataset}")

        logger.info("Writing baseline data")
        create_baseline_dataset(run, h5file)

    logger.info(f"Writing run metadata to {scan_id_dir_path}")
    run_metadata_to_json(run, scan_id, scan_id_dir_path)
    logger.info(
        f"Finished exporting tomography dark/flat run {scan_id} to {scan_id_dir_path}"
    )
