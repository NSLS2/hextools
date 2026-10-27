"""Exporters for tomography data."""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
from bluesky_tiled_plugins import CatalogOfBlueskyRuns
from bluesky_tiled_plugins.clients.bluesky_run import BlueskyRunV3

from hextools.exporters.utils import (
    average_per_projection,
    confirm_required_start_fields,
    create_baseline_dataset,
    find_hdf_datasets,
    read_recorded_captures,
    run_metadata_to_json,
)
from hextools.utils import ensure_available

logger = logging.getLogger("hextools.exporters.tomography")


def resolve_dark_flat_run(
    tiled_client: Mapping[Any, Any],
    run_start: Mapping[str, Any],
    dark_flat_id: str | int | None = None,
) -> BlueskyRunV3 | None:
    """Return the dark/flat run to link into a tomography export, or None.

    Uses ``dark_flat_id`` if given, else the run's ``last_dark_flat_uid``, and
    only if that dark/flat was taken in the same proposal and cycle as the run.
    """
    uid = (
        dark_flat_id
        if dark_flat_id is not None
        else run_start.get("last_dark_flat_uid")
    )
    if uid is None:
        logger.warning("No last_dark_flat_uid in the run start document")
        return None
    try:
        dark_flat_run = tiled_client[uid]
    except KeyError:
        dark_flat_run = None
    if not isinstance(dark_flat_run, BlueskyRunV3):
        logger.warning(f"Dark/flat run {uid} not found; omitting dark/flat entries")
        return None
    # A dark/flat from the same proposal in an earlier cycle is stale too.
    dark_flat_from = tuple(
        dark_flat_run.start.get(k) for k in ("data_session", "cycle")
    )
    scan_from = tuple(run_start.get(k) for k in ("data_session", "cycle"))
    if dark_flat_from != scan_from:
        logger.warning(
            f"Dark/flat run {uid} is from {dark_flat_from}, not {scan_from}; "
            "omitting dark/flat entries"
        )
        return None
    logger.info(f"Using dark/flat run {uid}")
    return dark_flat_run


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

    dark_flat_run = resolve_dark_flat_run(tiled_client, run_start, dark_flat_id)

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

    # When averaging, every PandA capture (e.g. rotation_angle) is written as the
    # mean of each projection's images; the raw link is kept.
    images_to_average = run_start.get("images_to_average", 1)
    num_projections = run_start.get("num_points")
    if not isinstance(images_to_average, int):
        raise ValueError(f"images_to_average is not an int: {images_to_average!r}")
    num_events = (run.stop or {}).get("num_events")
    recorded = num_events.get("primary") if isinstance(num_events, Mapping) else None
    # A run that did not finish leaves files longer than its events, which Tiled
    # will not serve, so its captures are read from the files up to what it recorded.
    unfinished = (
        isinstance(num_projections, int)
        and isinstance(recorded, int)
        and recorded < num_projections
    )
    if unfinished:
        logger.warning(
            f"Run recorded {recorded} of {num_projections} projections; exporting "
            "those. Linked detector data may hold frames past the last projection."
        )
    averaged = {}
    for name, (relpath, dataset, ndim) in primary.items():
        if name in det_names or (
            images_to_average == 1 and ndim == 1 and not unfinished
        ):
            continue
        if not isinstance(num_projections, int):
            raise ValueError(f"num_points is not an int: {num_projections!r}")
        if unfinished:
            values = read_recorded_captures(
                scan_id_dir_path / relpath, dataset, recorded, images_to_average
            )
        else:
            values = run["primary"][name].read()
        averaged[name] = average_per_projection(
            values, recorded if unfinished else num_projections, images_to_average
        )
        links[f"{name}_raw"] = links.pop(name)
        logger.info(f"Averaging {name} over {images_to_average} images per projection")

    if dark_flat_run is not None:
        for stream_name in ("dark", "flat"):
            found = find_hdf_datasets(dark_flat_run, stream_name, scan_id_dir_path)
            missing = set(det_names) - found.keys()
            if missing:
                logger.warning(
                    f"Dark/flat run {dark_flat_run.start.get('uid')} has no "
                    f"{stream_name} data for {sorted(missing)}"
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
