
from collections.abc import Mapping
from typing import Any
import logging
import os
from pathlib import Path

import numpy as np

from bluesky_tiled_plugins import CatalogOfBlueskyRuns
from tiled.structures.data_source import DataSource
from tiled.utils import path_from_uri
import h5py
import xarray as xr
from bluesky_tiled_plugins.clients.bluesky_run import BlueskyRunV3
from bluesky_tiled_plugins.clients.bluesky_event_stream import BlueskyEventStreamV3
import json

from hextools.utils import ensure_available

logger = logging.getLogger("hextools.exporters.utils")


__all__ = ["_json_default"]

def _json_default(obj):
    if isinstance(obj, Mapping):
        return dict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def run_metadata_to_json(run: BlueskyRunV3, scan_id: int, metadata_dir: Path):
    all_metadata = {
        "run": run.metadata,
    }
    all_metadata.update({stream: run[stream].metadata for stream in run})

    json_file_path = metadata_dir / f"scan_{scan_id:05}.json"
    with open(json_file_path, "w") as fp:
        json.dump(all_metadata, fp, indent=4, default=_json_default)


def confirm_required_start_fields(run_start: dict[str, Any]) -> tuple[str, str, int]:
    required_fields = ["data_session", "cycle", "scan_id"]
    field_values = [run_start.get(field) for field in required_fields]
    for n, v in zip(required_fields, field_values):
        if v is None:
            raise ValueError(f"Missing required start field: {n}!")

    return tuple(field_values)  # type: ignore


def create_baseline_dataset(run: BlueskyRunV3, h5file: h5py.File):
    if "baseline" not in run:
        logger.warning("No baseline data found in the run.")
        return
    baseline_data = run["baseline"].read()
    baseline_group = h5file.create_group("baseline")
    for key, value in baseline_data.items():
        baseline_group.create_dataset(key, data=value)

def find_hdf_datasets(bs_run: BlueskyRunV3, stream_name: str, scan_id_dir_path: Path) -> dict[str, tuple[str, str, int]]:
    """Map dataset name to (relative file path, internal dataset path, ndim)."""
    stream = bs_run[stream_name]
    if not isinstance(stream, BlueskyEventStreamV3):
        raise ValueError(f"Run {bs_run.start['uid']} has no '{stream_name}' event stream")
    # One request for the whole stream; CompositeClient[key] re-lists the stream on every lookup
    contents = stream.include_data_sources().get_contents()
    logger.debug(f"Fetched {len(contents)} entries from stream {stream.item['id']}")
    found = {}
    for dataset_name, item in contents.items():
        data_sources = [DataSource.from_json(d) for d in item["attributes"].get("data_sources") or []]
        data_source = next((ds for ds in data_sources if ds.mimetype == "application/x-hdf5"), None)
        if data_source is None:
            logger.debug(f"Skipping {dataset_name}: no HDF5 data source")
            continue
        filepath = path_from_uri(data_source.assets[0].data_uri)
        if not filepath.is_file():
            raise FileNotFoundError(f"{filepath} does not exist")
        dataset = data_source.parameters.get("dataset", "entry/data/data")
        if isinstance(dataset, (list, tuple)):
            dataset = "/".join(dataset)
        # Relative links keep the .nxs valid if the proposal directory is moved
        relpath = os.path.relpath(filepath, scan_id_dir_path)
        found[dataset_name] = (relpath, dataset, len(item["attributes"]["structure"]["shape"]))
    return found
