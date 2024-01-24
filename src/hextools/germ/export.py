from __future__ import annotations

import datetime
from pathlib import Path

import h5py
import numpy as np

db = None


GERM_DETECTOR_KEYS = [
    "count_time",
    "gain",
    "shaping_time",
    "hv_bias",
    "voltage",
]


def get_detector_parameters(det=None, keys=GERM_DETECTOR_KEYS):
    if det is None:
        msg = "The 'det' cannot be None"
        raise ValueError(msg)
    group_key = f"{det.name.lower()}_detector"
    detector_metadata = {group_key: {}}
    for key in keys:
        obj = getattr(det, key)
        as_string = bool(obj.enum_strs)
        detector_metadata[group_key][key] = obj.get(as_string=as_string)
    return detector_metadata


def nx_export_callback(name, doc):
    print(f"Exporting the nx file at {datetime.datetime.now().isoformat()}")
    if name == "stop":
        run_start = doc["run_start"]
        # TODO: rewrite with SingleRunCache.
        hdr = db[run_start]
        for nn, dd in hdr.documents():
            if nn == "resource" and dd["spec"] == "AD_HDF5_GERM":
                resource_root = dd["root"]
                resource_path = dd["resource_path"]
                h5_filepath = Path(resource_root) / Path(resource_path)
                nx_filepath = str(
                    Path.joinpath(h5_filepath.parent / f"{h5_filepath.stem}.nxs")
                )
                # TODO 1: prepare metadata
                # TODO 2: save .nxs file

                def get_dtype(value):
                    if isinstance(value, str):
                        return h5py.special_dtype(vlen=str)
                    elif isinstance(value, float):
                        return np.float32
                    elif isinstance(value, int):
                        return np.int32
                    else:
                        return type(value)

                with h5py.File(nx_filepath, "w") as h5_file:
                    entry_grp = h5_file.require_group("entry")
                    data_grp = entry_grp.require_group("data")

                    meta_dict = get_detector_parameters()
                    for _, v in meta_dict.items():
                        meta = v
                        break
                    current_metadata_grp = h5_file.require_group(
                        "entry/instrument/detector"
                    )  # TODO: fix the location later.
                    for key, value in meta.items():
                        if key not in current_metadata_grp:
                            dtype = get_dtype(value)
                            current_metadata_grp.create_dataset(
                                key, data=value, dtype=dtype
                            )

                    # External link
                    data_grp["data"] = h5py.ExternalLink(h5_filepath, "entry/data/data")
