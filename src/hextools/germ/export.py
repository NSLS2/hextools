from __future__ import annotations

import datetime
from pathlib import Path

import h5py
import numpy as np

GERM_DETECTOR_KEYS = [
    "count_time",
    "gain",
    "shaping_time",
    "hv_bias",
    "voltage",
]


def get_detector_parameters(det=None, keys=None):
    """Auxiliary function to get detector parameters.

    Parameters:
    -----------
    det : ophyd.Device
        ophyd detector object
    keys : dict
        the detector keys to to get the values for to the returned dictionary

    Returns:
    --------
    detector_metadata : dict
        the dictionary with detector parameters
    """
    if det is None:
        msg = "The 'det' cannot be None"
        raise ValueError(msg)
    if keys is None:
        keys = GERM_DETECTOR_KEYS
    group_key = f"{det.name.lower()}_detector"
    detector_metadata = {group_key: {}}
    for key in keys:
        obj = getattr(det, key)
        as_string = bool(obj.enum_strs)
        detector_metadata[group_key][key] = obj.get(as_string=as_string)
    return detector_metadata


def nx_export_callback(name, doc):
    """A bluesky callback function for NeXus file exporting.

    Parameters:
    -----------
    name : str
        the name of the incoming bluesky/event-model document (start, event, stop, etc.)
    doc : dict
        the dictionary representing the document

    Check https://blueskyproject.io/event-model/main/user/explanations/data-model.html for details.
    """
    print(f"Exporting the nx file at {datetime.datetime.now().isoformat()}")
    if name == "stop":
        run_start = doc["run_start"]
        # TODO: rewrite with SingleRunCache.
        try:
            db = globals().get("db", None)
            hdr = db[run_start]
        except Exception as exc:
            msg = "The databroker object 'db' is not defined"
            raise RuntimeError(msg) from exc
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
                    if isinstance(value, float):
                        return np.float32
                    if isinstance(value, int):
                        return np.int32
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


def save_hdf5(
    fname,
    data,
    group_name="/entry",
    group_path="data/data",
    dtype="float32",
    mode="x",
    update_existing=False,
):
    """The function to export the data to an HDF5 file."""
    h5file_desc = h5py.File(fname, mode, libver="latest")
    frame_shape = data.shape
    if not update_existing:
        group = h5file_desc.create_group(group_name)
        dataset = group.create_dataset(
            "data/data",
            data=np.full(fill_value=np.nan, shape=(1, *frame_shape)),
            maxshape=(None, *frame_shape),
            chunks=(1, *frame_shape),
            dtype=dtype,
        )
        frame_num = 0
    else:
        dataset = h5file_desc[f"{group_name}/{group_path}"]
        frame_num = dataset.shape[0]

    h5file_desc.swmr_mode = True

    dataset.resize((frame_num + 1, *frame_shape))
    dataset[frame_num, :, :] = data
    dataset.flush()
