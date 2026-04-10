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
        the detector keys to get the values for to the returned dictionary

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


def get_detector_parameters_from_tiled(run, det_name=None, keys=None):
    """Auxiliary function to get detector parameters from tiled.

    Parameters:
    -----------
    run : bluesky run
        the bluesky run to get detector parameters for
    det_name : str
        ophyd detector name
    keys : dict
        the detector keys to get the values for to the returned dictionary

    Returns:
    --------
    detector_metadata : dict
        the dictionary with detector parameters
    """
    if det_name is None:
        msg = "The 'det_name' cannot be None"
        raise ValueError(msg)
    try:
        # make sure det_name is correct
        config = run.primary["config"][det_name].read()
    except KeyError as err:
        msg = f"{err} det_name is incorrect. Check ophyd device .name"
        raise ValueError(msg) from err
    if keys is None:
        keys = GERM_DETECTOR_KEYS
    group_key = f"{det_name.lower()}_detector"
    detector_metadata = {group_key: {}}
    for key in keys:
        detector_metadata[group_key][key] = config[f"{det_name}_{key}"].data[0]
    return detector_metadata


def nx_export(run, det_name, export_dir=None, file_prefix=None):
    """Function to export bluesky run to NeXus file

    Parameters:
    -----------
    run : bluesky run
        the bluesky run to export to NeXus.
    det_name : str
        the name of the detector to export the data/metadata for.
    export_dir : str (optional)
        the export directory for the resulting file.
    file_prefix : str (optional)
        the file prefix template for the resulting file.
    """
    start_doc = run.start
    if export_dir is None:
        export_dir = start_doc["export_dir"]
    date = datetime.datetime.fromtimestamp(start_doc["time"])

    # TODO: create defaults for file prefixes for different types of scans.
    if file_prefix is None:
        file_prefix = "scan_{start[scan_id]:05d}_{start[calibrant]}_{start[theta]:.3f}deg_{date.month:02d}_{date.day:02d}_{date.year:04d}.nxs"
    rendered_file_name = file_prefix.format(start=start_doc, date=date)

    # for name, doc in run.documents():
    #     if name == "resource" and doc["spec"] == "AD_HDF5_GERM":
    #         resource_root = doc["root"]
    #         resource_path = doc["resource_path"]
    #         h5_filepath = Path(resource_root) / Path(resource_path)
    #             # Path.joinpath(h5_filepath.parent / f"{h5_filepath.stem}.nxs")
    #             # Path.joinpath(Path("/tmp") / f"{h5_filepath.stem}.nxs")  # For testing
    #         break
    nx_filepath = str(Path(export_dir) / Path(rendered_file_name))
    print(f"{nx_filepath = }")

    def get_dtype(value):
        if isinstance(value, str):
            return h5py.special_dtype(vlen=str)
        if isinstance(value, float):
            return np.float32
        if isinstance(value, int):
            return np.int32
        return type(value)

    with h5py.File(nx_filepath, "x") as h5_file:
        entry_grp = h5_file.require_group("entry")
        data_grp = entry_grp.require_group("data")

        meta_dict = get_detector_parameters_from_tiled(run, det_name)
        for _, v in meta_dict.items():
            meta = v
            break
        current_metadata_grp = h5_file.require_group("entry/instrument/detector")
        for key, value in meta.items():
            if key not in current_metadata_grp:
                dtype = get_dtype(value)
                current_metadata_grp.create_dataset(key, data=value, dtype=dtype)

        # External link
        # data_grp["data"] = h5py.ExternalLink(h5_filepath, "entry/data/data")
        data = run.primary["data"][f"{det_name}_image"].read()
        frame_shape = data.shape[1:]
        data_grp.create_dataset(
            "data",
            data=data,
            maxshape=(None, *frame_shape),
            chunks=(1, *frame_shape),
            dtype=data.dtype,
        )
    return nx_filepath


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
    dtype="int64",
    mode="x",
):
    """The function to export the data to an HDF5 file."""

    update_existing = Path(fname).is_file()
    with h5py.File(fname, mode, libver="latest") as h5file_desc:
        frame_shape = data.shape
        if not update_existing:
            group = h5file_desc.create_group(group_name)
            dataset = group.create_dataset(
                group_path,
                data=np.full(fill_value=np.nan, shape=(1, *frame_shape)),
                maxshape=(None, *frame_shape),
                chunks=(1, *frame_shape),
                dtype=dtype,
                compression="gzip",
            )
            frame_num = 0
        else:
            dataset = h5file_desc[f"{group_name}/{group_path}"]
            frame_num = dataset.shape[0]

        h5file_desc.swmr_mode = True

        dataset.resize((frame_num + 1, *frame_shape))
        dataset[frame_num, :, :] = data
        dataset.flush()
        return dataset.shape
