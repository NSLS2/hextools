from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import tifffile
from area_detector_handlers import HandlerBase


class AreaDetectorTiffHandlerGERM(HandlerBase):
    """The handler to read TIFF files produced by the GeRM ophyd detector."""

    specs = {"AD_TIFF_GERM"}

    def __init__(self, fpath):
        self._path = str(Path(fpath) / "")

    def __call__(self):
        ret = []
        with tifffile.TiffFile(self._path) as tif:
            ret.append(tif.asarray())
        return np.array(ret)


class AreaDetectorHDF5HandlerGERM(HandlerBase):
    """The handler to read HDF5 files produced by the GeRM ophyd detector."""

    specs = {"AD_HDF5_GERM"}

    def __init__(self, filename):
        self._name = filename

    def __call__(self, frame):
        with h5py.File(self._name, "r") as f:
            entry = f["/entry/data/data"]
            return entry[frame, :]
