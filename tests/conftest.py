from __future__ import annotations

import pytest

from hextools.germ.ophyd import GeRMDetectorHDF5


@pytest.fixture()
def germ_det_hdf5():
    return GeRMDetectorHDF5(
        "XF:27ID1-ES{GeRM-Det:1}", name="GeRM", root_dir="/nsls2/data/hex/assets/germ/"
    )
