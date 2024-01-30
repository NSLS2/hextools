from __future__ import annotations

import pytest


@pytest.mark.hardware()
def test_germ_ops(germ_det_hdf5):
    germ_det_hdf5.summary()
    germ_det_hdf5.read()
