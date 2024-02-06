from __future__ import annotations

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import numpy as np
import pytest


@pytest.mark.hardware()
def test_germ_ops(germ_det):
    germ_det.summary()
    germ_det.read()


@pytest.mark.hardware()
@pytest.mark.parametrize("count_time", [0.1, 1, 2, 5, 10, 15, 20, 30, 45, 60, 120])
def test_germ_with_bluesky(RE, db, germ_det, count_time):
    RE(bps.mv(germ_det.count_time, count_time))
    (uid,) = RE(bp.count([germ_det], num=1))

    hdr = db[uid]
    tbl = hdr.table(fill=True)
    print(tbl.T)
    data = np.array(list(hdr.data("GeRM_image")))
    assert data.shape == (1, 192, 4096)
    assert not np.isnan(data[0, 0, 0])
