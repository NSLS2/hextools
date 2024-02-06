from __future__ import annotations

import asyncio
import contextlib

import databroker
import pytest
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.run_engine import RunEngine
from databroker import Broker

from hextools.germ.ophyd import GeRMDetectorHDF5
from hextools.handlers import AreaDetectorHDF5HandlerGERM


@pytest.fixture()
def germ_det():
    return GeRMDetectorHDF5(
        "XF:27ID1-ES{GeRM-Det:1}", name="GeRM", root_dir="/nsls2/data/hex/assets/germ/"
    )


@pytest.fixture()
def db():
    """Return a data broker"""
    # db = Broker.named("temp")
    db = Broker.named("hex")
    with contextlib.suppress(Exception):
        databroker.assets.utils.install_sentinels(db.reg.config, version=1)

    db.reg.register_handler("AD_HDF5_GERM", AreaDetectorHDF5HandlerGERM, overwrite=True)
    return db


@pytest.fixture()
def RE(db):
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    RE = RunEngine({}, loop=loop)
    RE.subscribe(db.insert)

    bec = BestEffortCallback()
    RE.subscribe(bec)

    # from bluesky.utils import ts_msg_hook
    # RE.msg_hook = ts_msg_hook

    return RE
