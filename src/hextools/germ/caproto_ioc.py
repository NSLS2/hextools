"""
To run as:

EPICS_CAS_AUTO_BEACON_ADDR_LIST=no EPICS_CAS_BEACON_ADDR_LIST=<EPICS_SUBNET> python germ_ioc.py -v
"""
from __future__ import annotations

import asyncio
import contextvars
import datetime
import functools
import textwrap
from pathlib import Path

import h5py
import numpy as np
from caproto import ChannelType
from caproto.server import PVGroup, ioc_arg_parser, pvproperty, run
from ophyd.status import SubscriptionStatus

from ..germ.ophyd import GeRMDetectorHDF5

internal_process = contextvars.ContextVar("internal_process", default=False)


def no_reentry(func):
    """
    This is needed for put completion.
    """

    @functools.wraps(func)
    async def inner(*args, **kwargs):
        if internal_process.get():
            return None
        try:
            internal_process.set(True)
            return await func(*args, **kwargs)
        finally:
            internal_process.set(False)

    return inner


class GeRMSaveIOC(PVGroup):
    """
    When a PV is written to, write the new value into a file as a string.
    """

    write_dir = pvproperty(
        value="/tmp",
        doc="The directory to write data to",
        string_encoding="utf-8",
        report_as_string=True,
        max_length=255,
    )
    file_name_prefix = pvproperty(
        value="test",
        doc="The file name prefix of the file to write to",
        string_encoding="utf-8",
        report_as_string=True,
        max_length=255,
    )
    stage = pvproperty(
        value="unstaged",
        enum_strings=["unstaged", "staged"],
        dtype=ChannelType.ENUM,
        doc="Stage/unstage the detector. 0=unstaged, 1=staged",
    )
    count = pvproperty(
        value="idle",
        enum_strings=["idle", "acquiring"],
        dtype=ChannelType.ENUM,
        doc="Trigger the detector and save the data",
    )
    frame_num = pvproperty(value=0, doc="Frame counter")

    def __init__(self, ophyd_det, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ophyd_det = ophyd_det
        self._data_file = None
        self._h5file_desc = None

    @stage.putter
    async def stage(self, instance, value):
        if instance.value in [True, "staged"] and value == "staged":
            msg = "The device is already staged. Unstage it first."
            raise ValueError(msg)
        if value == "staged":
            date = datetime.datetime.now()
            root_dir = self.write_dir.value
            assets_dir = date.strftime("%Y/%m/%d")
            full_path = Path(root_dir) / Path(assets_dir)
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
            self._data_file = str(full_path / f"{self.file_name_prefix.value}.h5")

            self._h5file_desc = h5py.File(self._data_file, "x", libver="latest")
            group = self._h5file_desc.create_group("/entry")
            self._frame_shape = self.ophyd_det._frame_shape
            self._dataset = group.create_dataset(
                "data/data",
                data=np.full(fill_value=np.nan, shape=(1, *self._frame_shape)),
                maxshape=(None, *self._frame_shape),
                chunks=(1, *self._frame_shape),
                dtype="float32",
            )
            self._h5file_desc.swmr_mode = True
            return True

        elif value == "unstaged":
            self._h5file_desc.close()
            return False

    @count.putter
    @no_reentry
    async def count(self, instance, value):
        if value != "acquiring":
            return 0

        if instance.value in [True, "acquiring"] and value == "acquiring":
            print(
                "The device is already acquiring. Please wait until the 'idle' status."
            )
            return 1

        def is_done(value, old_value, **kwargs):
            if old_value == "Count" and value == "Done":
                return True
            return False

        status = SubscriptionStatus(self.ophyd_det.count, run=False, callback=is_done)
        self.ophyd_det.count.put("Count")

        while True:
            if not status.done:
                await instance.write(value)
                await asyncio.sleep(0.1)
                continue
            else:
                self._dataset.resize((self.frame_num.value + 1, *self._frame_shape))
                self._dataset[
                    self.frame_num.value, :, :
                ] = self.ophyd_det.get_current_image()
                self._dataset.flush()
                await self.frame_num.write(self.frame_num.value + 1)
                break

        return 0


if __name__ == "__main__":
    PV_PREFIX_CURLED = "XF:27ID1-ES{GeRM-Det:1}"
    PV_PREFIX = "XF:27ID1-ES:GeRM-Det:1:"
    ioc_options, run_options = ioc_arg_parser(
        default_prefix=PV_PREFIX, desc=textwrap.dedent(GeRMSaveIOC.__doc__)
    )
    ophyd_det = GeRMDetectorHDF5(PV_PREFIX_CURLED, name="GeRM")
    ioc = GeRMSaveIOC(ophyd_det=ophyd_det, **ioc_options)
    run(ioc.pvdb, **run_options)
