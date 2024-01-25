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
from enum import Enum
from pathlib import Path

import h5py
import numpy as np
from caproto import ChannelType
from caproto.asyncio.client import Context
from caproto.server import PVGroup, ioc_arg_parser, pvproperty, run

from ..germ.ophyd import GeRMMiniClassForCaprotoIOC

internal_process = contextvars.ContextVar("internal_process", default=False)


class AcqStatuses(Enum):
    idle = "Done"
    acquiring = "Count"


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

    frame_num = pvproperty(value=0, doc="Frame counter")
    frame_shape = pvproperty(value=(0, 0), doc="Frame shape")

    async def _callback(self, pv, response, prop_name=None):
        # Update our own value based on the monitored one:
        await getattr(self, prop_name).write(
            response.data,
            # We can even make the timestamp the same:
            timestamp=response.metadata.timestamp,
        )

    async def _add_subscription(self, prop_name):
        self.client_context = Context()

        pvname = getattr(self.ophyd_det, prop_name).pvname
        (pvobject,) = await self.client_context.get_pvs(pvname)

        # Subscribe to the target PV and register a customized self._callback.
        self.subscriptions[prop_name] = pvobject.subscribe()
        self.subscriptions[prop_name].add_callback(
            functools.partial(self._callback, prop_name=prop_name)
        )

    count = pvproperty(
        value=AcqStatuses.idle.value,
        enum_strings=[x.value for x in AcqStatuses],
        dtype=ChannelType.ENUM,
        doc="Trigger the detector via a mirrored PV and save the data",
    )

    @count.startup
    async def count(self, instance, async_lib):
        await self._add_subscription("count")

    mca = pvproperty(value=0, doc="Mirrored mca PV")

    @mca.startup
    async def mca(self, instance, async_lib):
        await self._add_subscription("mca")

    number_of_channels = pvproperty(value=0, doc="Mirrored number_of_channels PV")

    @number_of_channels.startup
    async def number_of_channels(self, instance, async_lib):
        await self._add_subscription("number_of_channels")

    energy = pvproperty(value=0, doc="Mirrored energy PV")

    @energy.startup
    async def energy(self, instance, async_lib):
        await self._add_subscription("energy")

    def __init__(self, ophyd_det, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subscriptions = {}

        self.ophyd_det = ophyd_det
        self._data_file = None
        self._h5file_desc = None
        self._dataset = None

    @frame_shape.startup
    async def frame_shape(self, instance, async_lib):
        await self.frame_shape.write(
            (self.number_of_channels.value, len(self.energy.value))
        )

    def _get_current_image(self):
        """The function to return a current image from detector's MCA."""
        raw_data = self.mca.value
        data = np.reshape(raw_data, self.frame_shape.value)
        return data

    @stage.putter
    async def stage(self, instance, value):
        """The stage method to perform preparation of a dataset to save the data."""
        if instance.value in [True, "staged"] and value == "staged":
            msg = "The device is already staged. Unstage it first."
            raise ValueError(msg)

        if value == "staged":
            await self.frame_num.write(0)
            date = datetime.datetime.now()
            root_dir = self.write_dir.value
            assets_dir = date.strftime("%Y/%m/%d")
            full_path = Path(root_dir) / Path(assets_dir)
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
            self._data_file = str(full_path / f"{self.file_name_prefix.value}.h5")

            self._h5file_desc = h5py.File(self._data_file, "x", libver="latest")
            group = self._h5file_desc.create_group("/entry")
            self._dataset = group.create_dataset(
                "data/data",
                data=np.full(fill_value=np.nan, shape=(1, *self._frame_shape)),
                maxshape=(None, *self._frame_shape),
                chunks=(1, *self._frame_shape),
                dtype="float32",
            )
            self._h5file_desc.swmr_mode = True
            return True

        if value == "unstaged":
            self._h5file_desc.close()
            return False

    @count.putter
    @no_reentry
    async def count(self, instance, value):
        """The count method to perform an individual count of the detector."""
        if value != AcqStatuses.acquiring.value:
            return 0

        if (
            instance.value in [True, AcqStatuses.acquiring.value]
            and value == AcqStatuses.acquiring.value
        ):
            print(
                f"The device is already acquiring. Please wait until the '{AcqStatuses.idle.value}' status."
            )
            return 1

        while True:
            if instance.value != AcqStatuses.idle.value:
                await asyncio.sleep(0.1)
                continue

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
    germ_mini = GeRMMiniClassForCaprotoIOC(PV_PREFIX_CURLED, name="germ_mini")
    ioc = GeRMSaveIOC(ophyd_det=germ_mini, **ioc_options)
    run(ioc.pvdb, **run_options)
