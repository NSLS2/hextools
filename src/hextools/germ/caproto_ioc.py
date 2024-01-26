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
from caproto.asyncio.client import Context
from caproto.server import PVGroup, ioc_arg_parser, pvproperty, run

from ..germ import AcqStatuses
from ..germ.ophyd import GeRMMiniClassForCaprotoIOC

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

    frame_num = pvproperty(value=0, doc="Frame counter")
    frame_shape = pvproperty(value=(0, 0), doc="Frame shape", max_length=2)

    async def _add_subscription(self, prop_name):
        client_context = Context()

        pvname = getattr(self.ophyd_det, prop_name).pvname
        (pvobject,) = await client_context.get_pvs(pvname)

        # Subscribe to the target PV and register a customized self._callback.
        self.subscriptions[prop_name] = pvobject.subscribe(data_type="time")
        self.subscriptions[prop_name].add_callback(
            getattr(self, f"callback_{prop_name}")
        )

    ### Count ###
    count = pvproperty(
        value=AcqStatuses.IDLE.value,
        enum_strings=[x.value for x in AcqStatuses],
        dtype=ChannelType.ENUM,
        doc="Trigger the detector via a mirrored PV and save the data",
    )

    async def callback_count(self, pv, response):
        """A callback method for the 'count' PV."""
        # pylint: disable=unused-argument
        await self.count.write(
            response.data,
            # We can even make the timestamp the same:
            timestamp=response.metadata.timestamp,
        )

    @count.startup
    async def count(self, instance, async_lib):
        """Startup behavior of count."""
        # pylint: disable=unused-argument
        await self._add_subscription("count")

    ### MCA ###
    mca = pvproperty(value=0, doc="Mirrored mca PV", max_length=786432)

    async def callback_mca(self, pv, response):
        """A callback method for the 'mca' PV."""
        # pylint: disable=unused-argument
        await self.mca.write(
            response.data,
            # We can even make the timestamp the same:
            timestamp=response.metadata.timestamp,
        )

    @mca.startup
    async def mca(self, instance, async_lib):
        """Startup behavior of mca."""
        # pylint: disable=unused-argument
        await self._add_subscription("mca")

    ### Number of channels ###
    number_of_channels = pvproperty(value=0, doc="Mirrored number_of_channels PV")

    async def callback_number_of_channels(self, pv, response):
        """A callback method for the 'number_of_channels' PV."""
        # pylint: disable=unused-argument
        await self.number_of_channels.write(
            response.data,
            # We can even make the timestamp the same:
            timestamp=response.metadata.timestamp,
        )

    @number_of_channels.startup
    async def number_of_channels(self, instance, async_lib):
        """Startup behavior of number_of_channels."""
        # pylint: disable=unused-argument
        await self._add_subscription("number_of_channels")

    ### Energy ###
    energy = pvproperty(value=0, doc="Mirrored energy PV", max_length=4096)

    async def callback_energy(self, pv, response):
        """A callback method for the 'energy' PV."""
        # pylint: disable=unused-argument
        await self.energy.write(
            response.data,
            # We can even make the timestamp the same:
            timestamp=response.metadata.timestamp,
        )

    @energy.startup
    async def energy(self, instance, async_lib):
        """Startup behavior of energy."""
        # pylint: disable=unused-argument
        await self._add_subscription("energy")

    def __init__(self, ophyd_det, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subscriptions = {}
        self.client_context = None

        self.ophyd_det = ophyd_det
        self._data_file = None
        self._h5file_desc = None
        self._dataset = None

    @frame_shape.getter
    async def frame_shape(self, instance):
        """Calculate the frame shape using the PVs from the real IOC.

        Note:
        -----
        It may be considered as bad practice (see
        https://caproto.github.io/caproto/v1.1.1/iocs.html#don-t-use-a-getter),
        but the .getter was the only way to get the shape updated after all
        subscriptions to the real IOC's PVs had been done, as it was not
        working on .startup as expected.
        """
        # pylint: disable=unused-argument
        await self.frame_shape.write(
            (self.number_of_channels.value, len(self.energy.value))
        )

    def _get_current_image(self):
        """The function to return a current image from detector's MCA."""
        raw_data = self.mca.value
        return np.reshape(raw_data, self.frame_shape.value)

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
                data=np.full(fill_value=np.nan, shape=(1, *self.frame_shape.value)),
                maxshape=(None, *self.frame_shape.value),
                chunks=(1, *self.frame_shape.value),
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
        if value != AcqStatuses.ACQUIRING.value:
            return 0

        if (
            instance.value in [True, AcqStatuses.ACQUIRING.value]
            and value == AcqStatuses.ACQUIRING.value
        ):
            print(
                f"The device is already acquiring. Please wait until the '{AcqStatuses.IDLE.value}' status."
            )
            return 1

        while True:
            if instance.value != AcqStatuses.IDLE.value:
                await asyncio.sleep(0.1)
                continue

            self._dataset.resize((self.frame_num.value + 1, *self.frame_shape.value))
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
