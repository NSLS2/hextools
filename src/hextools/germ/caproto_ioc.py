"""
To run as:

EPICS_CAS_AUTO_BEACON_ADDR_LIST=no EPICS_CAS_BEACON_ADDR_LIST=${EPICS_CA_ADDR_LIST} python -m hextools.germ.caproto_ioc --list-pvs --prefix='XF:27ID1-ES{{GeRM-Det:1}}:'
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime
import functools
import textwrap
import threading
import time as ttime
import uuid
from pathlib import Path

import numpy as np
from caproto import ChannelType
from caproto.asyncio.client import Context
from caproto.server import PVGroup, pvproperty, run, template_arg_parser

from ..utils import now
from . import AcqStatuses, StageStates
from .export import save_hdf5
from .ophyd import GeRMMiniClassForCaprotoIOC

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
    """An IOC to write GeRM detector data to an HDF5 file."""

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
    data_file = pvproperty(
        value="", doc="Full path to the data file", dtype=str, read_only=True
    )
    stage = pvproperty(
        value=StageStates.UNSTAGED.value,
        enum_strings=[x.value for x in StageStates],
        dtype=ChannelType.ENUM,
        doc="Stage/unstage the detector",
    )

    frame_num = pvproperty(value=0, doc="Frame counter", dtype=int)
    frame_shape = pvproperty(
        value=(0, 0), doc="Frame shape", max_length=2, dtype=int, read_only=True
    )

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
            # timestamp=response.metadata.timestamp,  # Note: we use caproto IOC timestamp for this PV.
        )

    @count.startup
    async def count(self, instance, async_lib):
        """Startup behavior of count."""
        # pylint: disable=unused-argument
        await self._add_subscription("count")

    ### MCA ###
    mca = pvproperty(value=0, doc="Mirrored mca PV", max_length=786432, read_only=True)

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
    number_of_channels = pvproperty(
        value=0, doc="Mirrored number_of_channels PV", dtype=int, read_only=True
    )

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
    energy = pvproperty(
        value=0, doc="Mirrored energy PV", max_length=4096, read_only=True
    )

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

    queue = pvproperty(value=0, doc="A PV to facilitate threading-based queue")

    @queue.startup
    async def queue(self, instance, async_lib):
        """The startup behavior of the count property to set up threading queues."""
        # pylint: disable=unused-argument
        self._request_queue = async_lib.ThreadsafeQueue()
        self._response_queue = async_lib.ThreadsafeQueue()

        # Start a separate thread that consumes requests and sends responses.
        thread = threading.Thread(
            target=self.saver,
            daemon=True,
            kwargs={
                "request_queue": self._request_queue,
                "response_queue": self._response_queue,
            },
        )
        thread.start()

    def __init__(self, ophyd_det, *args, update_rate=10.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.subscriptions = {}
        self.client_context = None

        self.ophyd_det = ophyd_det
        self._update_rate = update_rate
        self._update_period = 1.0 / update_rate

        # Threading-based queues attributes:
        self._request_queue = None
        self._response_queue = None

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
            (int(self.number_of_channels.value), len(self.energy.value))
        )

    def _get_current_image(self):
        """The function to return a current image from detector's MCA."""
        raw_data = self.mca.value
        return np.reshape(raw_data, self.frame_shape.value)

    @stage.putter
    async def stage(self, instance, value):
        """The stage method to perform preparation of a dataset to save the data."""
        if (
            instance.value in [True, StageStates.STAGED.value]
            and value == StageStates.STAGED.value
        ):
            msg = "The device is already staged. Unstage it first."
            raise ValueError(msg)

        if value == StageStates.STAGED.value:
            await self.frame_num.write(0)
            date = datetime.datetime.now()
            root_dir = self.write_dir.value
            assets_dir = date.strftime("%Y/%m/%d")
            full_path = Path(root_dir) / Path(assets_dir)
            if not full_path.exists():
                # full_path.mkdir(parents=True, exist_ok=True)
                msg = f"Path '{full_path}' does not exist."
                raise OSError(msg)

            await self.data_file.write(
                str(full_path / f"{self.file_name_prefix.value}.h5")
            )

            return True

        # TODO: Figure out how to do clean up on unstage without breaking the next scan:
        # await self.data_file.write("")

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
            # TODO: figure out why the subscription does not update the value:
            count_value = await self.subscriptions["count"].pv.read()
            if count_value.data[0] != 0:  # 1=Count, 0=Done
                await asyncio.sleep(self._update_period)
                continue

            # The count is done at this point.
            # Delegate saving the resulting data to a blocking callback in a thread.
            payload = {
                "filename": self.data_file.value,
                "data": self._get_current_image(),
                "uid": str(uuid.uuid4()),
                "timestamp": ttime.time(),
            }

            await self._request_queue.async_put(payload)
            response = await self._response_queue.async_get()

            if response["success"]:
                # Increment the counter only on a successful saving of the file.
                await self.frame_num.write(self.frame_num.value + 1)

            break

        return 0

    @staticmethod
    def saver(request_queue, response_queue):
        """The saver callback for threading-based queueing."""
        while True:
            received = request_queue.get()
            filename = received["filename"]
            data = received["data"]
            try:
                save_hdf5(fname=filename, data=data)
                print(f"{now()}: saved {data.shape} data into:\n  {filename}")

                success = True
                error_message = ""
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # The GeRM detector happens to response twice for a single
                # ".CNT" put, so capture an attempt to save the file with the
                # same name here and do nothing.
                success = False
                error_message = exc
                print(
                    f"Cannot save file {filename!r} due to the following exception:\n{exc}"
                )

            response = {"success": success, "error_message": error_message}
            response_queue.put(response)


if __name__ == "__main__":
    parser, split_args = template_arg_parser(
        default_prefix="", desc=textwrap.dedent(GeRMSaveIOC.__doc__)
    )

    parsed_args = parser.parse_args()
    prefix = parsed_args.prefix
    if not prefix:
        parser.error("The 'prefix' argument must be specified.")
    pv_prefix_ophyd = prefix.replace("{{", "{").replace("}}", "}")
    # Remove the trailing ':'
    if pv_prefix_ophyd[-1] == ":":
        pv_prefix_ophyd = pv_prefix_ophyd[:-1]

    ioc_options, run_options = split_args(parsed_args)

    det = GeRMMiniClassForCaprotoIOC(pv_prefix_ophyd, name="det")

    ioc = GeRMSaveIOC(ophyd_det=det, **ioc_options)
    run(ioc.pvdb, **run_options)
