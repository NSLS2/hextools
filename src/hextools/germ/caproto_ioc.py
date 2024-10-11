"""
To run as:

EPICS_CAS_AUTO_BEACON_ADDR_LIST=no EPICS_CAS_BEACON_ADDR_LIST=${EPICS_CA_ADDR_LIST} python -m hextools.germ.caproto_ioc --list-pvs --prefix='XF:27ID1-ES{{GeRM-Det:1}}:'
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import os
import textwrap
import threading
import time as ttime
import uuid
from pathlib import Path

import numpy as np
from caproto import ChannelType
from caproto.asyncio.client import Context
from caproto.ioc_examples.setpoint_rbv_pair import pvproperty_with_rbv
from caproto.server import PVGroup, pvproperty, run, template_arg_parser

from ..utils import now
from . import AcqStatuses, StageStates, YesNo
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
        value="/tmp/",
        doc="The directory to write data to",
        string_encoding="utf-8",
        dtype=ChannelType.CHAR,
        max_length=255,
    )

    @write_dir.putter
    async def write_dir(self, instance, value):
        # pylint: disable=unused-argument
        """Put behavior of write_dir.

        In this callback, we check if the target directory exists, and if not - attempt to create it.
        If it exists, we check if we have write access to it.
        Finally, we add a trailing slash to the directory name and update the PV with it.
        """
        path = Path(value)
        if not path.exists():
            print(f"Path '{path}' does not exist. Creating one.")
            try:
                path.mkdir(mode=0o750, exist_ok=True)
                dir_exists = YesNo.YES.value
            except Exception as e:  # pylint: disable=broad-exception-caught
                dir_exists = YesNo.NO.value
                print(f"Failed to create directory {path}: {e}")
        else:
            if not os.access(path, os.W_OK):
                dir_exists = YesNo.NO.value
            dir_exists = YesNo.YES.value
        await self.directory_exists.write(dir_exists)

        return f"{Path(value)}/"

    directory_exists = pvproperty(
        value=YesNo.NO.value,
        enum_strings=[x.value for x in YesNo],
        dtype=ChannelType.ENUM,
        doc="The PV to indicate if the write_dir exists",
        read_only=True,
    )

    file_name = pvproperty(
        value="test.h5",
        doc="The file name of the file to write to",
        string_encoding="utf-8",
        dtype=ChannelType.CHAR,
        max_length=255,
    )

    @file_name.putter
    async def file_name(self, instance, value):
        # pylint: disable=unused-argument
        """Put behavior of file_name.

        In this callback, we check the suffix of the supplied file name to be in the list of supported extensions.
        If there is no suffix found, the default .h5 is used.
        """
        fname = Path(value)
        suffix = fname.suffix
        supported_suffixes = [".h5", ".hdf", ".hdf5", ".nxs"]
        if suffix not in supported_suffixes:
            if suffix == "":
                return str(fname.with_suffix(".h5"))
            msg = f"File name extension '{suffix}' not supported.\nSupported extensions: {supported_suffixes}"
            raise OSError(msg)
        return value

    # file_template = pvproperty(
    #     value="%s%s_%6.6d.h5",
    #     doc="The file path template",
    #     string_encoding="utf-8",
    #     report_as_string=True,
    #     max_length=255,
    # )

    data_file = pvproperty(
        value="",
        doc="Full path to the data file",
        dtype=str,
        read_only=True,
        max_length=255,
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
    count = pvproperty_with_rbv(
        value=AcqStatuses.IDLE.value,
        enum_strings=[x.value for x in AcqStatuses],
        dtype=ChannelType.ENUM,
        doc="Trigger the detector via a mirrored PV and save the data",
    )

    async def callback_count(self, pv, response):
        """A callback method for the 'count' PV."""
        # pylint: disable=unused-argument
        await self.count.readback.write(response.data)

    @count.setpoint.startup
    async def count(obj, instance, async_lib):
        # pylint: disable=[function-redefined, no-self-argument, unused-argument, protected-access]
        """Startup behavior of count."""
        await obj.parent._add_subscription("count")

    ### MCA ###
    # MCA reported by the GeRM libCA IOC has type 'DBF_LONG' (np.int64 or '<i8').
    mca = pvproperty(
        value=0, doc="Mirrored mca PV", max_length=192 * 4096, dtype=int, read_only=True
    )

    async def callback_mca(self, pv, response):
        """A callback method for the 'mca' PV.

        The mca is returned as a 1-d array of size 192 * 4096 = 786432, where
        192 is the number of channels (see 'number_of_channels' pvproperty) and
        4096 is the number of elements in the 'energy' pvproperty.
        """
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

    async def _update_frame_shape(self):
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

    @frame_shape.getter
    async def frame_shape(self, instance):
        """Calculate the frame shape."""
        # pylint: disable=unused-argument
        await self._update_frame_shape()

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
            write_dir = self.write_dir.value
            file_name = self.file_name.value

            await self.data_file.write(str(Path(write_dir) / file_name))

            await self._update_frame_shape()

            return True

        # TODO: Figure out how to do clean up on unstage without breaking the next scan:
        # await self.data_file.write("")

        return False

    @count.setpoint.putter
    @no_reentry
    async def count(obj, instance, value):
        """The count method to perform an individual count of the detector."""
        # pylint: disable=[function-redefined, no-self-argument, protected-access]
        if (
            value != AcqStatuses.ACQUIRING.value
            or obj.parent.directory_exists.value == YesNo.NO.value
        ):
            return 0

        if value == AcqStatuses.ACQUIRING.value and instance.value in [
            True,
            AcqStatuses.ACQUIRING.value,
        ]:
            print(
                f"The device is already acquiring. Please wait until the '{AcqStatuses.IDLE.value}' status."
            )
            return 1

        self = obj.parent

        num_acq_statuses = {val.value: idx for idx, val in enumerate(list(AcqStatuses))}
        external_count_pv = self.subscriptions["count"].pv

        # Note: updating the setpoint is needed to reflect the state change for
        # camonitor, etc.
        await self.count.setpoint.write(value)

        # Note: while the value is set successfully on the libCA IOC, it does
        # not confirm the writing was done, so the external_count_pv.write(...)
        # was failing. We do not wait for confirmation here.
        await external_count_pv.write(num_acq_statuses[value], wait=False)

        while True:
            count_value = await external_count_pv.read()
            if (
                count_value.data[0] != num_acq_statuses[AcqStatuses.IDLE.value]
            ):  # 1=Count, 0=Done
                await asyncio.sleep(self._update_period)
            else:
                break

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

        return 0

    @staticmethod
    def saver(request_queue, response_queue):
        """The saver callback for threading-based queueing."""
        while True:
            received = request_queue.get()
            filename = received["filename"]
            data = received["data"]
            try:
                dataset_shape = save_hdf5(fname=filename, data=data, mode="a")
                print(
                    f"{now()}: saved {data.shape} data into:\n  {filename}\n"
                    f"  Dataset shape in the file: {dataset_shape}"
                )
                success = True
                error_message = ""
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # The GeRM detector happens to response twice for a single
                # ".CNT" put, so capture an attempt to save the file with the
                # same name here and do nothing.
                success = False
                error_message = exc
                print(
                    f"{now()}: Cannot save file {filename!r} due to the following exception:\n{exc}"
                )

            response = {"success": success, "error_message": error_message}
            response_queue.put(response)


if __name__ == "__main__":
    from hextools.utils import replace_curlies

    parser, split_args = template_arg_parser(
        default_prefix="", desc=textwrap.dedent(GeRMSaveIOC.__doc__)
    )

    parsed_args = parser.parse_args()
    prefix = parsed_args.prefix
    if not prefix:
        parser.error("The 'prefix' argument must be specified.")
    # Note: for an ophyd object to instantiate, we need a single pairs of curly
    # braces, so replacing 4 pair with one.
    pv_prefix_ophyd = replace_curlies(prefix, how_many=4)
    # Remove the trailing ':'
    if pv_prefix_ophyd[-1] == ":":
        pv_prefix_ophyd = pv_prefix_ophyd[:-1]

    # Note: we need to escape curly braces twice ({.} -> {{.}} -> {{{{.}}}}),
    # meaning we should have four pairs in the args for the PV prefix. The
    # escaping is needed because .format(...) is called twice due to the
    # pvproperty_with_rbv is used.
    ioc_options, run_options = split_args(parsed_args)

    det = GeRMMiniClassForCaprotoIOC(pv_prefix_ophyd, name="det")

    ioc = GeRMSaveIOC(ophyd_det=det, **ioc_options)
    pvdb = {}
    # Note: for a caproto IOC to instantiate, we need a double pair of curly
    # braces, so replacing 4 pairs with 2.
    for k, v in ioc.pvdb.items():
        pvdb[replace_curlies(k)] = v
    run(pvdb, **run_options)
