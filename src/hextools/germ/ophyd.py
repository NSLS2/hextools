from __future__ import annotations

import datetime
import logging
from collections import deque
from pathlib import Path
from pprint import pformat

import numpy as np
from event_model import StreamRange, compose_stream_resource
from ophyd import Component as Cpt
from ophyd import Device, EpicsSignal, Kind, Signal
from ophyd.sim import new_uid
from ophyd.status import SubscriptionStatus

from hextools.utils import get_ioc_hostname

from . import AcqStatuses, StageStates

logger = logging.getLogger(__name__)


class ExternalFileReference(Signal):
    """
    A pure software Signal that describe()s an image in an external file.
    """

    def describe(self):
        resource_document_data = super().describe()
        resource_document_data[self.name].update(
            {
                "external": "STREAM:",
                "dtype": "array",
            }
        )
        return resource_document_data


class GeRMMiniClassForCaprotoIOC(Device):
    """Minimal GeRM detector ophyd class used in caproto IOC."""

    count = Cpt(EpicsSignal, ".CNT", kind=Kind.omitted, string=True)
    mca = Cpt(EpicsSignal, ".MCA", kind=Kind.omitted)
    number_of_channels = Cpt(EpicsSignal, ".NELM", kind=Kind.config)
    energy = Cpt(EpicsSignal, ".SPCTX", kind=Kind.omitted)


class GeRMDetectorBase(GeRMMiniClassForCaprotoIOC):
    """The base ophyd class for GeRM detector."""

    gain = Cpt(EpicsSignal, ".GAIN", kind=Kind.config)
    shaping_time = Cpt(EpicsSignal, ".SHPT", kind=Kind.config)
    count_time = Cpt(EpicsSignal, ".TP", kind=Kind.config)
    auto_time = Cpt(EpicsSignal, ".TP1", kind=Kind.config)
    run_num = Cpt(EpicsSignal, ".RUNNO", kind=Kind.omitted)
    fast_data_filename = Cpt(EpicsSignal, ".FNAM", string=True, kind=Kind.config)
    operating_mode = Cpt(EpicsSignal, ".MODE", kind=Kind.omitted)
    single_auto_toggle = Cpt(EpicsSignal, ".CONT", kind=Kind.omitted)
    gmon = Cpt(EpicsSignal, ".GMON", kind=Kind.omitted)
    ip_addr = Cpt(EpicsSignal, ".IPADDR", string=True, kind=Kind.omitted)
    temp_1 = Cpt(EpicsSignal, ":Temp1", kind=Kind.omitted)
    temp_2 = Cpt(EpicsSignal, ":Temp2", kind=Kind.omitted)
    fpga_cpu_temp = Cpt(EpicsSignal, ":ztmp", kind=Kind.omitted)
    calibration_file = Cpt(EpicsSignal, ".CALF", kind=Kind.omitted)
    multi_file_supression = Cpt(EpicsSignal, ".MFS", kind=Kind.omitted)
    tdc = Cpt(EpicsSignal, ".TDC", kind=Kind.omitted)
    leakage_pulse = Cpt(EpicsSignal, ".LOAO", kind=Kind.omitted)
    internal_leak_curr = Cpt(EpicsSignal, ".EBLK", kind=Kind.omitted)
    pileup_rejection = Cpt(EpicsSignal, ".PUEN", kind=Kind.omitted)
    test_pulse_aplitude = Cpt(EpicsSignal, ".TPAMP", kind=Kind.omitted)
    channel = Cpt(EpicsSignal, ".MONCH", kind=Kind.omitted)
    tdc_slope = Cpt(EpicsSignal, ".TDS", kind=Kind.omitted)
    test_pulse_freq = Cpt(EpicsSignal, ".TPFRQ", kind=Kind.omitted)
    tdc_mode = Cpt(EpicsSignal, ".TDM", kind=Kind.omitted)
    test_pulce_enable = Cpt(EpicsSignal, ".TPENB", kind=Kind.omitted)
    test_pulse_count = Cpt(EpicsSignal, ".TPCNT", kind=Kind.omitted)
    input_polarity = Cpt(EpicsSignal, ".POL", kind=Kind.omitted)
    voltage = Cpt(EpicsSignal, ":HV_RBV", kind=Kind.config)
    current = Cpt(EpicsSignal, ":HV_CUR", kind=Kind.omitted)
    peltier_2 = Cpt(EpicsSignal, ":P2", kind=Kind.omitted)
    peliter_2_current = Cpt(EpicsSignal, ":P2_CUR", kind=Kind.omitted)
    peltier_1 = Cpt(EpicsSignal, ":P1", kind=Kind.omitted)
    peltier_1_current = Cpt(EpicsSignal, ":P1_CUR", kind=Kind.omitted)
    hv_bias = Cpt(EpicsSignal, ":HV", kind=Kind.config)
    ring_hi = Cpt(EpicsSignal, ":DRFTHI", kind=Kind.omitted)
    ring_lo = Cpt(EpicsSignal, ":DRFTLO", kind=Kind.omitted)
    channel_enabled = Cpt(EpicsSignal, ".TSEN", kind=Kind.omitted)

    image = Cpt(ExternalFileReference, kind=Kind.normal)

    # Caproto IOC components:
    write_dir = Cpt(
        EpicsSignal,
        ":write_dir",
        kind=Kind.config,
        string=True,
    )
    file_name = Cpt(
        EpicsSignal,
        ":file_name",
        kind=Kind.config,
        string=True,
    )
    frame_num = Cpt(EpicsSignal, ":frame_num", kind=Kind.omitted)
    frame_shape = Cpt(EpicsSignal, ":frame_shape", kind=Kind.omitted)
    ioc_stage = Cpt(EpicsSignal, ":stage", kind=Kind.omitted)
    count = Cpt(EpicsSignal, ":count", kind=Kind.omitted, string=True)

    def __init__(self, *args, root_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        if root_dir is None:
            msg = "The 'root_dir' kwarg cannot be None"
            raise RuntimeError(msg)
        self._root_dir = root_dir
        self._stream_resource_document, self._stream_datum_factory = None, None
        self._asset_docs_cache = deque()

    def collect_asset_docs(self):
        """The method to collect resource/datum documents."""
        items = list(self._asset_docs_cache)
        self._asset_docs_cache.clear()
        yield from items

    def unstage(self):
        super().unstage()
        self._stream_resource_document = None
        self._stream_datum_factory = None

    def get_current_image(self):
        """The function to return a current image from detector's MCA."""
        # This is the reshaping we want
        # This doesn't trigger the detector
        raw_data = self.mca.get()
        return np.reshape(raw_data, self.frame_shape)


def done_callback(value, old_value, **kwargs):
    """The callback function used by ophyd's SubscriptionStatus."""
    # pylint: disable=unused-argument
    if old_value == AcqStatuses.ACQUIRING.value and value == AcqStatuses.IDLE.value:
        return True
    return False


class GeRMDetectorHDF5(GeRMDetectorBase):
    """The ophyd class for GeRM detector producing HDF5 files."""

    def __init__(self, *args, date_template="%Y/%m/%d", **kwargs):
        super().__init__(*args, **kwargs)
        self._date_template = date_template

    def _generate_file_path(self):
        date = datetime.datetime.now()
        assets_dir = date.strftime(self._date_template)
        data_file = f"{new_uid()}.h5"

        return Path(self._root_dir) / Path(assets_dir) / Path(data_file)

    def stage(self):
        super().stage()

        # Clear asset docs cache which may have some documents from the previous failed run.
        self._asset_docs_cache.clear()

        full_path = self._generate_file_path()
        frame_shape = self.frame_shape.get()

        hostname = get_ioc_hostname(self.count.pvname)
        uri = f"file://{hostname}/{str(full_path).strip('/')}"

        (
            self._stream_resource_document,
            self._stream_datum_factory,
        ) = compose_stream_resource(
            mimetype="application/x-hdf5",
            uri=uri,
            data_key=self.image.name,
            parameters={
                "chunk_shape": (1, *frame_shape),
                "dataset": "/entry/data/data",
            },
        )

        logger.debug(
            "stream_resource_doc:\n %s", {pformat(self._stream_resource_document)}
        )

        # self._stream_resource_document.pop("run_start")
        self._asset_docs_cache.append(
            ("stream_resource", self._stream_resource_document)
        )

        # Update caproto IOC parameters:
        self.write_dir.put(str(full_path.parent))
        self.file_name.put(str(full_path.stem))
        self.ioc_stage.put(StageStates.STAGED.value)

    def describe(self):
        res = super().describe()
        res[self.image.name].update(
            {"shape": self.frame_shape.get().tolist(), "dtype_numpy": "<i8"}
        )
        return res

    def trigger(self):
        status = SubscriptionStatus(self.count, run=False, callback=done_callback)

        # Reuse the counter from the caproto IOC
        current_frame = self.frame_num.get()
        stream_datum_document = self._stream_datum_factory(
            StreamRange(start=current_frame, stop=current_frame + 1),
        )

        self.count.put(AcqStatuses.ACQUIRING.value)

        logger.debug("stream_datum_document:\n%s", pformat(stream_datum_document))

        self._asset_docs_cache.append(("stream_datum", stream_datum_document))

        return status

    def unstage(self):
        self.ioc_stage.put(StageStates.UNSTAGED.value)
        super().unstage()


class HEXGeRMDetectorHDF5(GeRMDetectorHDF5):
    """The HEX-specific ophyd class for GeRM detector producing HDF5 files."""

    def __init__(self, *args, md=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._md = md or {"data_session": "", "cycle": "", "scan_id": 1}

    def _generate_file_path(self):
        date = datetime.datetime.now()
        date_dir = date.strftime(self._date_template)
        data_file = f"{new_uid()}.h5"

        return (
            Path(self._root_dir)
            / Path(self._md["cycle"])
            / Path(self._md["data_session"])
            / "assets"
            / self.name
            / Path(date_dir)
            / Path(f"scan_{self._md['scan_id']:05d}")
            / Path(data_file)
        )
