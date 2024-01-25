from __future__ import annotations

import datetime
import itertools
import uuid
from collections import deque
from pathlib import Path

import h5py
import numpy as np
from event_model import compose_resource
from ophyd import Component as Cpt
from ophyd import Device, EpicsSignal, Kind, Signal
from ophyd.sim import new_uid
from ophyd.status import SubscriptionStatus
from PIL import Image


class ExternalFileReference(Signal):
    """
    A pure software Signal that describe()s an image in an external file.
    """

    def describe(self):
        resource_document_data = super().describe()
        resource_document_data[self.name].update(
            {
                "external": "FILESTORE:",
                "dtype": "array",
            }
        )
        return resource_document_data


class GeRMMiniClassForCaprotoIOC(Device):
    count = Cpt(EpicsSignal, ".CNT", kind=Kind.omitted, string=True)
    mca = Cpt(EpicsSignal, ".MCA", kind=Kind.omitted)
    number_of_channels = Cpt(EpicsSignal, ".NELM", kind=Kind.config)
    energy = Cpt(EpicsSignal, ".SPCTX", kind=Kind.omitted)


class GeRMDetectorBase(Device):
    """The base ophyd class for GeRM detector."""

    count = Cpt(EpicsSignal, ".CNT", kind=Kind.omitted, string=True)
    mca = Cpt(EpicsSignal, ".MCA", kind=Kind.omitted)
    number_of_channels = Cpt(EpicsSignal, ".NELM", kind=Kind.config)
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
    energy = Cpt(EpicsSignal, ".SPCTX", kind=Kind.omitted)

    image = Cpt(ExternalFileReference, kind=Kind.normal)

    def __init__(self, *args, root_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        if root_dir is None:
            msg = "The 'root_dir' kwarg cannot be None"
            raise RuntimeError(msg)
        self._root_dir = root_dir
        self._resource_document, self._datum_factory = None, None
        self._asset_docs_cache = deque()

        height = int(self.number_of_channels.get())
        width = len(self.energy.get())
        self.frame_shape = (height, width)

    def collect_asset_docs(self):
        """The method to collect resource/datum documents."""
        items = list(self._asset_docs_cache)
        self._asset_docs_cache.clear()
        yield from items

    def unstage(self):
        super().unstage()
        self._resource_document = None
        self._datum_factory = None

    def get_current_image(self):
        """The function to return a current image from detector's MCA."""
        # This is the reshaping we want
        # This doesn't trigger the detector
        raw_data = self.mca.get()
        data = np.reshape(raw_data, self.frame_shape)
        return data


def done_callback(value, old_value, **kwargs):
    """The callback function used by ophyd's SubscriptionStatus."""
    if not kwargs:
        pass
    if old_value == "Count" and value == "Done":
        return True
    return False


class GeRMDetectorTiff(GeRMDetectorBase):
    """The ophyd class for GeRM detector producing TIFF files."""

    def trigger(self):
        "The trigger method to acquire a single frame."

        status = SubscriptionStatus(self.count, run=False, callback=done_callback)

        self.count.put("Count")
        status.wait()

        # Read the image array:
        img = self.get_current_image()

        # Save TIFF files:
        date = datetime.datetime.now()
        assets_dir = date.strftime("%Y/%m/%d")
        file_name = f"{uuid.uuid4()}.tiff"
        file_path = str(Path(self._root_dir) / Path(assets_dir) / Path(file_name))
        self.save_image(file_path=file_path, mat=img)

        # Create resource/datum docs:
        self._resource_document, self._datum_factory, _ = compose_resource(
            start={"uid": "needed for compose_resource() but will be discarded"},
            spec="AD_TIFF_GERM",  # TODO: convert to use standard AD_TIFF.
            root=self._root_dir,
            resource_path=str(Path(assets_dir) / Path(file_name)),
            resource_kwargs={},
        )

        self._resource_document.pop("run_start")
        self._asset_docs_cache.append(("resource", self._resource_document))

        datum_document = self._datum_factory(datum_kwargs={})
        self._asset_docs_cache.append(("datum", datum_document))

        self.image.put(datum_document["datum_id"])

        return status

    def save_image(self, file_path, mat, overwrite=True):
        image = Image.fromarray(mat)
        if not overwrite:
            file_path = self._make_file_name(file_path)
            image.save(file_path)
        return file_path

    def _make_file_name(self, file_path):
        file_path = Path(file_path)
        file_base = file_path.parent / file_path.stem
        file_ext = file_path.suffix
        if file_path.is_file():
            nfile = 0
            check = True
            while check:
                name_add = f"{nfile:04d}"
                file_path = Path(f"{file_base}_{name_add}{file_ext}")
                if file_path.is_file():
                    nfile += nfile
                else:
                    check = False
        return str(file_path)


class GeRMDetectorHDF5(GeRMDetectorBase):
    def stage(self):
        super().stage()

        # Clear asset docs cache which may have some documents from the previous failed run.
        self._asset_docs_cache.clear()

        date = datetime.datetime.now()
        self._assets_dir = date.strftime("%Y/%m/%d")
        data_file = f"{new_uid()}.h5"

        self._resource_document, self._datum_factory, _ = compose_resource(
            start={"uid": "needed for compose_resource() but will be discarded"},
            spec="AD_HDF5_GERM",
            root=self._root_dir,
            resource_path=str(Path(self._assets_dir) / Path(data_file)),
            resource_kwargs={},
        )

        self._data_file = str(
            Path(self._resource_document["root"])
            / Path(self._resource_document["resource_path"])
        )

        # now discard the start uid, a real one will be added later
        self._resource_document.pop("run_start")
        self._asset_docs_cache.append(("resource", self._resource_document))

        # TODO: replace this logic with the caproto IOC calls:
        # - .write_dir <...>
        # - .file_name_prefix <uuid4>
        # - .stage staged
        self._h5file_desc = h5py.File(self._data_file, "x")
        group = self._h5file_desc.create_group("/entry")
        self._dataset = group.create_dataset(
            "data/data",
            data=np.full(fill_value=np.nan, shape=(1, *self._frame_shape)),
            maxshape=(None, *self._frame_shape),
            chunks=(1, *self._frame_shape),
            dtype="float32",
        )
        self._counter = itertools.count()

    def describe(self):
        res = super().describe()
        res[self.image.name].update({"shape": self._frame_shape})
        return res

    def trigger(self):
        def is_done(value, old_value, **kwargs):
            if old_value == "Count" and value == "Done":
                data = self.get_current_image()

                self._dataset.resize((self.current_frame + 1, *self._frame_shape))
                self._dataset[self.current_frame, :, :] = data

                return True

            return False

        status = SubscriptionStatus(self.count, run=False, callback=is_done)

        # Reuse the counter from the caproto IOC
        self.current_frame = next(self._counter)
        self.count.put("Count")

        datum_document = self._datum_factory(datum_kwargs={"frame": self.current_frame})
        self._asset_docs_cache.append(("datum", datum_document))

        self.image.put(datum_document["datum_id"])

        return status

    def unstage(self):
        super().unstage()
        self._h5file_desc.close()
