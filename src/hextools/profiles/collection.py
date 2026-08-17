"""Bluesky profile for the 27-ID-1 HEX beamline at NSLS-II."""

import os

from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.run_engine import (
    RunEngine,
    autoawait_in_bluesky_event_loop,
)
from bluesky.utils import ProgressBarManager
from IPython.core.getipython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from nslsii.ophyd_async.providers import NSLS2PathProvider
from ophyd_async.epics.adcore import ADWriterFactory
from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.epics.advimba import VimbaDetector
from ophyd_async.fastcs.panda import HDFPanda
from tiled.client import from_uri, simple

from hextools.detectors.phantom import PhantomDetector
from hextools.photon_delivery_system import (
    Filter,
    FilterPosition,
    FilterSetting,
    DCLM,
    Shutter,
)
from hextools.utils import (
    ProposalIDPrompt,
    auto_init_devices,
    initialize_run_engine,
    is_running_in_ci,
)

os.environ["REDIS_HOST"] = "xf27id1-hex-redis1.nsls2.bnl.gov"
os.environ["OPHYD_ASYNC_PRESERVE_DETECTOR_STATE"] = "YES"

RE: RunEngine = initialize_run_engine()
RE.md["facility"] = "NSLS-II"
RE.md["group"] = "HEX"
RE.md["beamline_id"] = "27-ID-1"

# Setup progress bars
RE.waiting_hook = ProgressBarManager()  # type: ignore[assignment]

if not is_running_in_ci():
    tiled_writing_client = from_uri(
        "https://tiled.nsls2.bnl.gov",
        api_key=os.environ.get("TILED_BLUESKY_WRITING_API_KEY_HEX", ""),
    )["hex"]["raw"]
else:
    tiled_writing_client = tiled_reading_client = c = simple()

bec = BestEffortCallback()
RE.subscribe(bec)

ipython = get_ipython()
if ipython is not None and isinstance(ipython, TerminalInteractiveShell):
    ipython.prompts = ProposalIDPrompt(RE, ipython)
    autoawait_in_bluesky_event_loop()
    if not is_running_in_ci():
        tiled_reading_client = c = from_uri("https://tiled.nsls2.bnl.gov")["hex"]["raw"]

# Define our global path provider for the beamline
path_provider = NSLS2PathProvider(RE.md)

with auto_init_devices(timeout=1.0):
    # Photon delivery system
    photon_shutter = Shutter("XF:27ID1A-OP:1{Shtr:1}", name="photon_shutter")
    dclm = DCLM("XF:27IDA-OP:1{Mono:DCLM-Ax:", name="dclm")

    filter1_upstream = Filter(
        motor_pv="XF:27IDA-OP:1{Fltr:1-Ax:Yu}Mtr",
        in_pos_switch_pv="XF:27IDA-OP:0{Fltr:1_US}Sw:InPos-Sts",
        positions={
            FilterPosition.UPPER_LIMIT: FilterSetting(68.0),
            FilterPosition.PASS_THROUGH: FilterSetting(66.6),
            FilterPosition.POS_1: FilterSetting(41.5, "12 mm SiC"),
            FilterPosition.POS_2: FilterSetting(6.5, "9 mm SiC"),
            FilterPosition.POS_3: FilterSetting(-28.5, "6 mm SiC"),
            FilterPosition.POS_4: FilterSetting(-58.0, "3 mm SiC"),
            FilterPosition.LOWER_LIMIT: FilterSetting(-63.2),
        },
    )

    filter1_downstream = Filter(
        motor_pv="XF:27IDA-OP:1{Fltr:1-Ax:Yd}Mtr",
        in_pos_switch_pv="XF:27IDA-OP:0{Fltr:1_DS}Sw:InPos-Sts",
        positions={
            FilterPosition.UPPER_LIMIT: FilterSetting(62.8),
            FilterPosition.PASS_THROUGH: FilterSetting(60.3),
            FilterPosition.POS_1: FilterSetting(35.3, "4 mm Cu"),
            FilterPosition.POS_2: FilterSetting(0.3, "2 mm Cu"),
            FilterPosition.POS_3: FilterSetting(-34.7, "24 mm SiC"),
            FilterPosition.POS_4: FilterSetting(-69.7, "12 mm SiC"),
            FilterPosition.LOWER_LIMIT: FilterSetting(-73.5),
        },
    )

    filter2 = Filter(
        motor_pv="XF:27IDA-OP:1{Fltr:2-Ax:Y}Mtr",
        in_pos_switch_pv="XF:27IDA-OP:0{Fltr:2}Sw:InPos-Sts",
        positions={
            FilterPosition.UPPER_LIMIT: FilterSetting(72.2),
            FilterPosition.PASS_THROUGH: FilterSetting(70.2),
            FilterPosition.POS_1: FilterSetting(45.2, "2 mm Cu"),
            FilterPosition.POS_2: FilterSetting(10.2, "1.5 mm Cu"),
            FilterPosition.POS_3: FilterSetting(-24.8, "1 mm Cu"),
            FilterPosition.POS_4: FilterSetting(-58.0, "0.5 mm Cu"),
            FilterPosition.LOWER_LIMIT: FilterSetting(-61.2),
        },
    )

    filter3 = Filter(
        motor_pv="XF:27IDA-OP:1{Fltr:3-Ax:Y}Mtr",
        in_pos_switch_pv="XF:27IDA-OP:0{Fltr:3}Sw:InPos-Sts",
        positions={
            FilterPosition.UPPER_LIMIT: FilterSetting(66.6),
            FilterPosition.PASS_THROUGH: FilterSetting(66.0),
            FilterPosition.POS_1: FilterSetting(40.0),
            FilterPosition.POS_2: FilterSetting(5.0),
            FilterPosition.POS_3: FilterSetting(-30.0),
            FilterPosition.POS_4: FilterSetting(-63.0),
            FilterPosition.LOWER_LIMIT: FilterSetting(-64.9),
        },
    )

    filter4 = Filter(
        motor_pv="XF:27IDA-OP:1{Fltr:4-Ax:Y}Mtr",
        in_pos_switch_pv="XF:27IDA-OP:0{Fltr:4}Sw:InPos-Sts",
        positions={
            FilterPosition.UPPER_LIMIT: FilterSetting(71.8),
            FilterPosition.PASS_THROUGH: FilterSetting(70.0),
            FilterPosition.POS_1: FilterSetting(45.0),
            FilterPosition.POS_2: FilterSetting(10.0),
            FilterPosition.POS_3: FilterSetting(-25.0),
            FilterPosition.POS_4: FilterSetting(-57.0),
            FilterPosition.LOWER_LIMIT: FilterSetting(-59.8),
        },
    )

    panda1 = HDFPanda("XF:27ID1-ES{PANDA:1}", path_provider, name="panda1")

    kinetix1 = KinetixDetector(
        "XF:27ID1-ES{Kinetix:1}", ADWriterFactory.hdf(path_provider), name="kinetix1"
    )
    kinetix2 = KinetixDetector(
        "XF:27ID1-ES{Kinetix:1}", ADWriterFactory.hdf(path_provider), name="kinetix2"
    )
    kinetix3 = KinetixDetector(
        "XF:27ID1-ES{Kinetix:1}", ADWriterFactory.hdf(path_provider), name="kinetix3"
    )
    kinetix4 = KinetixDetector(
        "XF:27ID1-ES{Kinetix:1}", ADWriterFactory.hdf(path_provider), name="kinetix4"
    )

    phantom1 = PhantomDetector(
        "XF:27ID1-ES{Phantom:1}", ADWriterFactory.hdf(path_provider), name="phantom1"
    )

    sample_camera = VimbaDetector(
        "XF:27ID1-ES{Vimba:1}", ADWriterFactory.hdf(path_provider), name="sample_camera"
    )
    fs_camera = VimbaDetector(
        "XF:27ID1-ES{Vimba:2}",
        ADWriterFactory.hdf(path_provider),
        name="fluorescence_screen_camera",
    )

    # TODO: Perkin elmer.
