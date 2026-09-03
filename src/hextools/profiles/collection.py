"""Bluesky profile for the 27-ID-1 HEX beamline at NSLS-II."""

import os

# Remove PYEPICS_LIBCA set by the conda epics-base activation script.
# It points to conda's libca.so which conflicts with epicscorelibs' libca.so
# (used by aioca/ophyd-async), causing PV connections to fail and the process
# to hang on exit with "double free or corruption".
os.environ.pop("PYEPICS_LIBCA", None)

from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.run_engine import (
    RunEngine,
    autoawait_in_bluesky_event_loop,
)
from bluesky.utils import ProgressBarManager
from bluesky_tiled_plugins import TiledWriter
from IPython.core.getipython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from nslsii.ophyd_async.providers import NSLS2PathProvider
from ophyd_async.epics.adcore import ADWriterFactory, NDStatsIO, PluginSignalDataLogic
from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.epics.advimba import VimbaDetector
from ophyd_async.fastcs.panda import HDFPanda
from tiled.client import from_uri, simple
from bluesky.suspenders import SuspendFloor
from bluesky import plans as bp, plan_stubs as bps, preprocessors as bpp

from hextools.detectors.phantom import PhantomDetector
from hextools.machine import NSLS2StorageRing
from hextools.motors import DoubleObjCamera, OpticsTable, SampleTower, WideFOVCamera
from hextools.photon_delivery_system import (
    DCLM,
    Filter,
    FilterPosition,
    FilterSetting,
    Shutter,
    load_filters,
)
from hextools.utils import (
    ProposalIDPrompt,
    auto_init_devices,
    initialize_run_engine,
    is_running_in_ci,
    print_proposal_info,
    print_version_info
)

# Environment variables for Redis host and ophyd_async detector state preservation
os.environ["REDIS_HOST"] = "xf27id1-hex-redis1.nsls2.bnl.gov"
os.environ["OPHYD_ASYNC_PRESERVE_DETECTOR_STATE"] = "YES"

# Print version information for bluesky, ophyd_async, tiled, and hextools.
print_version_info()

# Setup the RunEngine and its metadata.
RE: RunEngine = initialize_run_engine()
RE.md["facility"] = "NSLS-II"
RE.md["group"] = "HEX"
RE.md["beamline_id"] = "27-ID-1"

# Setup progress bars
RE.waiting_hook = ProgressBarManager()  # type: ignore[assignment]

# Display active proposal information in the console.
print_proposal_info(RE.md)

# Get the IPython shell and set up the custom prompt to show the current proposal ID.
ipython = get_ipython()
if ipython is not None and isinstance(ipython, TerminalInteractiveShell):
    ipython.prompts = ProposalIDPrompt(RE, ipython)
    autoawait_in_bluesky_event_loop()

# Construct our tiled clients for writing and (in an interactive session) reading.
# If we're running in CI, we use a simple client for both.
if not is_running_in_ci():
    tiled_writing_client = from_uri(
        "https://tiled.nsls2.bnl.gov",
        api_key=os.environ.get("TILED_BLUESKY_WRITING_API_KEY_HEX", ""),
    )["hex"]["raw"]
    if ipython is not None and isinstance(ipython, TerminalInteractiveShell):
        tiled_reading_client = c = from_uri("https://tiled.nsls2.bnl.gov")["hex"]["raw"]
else:
    tiled_writing_client = tiled_reading_client = c = simple()


# Subscribe the tiled writer to the RunEngine
RE.subscribe(TiledWriter(tiled_writing_client))

# Subscribe the best effort callback
bec = BestEffortCallback()
RE.subscribe(bec)

# Define our global default path provider for the beamline
path_provider = NSLS2PathProvider(RE.md)

with auto_init_devices(timeout=1.0):
    # Shutters (Front-end and photon)
    fe_shutter = Shutter("XF:27IDA-PPS{Sh:FE}", name="front_end_shutter")
    photon_shutter = Shutter("XF:27IDA-PPS{L1-S1}", name="photon_shutter")

    # Storage ring information
    storage_ring = NSLS2StorageRing()

    # Monochromator DCLM (Double Crystal Laue Monochromator)
    dclm = DCLM("XF:27IDA-OP:1{Mono:DCLM-Ax:", name="dclm")

    # Motors for the optics table
    optics_table = OpticsTable("XF:27ID1A-OP:1{OPT:1-Ax:", name="optics_table")

    # Sample tower
    sample_tower = SampleTower("XF:27ID1A-OP:1{SMPL:1-Ax:", name="sample_tower")

    # Generate filter objects from the configuration file
    filters: list[Filter] = load_filters()

    # Add filters directly to namespace for convenience in interactive sessions
    if ipython is not None:
        for filter in filters:
            ipython.user_ns[filter.name] = filter


    # PandABox
    panda1 = HDFPanda("XF:27ID1-ES{PANDA:1}", path_provider, name="panda1")

    # Kinetix and Phantom detectors
    kinetix1 = KinetixDetector(
        "XF:27ID1-BI{Kinetix-Det:1}", ADWriterFactory.hdf(path_provider), name="kinetix1"
    )
    kinetix2 = KinetixDetector(
        "XF:27ID1-BI{Kinetix-Det:2}", ADWriterFactory.hdf(path_provider), name="kinetix2"
    )
    kinetix3 = KinetixDetector(
        "XF:27ID1-BI{Kinetix-Det:3}", ADWriterFactory.hdf(path_provider), name="kinetix3"
    )
    kinetix4 = KinetixDetector(
        "XF:27ID1-BI{Kinetix-Det:4}", ADWriterFactory.hdf(path_provider), name="kinetix4"
    )

    # Optique-Peter microscope optics
    double_obj_camera = DoubleObjCamera("XF:27IDF-OP:1{OPT:1-Ax:", name="double_obj_camera")
    wide_fov_camera = WideFOVCamera("XF:27IDF-OP:1{OPT:2-Ax:", name="wide_fov_camera")

    phantom1 = PhantomDetector(
        "XF:27ID1-ES{Phantom-Det:1}", ADWriterFactory.hdf(path_provider), name="phantom1"
    )


    diamond_window_camera = VimbaDetector(
        "XF:27IDA-BI{FAM:1-Cam:1}cam1:", ADWriterFactory.hdf(path_provider), name="diamond_window_camera"
    )
    sample_camera = VimbaDetector(
        "XF:27ID1-ES{Sample-Cam:1}cam1:", ADWriterFactory.hdf(path_provider), name="sample_camera"
    )

    fs_window_stats = NDStatsIO("XF:27IDA-BI{FS:1-Cam:1}Stats1:", name="fs_window_stats")
    fs_window = VimbaDetector(
        "XF:27IDA-BI{FS:1-Cam:1}cam1:",
        ADWriterFactory.hdf(path_provider),
        name="fs_window",
        plugins = {
            "stats1": fs_window_stats
        }
    )
    # TODO: Remove this once the StandardDetector -> StandardReadble change is merged.
    # TODO: Use mean rather than total, once available.
    fs_window.add_detector_logics(PluginSignalDataLogic(fs_window.driver, fs_window_stats.total))

    f_hutch_camera = VimbaDetector(
        "XF:27IDA-BI{GigE-Cam:5}cam1:", ADWriterFactory.hdf(path_provider), name="f_hutch_camera"
    )


    # TODO: Perkin elmer.


# Install a suspender to pause the RunEngine if the beam current drops below 100 mA
# and resume when it rises above 300 mA.
RE.install_suspender(SuspendFloor(storage_ring.beam_current, 100, resume_thresh=390))

# Configure baseline supplemental data to include in the metadata of every run.
sd = bpp.SupplementalData(baseline=[
    storage_ring.beam_current,
    sample_tower,
    dclm,
    optics_table,
])
RE.preprocessors.append(sd)
