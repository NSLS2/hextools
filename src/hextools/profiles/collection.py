"""Bluesky profile for the 27-ID-1 HEX beamline at NSLS-II."""

import os

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
from ophyd_async.epics.adcore import ADWriterFactory
from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.epics.advimba import VimbaDetector
from ophyd_async.fastcs.panda import HDFPanda
from tiled.client import from_uri, simple
from bluesky.suspenders import SuspendFloor
from bluesky import plans as bp, plan_stubs as bps, preprocessors as bpp

from hextools.detectors.phantom import PhantomDetector
from hextools.machine import NSLS2StorageRing
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


# Subscribe the tiled writer to the RunEngine
RE.subscribe(TiledWriter(tiled_writing_client))

# Subscribe the best effort callback
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
    # Shutters (Front-end and photon)
    fe_shutter = Shutter("XF:27IDA-PPS{Sh:FE}", name="front_end_shutter")
    photon_shutter = Shutter("XF:27IDA-PPS{L1-S1}", name="photon_shutter")

    # Storage ring information
    storage_ring = NSLS2StorageRing()

    # Monochromator DCLM (Double Crystal Laue Monochromator)
    dclm = DCLM("XF:27IDA-OP:1{Mono:DCLM-Ax:", name="dclm")

    # Generate filter objects from the configuration file
    filters = load_filters()

    # Add filters directly to namespace for convenience in interactive sessions
    if ipython is not None:
        for filter in filters:
            ipython.user_ns[filter.name] = filter


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


# Install a suspender to pause the RunEngine if the beam current drops below 100 mA
# and resume when it rises above 300 mA.
RE.install_suspender(SuspendFloor(storage_ring.beam_current, 100, resume_thresh=300))

# Configure baseline supplemental data to include in the metadata of every run.
sd = bpp.SupplementalData(baseline=[
    storage_ring.beam_current,
])
RE.preprocessors.append(sd)
