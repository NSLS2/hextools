"""Bluesky profile for the 27-ID-1 HEX beamline at NSLS-II."""

import os
from pathlib import (
    Path,  # noqa: F401 - operator namespace
    PureWindowsPath,
)

from bluesky import plan_stubs as bps  # noqa: F401 - operator namespace
from bluesky import plans as bp  # noqa: F401 - operator namespace
from bluesky import preprocessors as bpp
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.run_engine import (
    RunEngine,
    autoawait_in_bluesky_event_loop,
    call_in_bluesky_event_loop,  # noqa: F401 - operator namespace
)
from bluesky.utils import ProgressBarManager  # noqa: F401 - operator namespace
from bluesky_tiled_plugins import BufferingWrapper, TiledWriter
from IPython.core.getipython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from nslsii.ophyd_async.providers import NSLS2PathProvider
from nslsii.utils import ProposalIDPrompt, print_proposal_info, print_version_info
from ophyd_async.core import init_devices, print_connection_status
from ophyd_async.epics.adcore import (
    ADWriterFactory,
    NDStatsIO,
    PluginSignalDataLogic,
)
from ophyd_async.epics.adkinetix import (
    KinetixDetector,  # noqa: F401 - operator namespace
)
from ophyd_async.epics.advimba import VimbaDetector
from ophyd_async.fastcs.panda import HDFPanda
from ophyd_async.sim import SimMotor
from tiled.client import from_uri, simple

from hextools.detectors.germ import (
    GeRMDetector,
    GeRMStatsDataLogic,  # noqa: F401 - operator namespace
)
from hextools.detectors.kinetix import kinetix_factory
from hextools.detectors.perkin_elmer import PerkinElmerDetector
from hextools.detectors.phantom import PhantomDetector
from hextools.edxd import *
from hextools.log import *
from hextools.machine import *
from hextools.motors import (
    CollimatorTable,
    DiffractionTable,
    EDXDTable,
    FOV_2_4_mm_Camera,
    FOV_20_40_mm_Camera,
    OpticsTable,
    SampleSmall,
    SampleTower,
    move_motor,  # noqa: F401 - operator namespace
)
from hextools.photon_delivery_system import *
from hextools.suspenders import SuspendFloorUnlessOpsMode
from hextools.tomography import *
from hextools.utils import (
    StartDocAwareFilenameProvider,
    initialize_run_engine,
    is_running_in_ci,
    reset_scan_id,  # noqa: F401 - operator namespace
    show_docs,  # noqa: F401 - operator namespace
)
from hextools.utils.path_providers import HEXPathProvider, mock_path_provider
from hextools.xrd import *

# TODO: Enable mock customization via flags once
# https://github.com/bluesky/ophyd-async/pull/1446 is merged
# parser = argparse.ArgumentParser(description="HEX Tools Collection Profile")
# parser.add_argument(
#   "--mock", action="store_true"help="Mock all beamline devices"
# )
# parser.add_argument(
#   "--mock-shutters", action="store_true"help="Mock all shutters"
# )
# parser.add_argument(
#   "--mock-motors", action="store_true"help="Mock all motors"
# )
# parser.add_argument(
#   "--mock-motors-no-rot",
#   action="store_true",
#   help="Mock all motors except rotation stage"
# )

# _args = parser.parse_args()
# _mock_config = {
#     Device: _args.mock,
#     AsyncEpicsMotor: _args.mock_motors or _args.mock_motors_no_rot,
#     Shutter: _args.mock_shutters,
#     RotationMotor: _args.mock_motors_no_rot,
# }

# Where mock detectors write their files, instead of the proposal directories.
MOCK_DETECTOR_DATA_DIR = "/tmp/hextools_dets"

# Where TiledWriter backs up documents it could not write to Tiled.
if is_running_in_ci():
    TILED_BACKUP_ROOT = "/tmp/bluesky-docs"
else:
    TILED_BACKUP_ROOT = "/nsls2/data/hex/shared/config/bluesky/tiled_backup"

# Environment variables for Redis host and ophyd_async detector state preservation
os.environ["REDIS_HOST"] = "xf27id1-hex-redis1.nsls2.bnl.gov"
os.environ["OPHYD_ASYNC_PRESERVE_DETECTOR_STATE"] = "YES"

# Print version information for bluesky, ophyd_async, tiled, and hextools.
print_version_info(additional_packages=["hextools"])

# Log hextools messages (including plan narration) to the console, color-coded by level.
logger = configure_logger()

# Setup the RunEngine and its metadata.
RE: RunEngine = initialize_run_engine()
RE.md["facility"] = "NSLS-II"
RE.md["group"] = "HEX"
RE.md["beamline_id"] = "27-ID-1"

# Log any exception raised during a plan as an error, and every RunEngine state change.
log_plan_exceptions(RE, logger)
log_state_changes(RE, logger)

# Uncomment if running in the terminal to get progress bars.
# RE.waiting_hook = ProgressBarManager()  # type: ignore[assignment]

print("\n\033[1mActive Proposal Information\033[0m\n")
# Display active proposal information in the console.
print_proposal_info(RE.md)

# Get the IPython shell and set up the custom prompt to show the current proposal ID.
ipython = get_ipython()
if ipython is not None and isinstance(ipython, TerminalInteractiveShell):
    ipython.prompts = ProposalIDPrompt(RE, ipython)
    ipython.xmode = "Minimal"
    autoawait_in_bluesky_event_loop()

    # Configure plots to work interactively
    from matplotlib import pyplot as plt

    plt.ion()

    # Make tracebacks minimal by default to reduce clutter.
    # Use %tb verbose to view full traceback.
    ipython.run_line_magic("xmode", "minimal")


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
    tiled_writing_client = tiled_reading_client = c = simple(
        readable_storage=MOCK_DETECTOR_DATA_DIR
    )

tw = BufferingWrapper(
    TiledWriter(tiled_writing_client, backup_directory=f"{TILED_BACKUP_ROOT}/sql")
)

RE.subscribe(tw)

# Subscribe the best effort callback, for its plots. The GUI's live table and plan
# narration already cover what its table and heading would print.
bec = BestEffortCallback()
bec.disable_table()
bec.disable_heading()
RE.subscribe(bec)

# Keep track of last dark-flat scan UID
# TODO: Consolidate these. Or better yet, create better way to access metadata
# from within plans
RE.subscribe(LastDarkFlatUIDTracker(RE.md))
RE.subscribe(LastDarkReferenceTracker(RE.md))

filename_provider = StartDocAwareFilenameProvider()
RE.subscribe(filename_provider.doc_callback)

# Define our global default path provider for the beamline
if is_running_in_ci():
    path_provider = mock_path_provider(MOCK_DETECTOR_DATA_DIR)
else:
    path_provider = NSLS2PathProvider(
        RE.md,
        filename_provider=filename_provider,
        include_scan_id_dir=True,
        granularity="year",
        scan_id_padding=5,
    )

# For the Perkin-Elmer detector, we use a separate path provider
# Since we need to specify the write directory as a windows path.
pe_path_provider = (
    path_provider
    if is_running_in_ci()
    else HEXPathProvider(
        RE.md,
        filename_provider=filename_provider,
        base_write_dir=PureWindowsPath("Z:\\proposals"),
        include_scan_id_dir=True,
        granularity="year",
        scan_id_padding=5,
    )
)

print("\nInitializing devices...\n")

with init_devices(
    mock=is_running_in_ci(), timeout=2.0, on_connect_hook=print_connection_status
):
    # Shutters (Front-end and photon)
    fe_shutter = PPSShutter("XF:27IDA-PPS{Sh:FE}", name="front-end-shutter")
    photon_shutter = Shutter("XF:27IDA-PPS{L1-S1}", name="photon-shutter")

    # Slits
    a_slits = Slits("XF:27IDA-OP:1{Slt:1-Ax:", name="a-slits")
    f_slits = Slits("XF:27IDF-OP:1{Slt:2-Ax:", name="f-slits")
    edxd_slits = Slits("XF:27IDF-OP:1{Slt:CMT-Ax:", name="edxd-slits")
    # Front-end Slit1: its Y axis picks a fixed aperture
    fe_slit1 = FrontEndSlit("FE:C27A-OP{Slt:1-Ax:Y}", name="fe-slit1")

    # Storage ring information
    storage_ring = NSLS2StorageRing()

    # Monochromator DCLM (Double Crystal Laue Monochromator)
    monochromator = mono = dclm = DCLM(
        "XF:27IDA-OP:1{Mono:DCLM-Ax:", name="monochromator"
    )

    # Motors for the optics table
    optics_table = OpticsTable("XF:27IDF-OP:1{OPT:1-Ax:", name="optics-table")

    # Sample tower
    sample_tower = SampleTower("XF:27IDF-OP:1{SMPL:1-Ax:", name="sample-tower")
    rot_motor = sample_tower.ry2

    sample_small = SampleSmall("XF:27IDF-OP:1{MC:3-Ax:SampleS", name="sample-small")

    # EDXD Table
    edxd_table = EDXDTable("XF:27IDF-OP:1{EDXD:1-Ax:", name="edxd-table")
    collimator_table = CollimatorTable("XF:27IDF-OP:1{", name="collimator-table")

    diff_table = DiffractionTable("XF:27IDF-OP:1{DIFF:1-Ax:", name="diffraction-table")

    # Filters, first the two filter 1 (upstream/downstream), then filters 2, 3, and 4
    filter1_upstream = Filter(
        "XF:27IDA-OP:1{Fltr:1-Ax:Yu}Mtr",
        "XF:27IDA-OP:0{Fltr:1_US}Sw:InPos-Sts",
        name="filter1_upstream",
    )
    filter1_downstream = Filter(
        "XF:27IDA-OP:1{Fltr:1-Ax:Yd}Mtr",
        "XF:27IDA-OP:0{Fltr:1_DS}Sw:InPos-Sts",
        name="filter1_downstream",
    )
    filter2 = Filter(
        "XF:27IDA-OP:1{Fltr:2-Ax:Y}Mtr",
        "XF:27IDA-OP:0{Fltr:2}Sw:InPos-Sts",
        name="filter2",
    )
    filter3 = Filter(
        "XF:27IDA-OP:3{Fltr:3-Ax:Y}Mtr",
        "XF:27IDA-OP:0{Fltr:3}Sw:InPos-Sts",
        name="filter3",
    )
    filter4 = Filter(
        "XF:27IDA-OP:2{Fltr:4-Ax:Y}Mtr",
        "XF:27IDA-OP:0{Fltr:4}Sw:InPos-Sts",
        name="filter4",
    )

    # PandABox
    panda = HDFPanda("XF:27ID1-ES{PANDA:1}:", path_provider, name="panda1")

    # Kinetix detectors
    kinetix1 = kinetix_factory(1, path_provider, name="kinetix-det1")
    kinetix2 = kinetix_factory(2, path_provider, name="kinetix-det2")
    kinetix3 = kinetix_factory(3, path_provider, name="kinetix-det3")
    kinetix4 = kinetix_factory(4, path_provider, name="kinetix-det4")

    # Optique-Peter microscope optics
    double_obj_camera = FOV_2_4_mm_Camera(
        "XF:27IDF-OP:1{OPT:1-Ax:", name="double-obj-camera"
    )
    wide_fov_camera = FOV_20_40_mm_Camera(
        "XF:27IDF-OP:1{OPT:2-Ax:", name="wide-fov-camera"
    )

    phantom = PhantomDetector(
        "XF:27ID1-ES{Phantom-Det:1}",
        ADWriterFactory.hdf(path_provider, hinted=False),
        name="phantom",
    )

    diamond_window_cam = VimbaDetector(
        "XF:27IDA-BI{FAM:1-Cam:1}",
        ADWriterFactory.hdf(path_provider, hinted=False),
        name="diamond-window-cam",
    )

    sample_cam = VimbaDetector(
        "XF:27ID1-ES{Sample-Cam:1}",
        ADWriterFactory.hdf(path_provider, hinted=False),
        name="sample-cam",
    )

    fs_window_stats = NDStatsIO(
        "XF:27IDA-BI{FS:1-Cam:1}Stats1:", name="fs-window-stats"
    )
    fs_window_cam = VimbaDetector(
        "XF:27IDA-BI{FS:1-Cam:1}",
        ADWriterFactory.hdf(path_provider, hinted=False),
        name="fs-window-cam",
        plugins={"stats1": fs_window_stats},
    )
    # TODO: Remove this once the StandardDetector -> StandardReadble change is merged.
    fs_window_cam.add_detector_logics(
        PluginSignalDataLogic(fs_window_cam.driver, fs_window_stats.mean_value)
    )

    f_hutch_cam = VimbaDetector(
        "XF:27IDA-BI{GigE-Cam:5}",
        ADWriterFactory.hdf(path_provider, hinted=False),
        name="f_hutch_cam",
    )

    perkin_elmer = PerkinElmerDetector(
        "XF:27ID1-ES{PE-Det:1}",
        pe_path_provider,
        name="perkin-elmer",
        proc_suffix="Proc1:",
    )

    germ = GeRMDetector(
        "XF:27ID1-ES{GeRM-Det:1}",
        path_provider,
        name="germ",
    )

    # A few simulated motors, useful for testing
    sim_x = SimMotor(name="sim_x", instant=False)
    sim_y = SimMotor(name="sim_y", instant=False)
    sim_z = SimMotor(name="sim_z", instant=False)


# Pause the RunEngine if the beam current drops below 100 mA and resume above
# 390 mA. Always installed; it ignores low beam while the ring is in maintenance
# or shutdown, checking the mode each time rather than once at startup.
RE.install_suspender(
    SuspendFloorUnlessOpsMode(
        storage_ring.beam_current,
        100,
        resume_thresh=390,
        ops_mode=storage_ring.operating_mode,
        skip_modes=(NSLS2OpsMode.MAINTENANCE, NSLS2OpsMode.SHUTDOWN),
    )
)

# Configure baseline supplemental data to include in the metadata of every run.
sd = bpp.SupplementalData(
    baseline=[
        storage_ring.beam_current,
        fe_slit1,
        a_slits,
        f_slits,
        sample_tower,
        monochromator,
        optics_table,
        edxd_table,
        diff_table,
        # collimator_table,
    ]
)
RE.preprocessors.append(sd)

# Disable bec printout of baseline readings
# since it is kind of noisy
bec.disable_baseline()
