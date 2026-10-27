# The collection profile

`src/hextools/profiles/collection.py` is the startup script for every
acquisition session. `pixi run start` runs it in IPython, and the GUI runs it
before attaching itself. When it finishes, the session contains a configured
RunEngine (`RE`), every beamline device, and the HEX plans.

This page walks through the file from top to bottom.

## Imports

```python
"""Bluesky profile for the 27-ID-1 HEX beamline at NSLS-II."""

import os

from hextools.detectors.germ import GeRMDetector, GeRMStatsDataLogic

from bluesky import preprocessors as bpp
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.run_engine import (
    RunEngine,
    autoawait_in_bluesky_event_loop,
    call_in_bluesky_event_loop,
)
from bluesky.utils import ProgressBarManager
from bluesky_tiled_plugins import TiledWriter
...
```

The import block pulls in three kinds of names:

- **Infrastructure** — `RunEngine`, `BestEffortCallback` (live tables and
  plots), `ProgressBarManager` (terminal progress bars), `TiledWriter` (sends
  documents to Tiled), `SuspendFloor`, the Tiled client constructors
  `from_uri` and `simple`, and `NSLS2PathProvider` (decides where detectors
  write files).
- **Device classes** — upstream ophyd-async detectors (`KinetixDetector`,
  `VimbaDetector`, `ContAcqDetector`, `HDFPanda`, `NDStatsIO`) and the HEX
  devices from `hextools` (`Shutter`, `Slits`, `DCLM`, `SampleTower`,
  `OpticsTable`, `EDXDTable`, `CollimatorTable`, the microscope cameras,
  `PhantomDetector`, `GeRMDetector`, `NSLS2StorageRing`, filters).
- **Plans** — everything a user should be able to run at the prompt:
  `bp` and `bps` (bluesky's built-in plans and stubs), `move_motor`,
  `move_slits`, `change_energy`, `change_beam_mode`, the tomography plans (`tomo_flyscan`,
  `tomo_alignment_scan`, `tomo_1d_step_scan`, `tomo_2d_step_scan`,
  `tomo_nd_step_scan`) and the EDXD plans (`edxd_scan`, `edxd_count`,
  `edxd_grid_scan`, `edxd_2theta_tilt`, `edxd_calib_scan`,
  `edxd_custom_pos_list_grid`, `configure_test_pulses`).

Because the profile is run with `%run -i`, everything it imports ends up in the
interactive namespace. That is also how the GUI finds plans and devices by
name, so **a plan has to be imported here to be usable from the GUI**.

```python
from ophyd_async.sim import SimMotor
```

Simulated motors, used for `sim_x`/`sim_y`/`sim_z` below.

## and environment

```python
os.environ["REDIS_HOST"] = "xf27id1-hex-redis1.nsls2.bnl.gov"
```

The Redis server that holds the beamline's shared metadata (active proposal,
cycle, user). `nslsii`'s Redis helpers read this variable.

```python
os.environ["OPHYD_ASYNC_PRESERVE_DETECTOR_STATE"] = "YES"
```

When a detector is triggered without first being prepared (for example by
`bp.count`), ophyd-async normally resets it to a default configuration. With
this set, it instead reads the detector's current settings (number of images,
exposure, ...) and keeps them, so settings made in the detector's own GUI are
respected.

## Banner and logging

```python
print_version_info()
```

Prints the bluesky, ophyd-async, Tiled and hextools versions, which is the
first thing to check when debugging.

```python
logger = configure_logger()
```

Configures the `hextools` logger to print to the console with level-colored
output. Plans log a plain-language description of each step to this logger,
and the GUI's Log and Debug views listen to it too.

## The RunEngine

```python
RE: RunEngine = initialize_run_engine()
```

Creates the RunEngine. On the beamline its metadata dictionary, `RE.md`, is a
`RedisJSONDict`: it lives in Redis, so the proposal set by `sync-experiment`
(from any process) is what gets written into every run. In mock mode it is an
ordinary dictionary pre-filled with a mock proposal (`pass-123456`).

```python
RE.md["facility"] = "NSLS-II"
RE.md["group"] = "HEX"
RE.md["beamline_id"] = "27-ID-1"
```

Static metadata recorded in the start document of every run.

```python
log_plan_exceptions(RE, logger)
log_state_changes(RE, logger)
```

Make the RunEngine report through the logger: a plan that fails is logged as
an error (with the underlying cause, not just "status failed"), an aborted or
halted plan as a warning, and every state change ("RunEngine changed from idle
to running.") as
info, except a pause, which is a warning, and a panic, which is an error.

```python
RE.waiting_hook = ProgressBarManager()
```

Shows a progress bar in the terminal whenever the RunEngine waits on a device
(motor moves, detector acquisitions).

```python
print_proposal_info(RE.md)
```

Prints the active proposal's title, type and PI.

## IPython integration

```python
ipython = get_ipython()
if ipython is not None and isinstance(ipython, TerminalInteractiveShell):
    ipython.prompts = ProposalIDPrompt(RE, ipython)
    ipython.xmode = "Minimal"
    autoawait_in_bluesky_event_loop()
```

Only in an interactive terminal session (not under the Queue Server):

- `ProposalIDPrompt` replaces `In [1]:` with the active proposal,
  e.g. `pass-123456 [1]:`, so you always know where data is going.
- `xmode = "Minimal"` keeps tracebacks to the essential line, which is what
  users need to see when a plan fails.
- `autoawait_in_bluesky_event_loop()` lets you `await` ophyd-async coroutines
  at the prompt (e.g. `await sample_tower.y.user_readback.get_value()`); they
  run in the RunEngine's event loop, where the devices live.

## Tiled

```python
if not is_running_in_ci():
    tiled_writing_client = from_uri(
        "https://tiled.nsls2.bnl.gov",
        api_key=os.environ.get("TILED_BLUESKY_WRITING_API_KEY_HEX", ""),
    )["hex"]["raw"]
    if ipython is not None and isinstance(ipython, TerminalInteractiveShell):
        tiled_reading_client = c = from_uri("https://tiled.nsls2.bnl.gov")["hex"]["raw"]
else:
    tiled_writing_client = tiled_reading_client = c = simple()
```

Two clients for the HEX raw catalog:

- `tiled_writing_client` authenticates with the beamline's writing API key
  (from the `TILED_BLUESKY_WRITING_API_KEY_HEX` environment variable) and is
  used only to store data.
- `tiled_reading_client`, also available as `c`, is for you to browse and load
  data at the prompt. It authenticates as you, so you only see data you have
  access to. It is created only in interactive sessions.

In mock mode, `simple()` starts a temporary Tiled server on the local machine
and both names point to it.

```python
RE.subscribe(TiledWriter(tiled_writing_client))
```

Every document the RunEngine emits (start, descriptors, events, stop, and
references to detector files) is written to Tiled.

```python
bec = BestEffortCallback()
RE.subscribe(bec)
```

The `BestEffortCallback` prints a live table of each scan in the terminal and
opens plots for scans that have a clear independent variable.

## Detector file paths

```python
path_provider = NSLS2PathProvider(RE.md)
```

Decides where detectors write their files. It is given `RE.md` so that it can
place files in the active proposal's directory.

## Devices

```python
with auto_init_devices(timeout=2.0, verbose=True):
    fe_shutter = Shutter("XF:27IDA-PPS{Sh:FE}", name="front-end-shutter")
    ...
```

Every device is created inside `auto_init_devices`. When the block ends, it
connects all of them concurrently (allowing 2 s each) and prints the
connection report (`[   OK   ]`, `[ FAILED ]`, and signal counts for partly
connected devices). With `verbose=True` it also prints why each failed device
failed. In mock mode the devices connect to mock backends instead. Devices
created without an explicit `name` are named after their variable.

The devices, in order:

| Variable(s)                                   | Device                                                                                                   |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `fe_shutter`, `photon_shutter`                | Front-end and photon shutters.                                                                           |
| `a_slits`, `f_slits`                          | Slits in the A (optics) and F (experimental) hutches.                                                    |
| `storage_ring`                                | Storage-ring status, including the beam current used by the suspender below.                              |
| `monochromator`, `mono`, `dclm`               | Three names for the double crystal Laue monochromator; its `set` changes the beam mode.                   |
| `optics_table`                                | Motors of the optics table.                                                                              |
| `sample_tower`, `rot_motor`                   | The sample stage. `rot_motor` is an alias for its rotation axis, `sample_tower.ry2`, used by tomography. |
| `edxd_table`, `collimator_table`              | The EDXD detector table and collimator table.                                                            |
| `filters`, plus one variable per filter       | Filters built from the filter configuration file by `load_filters()`.                                    |
| `panda`                                       | The PandABox, which triggers detectors and records rotation angles in flyscans.                          |
| `kinetix1` … `kinetix4`                       | The four Kinetix cameras used for imaging.                                                               |
| `double_obj_camera`, `wide_fov_camera`        | Motors of the two Optique-Peter microscope objectives.                                                   |
| `phantom`                                     | The Phantom high-speed camera.                                                                           |
| `diamond_window_cam`, `sample_cam`, `fs_window_cam`, `f_hutch_cam` | Visible-light (Vimba) cameras.                                                       |
| `fs_window_stats`                             | The statistics plugin of the fluorescence-screen camera, used by energy auto-tuning.                     |
| `perkin_elmer`                                | The Perkin Elmer area detector for XRD.                                                                  |
| `germ`                                        | The GeRM detector for EDXD.                                                                              |
| `sim_x`, `sim_y`, `sim_z`                     | Simulated motors for testing.                                                                            |

A few lines inside the block need a closer look:

```python
    if ipython is not None:
        for filter in filters:
            ipython.user_ns[filter.name] = filter
```

Puts each filter in the namespace under its own name so it can be moved
directly (`RE(bps.mv(filter2, ...))`).

```python
    fs_window_cam.add_detector_logics(
       PluginSignalDataLogic(fs_window_cam.driver, fs_window_stats.mean_value)
    )
```

Makes the fluorescence-screen camera also report its mean intensity as a
number with each image. Energy auto-tuning maximises this value. The `TODO`
above it notes that this will become unnecessary after an upstream
ophyd-async change.

```python
    pe_path_provider = NSLS2PathProvider(
        RE.md, base_write_dir=PureWindowsPath("Z:\\proposals")
    )
```

The Perkin Elmer IOC runs on Windows, so it gets its own path provider that
writes Windows paths (`Z:\proposals\...`) for the same proposal directory.

## Suspender

```python
RE.install_suspender(SuspendFloor(storage_ring.beam_current, 100, resume_thresh=390))
```

If the storage-ring current drops below 100 mA, the RunEngine suspends the
running plan. Once the current recovers above 390 mA it resumes automatically,
rewinding to the last checkpoint so the interrupted step is repeated.

## Baseline readings

```python
sd = bpp.SupplementalData(
    baseline=[
        storage_ring.beam_current,
        a_slits,
        f_slits,
        sample_tower,
        monochromator,
        optics_table,
        edxd_table,
    ]
)
RE.preprocessors.append(sd)
```

Every run records these devices once at the start and once at the end, in a
`baseline` stream. This captures the state of the beamline (beam current, slit
gaps, stage positions, monochromator setting) for every dataset without adding
them to each plan.

```python
bec.disable_baseline()
```

Stops the live table from printing the baseline readings, which would
otherwise fill the terminal at the start and end of every run.
