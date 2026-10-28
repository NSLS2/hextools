# Developer guide

This guide covers how the code is organised, how to write a new plan, how to
test it, and how to make it available in the GUI.

## Development setup

```bash
pixi install -e dev
pixi run -e py313 tests        # pytest with coverage
pixi run -e py313 lint         # pre-commit hooks (ruff, ...)
pixi run -e py313 type-check   # ty
pixi run start-mock            # try your code in a mock IPython session
pixi run hex-gui-mock          # try it in the GUI
```

## Repository layout

```text
src/hextools/
├── profiles/collection.py      # startup profile (see "The collection profile")
├── utils/                      # session helpers: RunEngine setup, device init, proposals
├── log.py                      # the "hextools" logger and RunEngine logging hooks
├── specs.py                    # Tiled specs and validators for HEX run types
├── machine.py                  # storage ring
├── motors.py                   # stages, tables, microscope optics, move_motor
├── photon_delivery_system/     # shutters, slits, filters, DCLM; beam mode and energy plans
├── detectors/                  # Phantom, GeRM, Kinetix helpers
├── tomography/                 # flyscans, step scans, radiography, alignment
├── edxd.py                     # energy-dispersive diffraction plans
├── xrd/                        # angle-dispersive diffraction plans
└── gui/                        # the HEX Data Acquisition GUI
tests/                          # pytest suite, mirroring src/hextools
docs/                           # this documentation
```

## Writing a plan

A plan is a generator function that yields bluesky messages, usually by
delegating to bluesky's built-in plans (`bluesky.plans`, `bp`) and plan stubs
(`bluesky.plan_stubs`, `bps`). Plans never touch hardware directly: they ask
the RunEngine to do it, which is what makes them pausable, resumable and
recorded.

### Conventions

HEX plans follow a few conventions. Most of them exist so that the GUI can
build a form for the plan automatically, so following them is what makes a
plan usable from the GUI with no extra work.

1. **Type-hint every parameter.** The GUI chooses an input widget from the
   annotation (see [the table below](form-fields)). A
   parameter without a usable annotation gets no field.
2. **Write a numpydoc docstring with a `Parameters` section.** Each parameter's
   description becomes the tooltip on its form field.
3. **Take supporting devices as optional keyword arguments defaulting to
   `None`**, and resolve them with `ensure_available`. These are devices the
   user rarely chooses (the PandA, rotation stage, shutters). The GUI hides them,
   and at the prompt the plan finds the device of the same name in the session
   namespace:

   ```python
   panda = ensure_available(HDFPanda, panda=panda)
   ```

   `ensure_available` returns the argument if one was given, otherwise the
   object of that name in the IPython namespace, and raises a clear error if
   neither exists or the type is wrong. The keyword must match the variable
   name used in the profile.
4. **Take the devices the user chooses (detectors, the scanned motor) as
   required parameters**, typed with the device class. These become device
   selectors in the GUI.
5. **Set `plan_name` in the run metadata to the function's name.** The Tiled
   specs and the GUI's Plan Status panel use it to identify the plan.
6. **Guard shutters explicitly.** Use `ensure_shutter_open`/`ensure_shutter_closed`
   and close the photon shutter in a cleanup step (`bpp.finalize_wrapper`) so it
   is closed even if the plan fails or is aborted. Never actuate the front-end
   shutter without the user asking for it.
7. **Use the `hextools` logger, not `print`,** for messages users should see.
   They appear in the terminal and in the GUI's Log tab:

   ```python
   import logging
   from hextools.log import LOGGER_NAME

   logger = logging.getLogger(LOGGER_NAME)
   logger.info("Found the rotation axis at %.3f mm", center)
   ```

### Example

The following plan takes one image per height while stepping the sample tower
vertically. It is short, but it follows every convention above.

```python
"""src/hextools/tomography/survey.py"""

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import bluesky.preprocessors as bpp
from ophyd_async.epics.adkinetix import KinetixDetector

from hextools.motors import SampleTower
from hextools.photon_delivery_system import Shutter
from hextools.photon_delivery_system.shutter import ensure_shutter_closed, ensure_shutter_open
from hextools.utils import ensure_available


def height_survey(
    detectors: list[KinetixDetector],
    start: float,
    stop: float,
    num: int,
    exposure_time: float,
    sample_name: str | None = None,
    use_shutter: bool = True,
    sample_tower: SampleTower | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
):
    """Take one image per height while stepping the sample tower vertically.

    Parameters
    ----------
    detectors : list[KinetixDetector]
        Detectors to take images with.
    start : float
        First sample tower z1 position, in mm.
    stop : float
        Last sample tower z1 position, in mm.
    num : int
        Number of positions, including start and stop.
    exposure_time : float
        Exposure time per image, in seconds.
    sample_name : str, optional
        Name of the sample, recorded in the run metadata.
    use_shutter : bool, optional
        Open the photon shutter for the survey and close it afterwards.
    sample_tower : SampleTower, optional
        The sample tower. If None, ``sample_tower`` from the session is used.
    fe_shutter : Shutter, optional
        Front-end shutter, which must already be open. If None, ``fe_shutter``
        from the session is used.
    photon_shutter : Shutter, optional
        Photon shutter. If None, ``photon_shutter`` from the session is used.
    """
    sample_tower = ensure_available(SampleTower, sample_tower=sample_tower)
    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)

    md = {"plan_name": "height_survey"}
    if sample_name is not None:
        md["sample_name"] = sample_name

    def _body():
        for det in detectors:
            yield from bps.mv(det.driver.acquire_time, exposure_time)
        if use_shutter:
            yield from ensure_shutter_open(fe_shutter)
            yield from ensure_shutter_open(photon_shutter, allow_actuation=True)
        return (yield from bp.scan(detectors, sample_tower.z1, start, stop, num, md=md))

    def _cleanup():
        if use_shutter:
            yield from ensure_shutter_closed(photon_shutter, allow_actuation=True)

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))
```

Things to notice:

- Nothing happens when `height_survey(...)` is called; the work starts when the
  RunEngine iterates the generator. Keep that in mind when validating inputs:
  raising inside the plan (as `ensure_available` does) reports the error when
  the plan starts.
- `bp.scan` opens and closes the run. Pass your metadata through its `md`
  argument; anything in `md` is stored in the run's start document and is
  searchable in Tiled.
- The plan returns what `bp.scan` returns (the run uid), so callers composing
  plans can use it.

### Making the plan available

1. Put the plan in the module for its technique (here
   `src/hextools/tomography/`), and export it from the package `__init__.py`
   if the package re-exports its plans.
2. Import it in `src/hextools/profiles/collection.py` so it exists in every
   session:

   ```python
   from hextools.tomography.survey import height_survey
   ```

3. Restart the session (`pixi run start-mock`) and try it:

   ```python
   RE(height_survey([kinetix1], 0, 5, 11, exposure_time=0.01, use_shutter=False))
   ```

### Testing a plan

Tests run plans against mock devices, so they need no hardware. Create devices
with `init_devices(mock=True)`, drive mock signals with
`set_mock_value`/`callback_on_mock_put`, and run the plan in a RunEngine.
Collect the emitted documents to check what was recorded:

```python
import pytest
from bluesky import RunEngine
from ophyd_async.core import init_devices

from hextools.motors import SampleTower
from hextools.photon_delivery_system import Shutter
from hextools.tomography.survey import height_survey


@pytest.fixture
def devices():
    with init_devices(mock=True):
        sample_tower = SampleTower("TEST:", name="sample_tower")
        fe_shutter = Shutter("FE:", name="fe_shutter")
        photon_shutter = Shutter("PH:", name="photon_shutter")
    return sample_tower, fe_shutter, photon_shutter


def test_height_survey_records_plan_name(RE: RunEngine, devices):
    sample_tower, fe_shutter, photon_shutter = devices
    docs = []
    RE.subscribe(lambda name, doc: docs.append((name, doc)))

    RE(height_survey([], 0, 1, 3, exposure_time=0.01, use_shutter=False,
                     sample_tower=sample_tower, fe_shutter=fe_shutter,
                     photon_shutter=photon_shutter))

    start = next(doc for name, doc in docs if name == "start")
    assert start["plan_name"] == "height_survey"
    assert sum(name == "event" for name, _ in docs) == 3
```

The `RE` fixture comes from `tests/conftest.py`. Pass every supporting device
explicitly in tests: there is no IPython namespace for `ensure_available` to
search. See `tests/tomography/` for fixtures that build mock shutters which
respond to open/close commands and mock Kinetix detectors with file writers.

### Describing the data (optional)

`src/hextools/specs.py` describes, for each kind of HEX run, which metadata and
streams it contains, as a Tiled `Spec` plus a validator. Specs let analysis code
and the GUI's Scan Browser find runs of a given kind. To add one:

1. Document the run's start-document keys and streams in a comment, following
   the existing entries.
2. Define the spec, e.g. `HEIGHT_SURVEY_V1 = Spec("HeightSurvey", version="1")`.
3. Write `validate_height_survey(run, version="1")` using the `_require_*`
   helpers.
4. Register it in `_VALIDATORS` and `ALL_SPECS`, and add tests to
   `tests/test_specs.py`.

## Adding a plan to the GUI

The GUI is in two parts:

- `src/hextools/gui/` is the generic GUI. It knows nothing about HEX and does
  not import the rest of `hextools` (`tests/gui/test_gui_is_generic.py`
  enforces this).
- `src/hextools/gui_config.py` is the HEX configuration: a `GuiSpec` called
  `HEX_GUI` listing the technique tabs, their plans and cameras, the expected
  devices, the shutters box, the proposal system and the site.

`python -m hextools.gui` builds the GUI from `HEX_GUI`. Each technique tab
(Beamline, Tomography, EDXD, ...) is a `Technique` in that spec, with a column
of plan tabs. Each plan tab is a `QtPlanWidget`, which builds its form from the
plan's signature, so adding a plan is usually one line.

### Adding a plan tab

Import the plan in `gui_config.py` and add a `(label, plan)` pair to the
technique's `plans`:

```python
from hextools.tomography.survey import height_survey
...
Technique(
    "Tomography",
    plans=[
        ("Flyscan", tomo_flyscan),
        ("Height Survey", height_survey),
        ...
    ],
    ...
)
```

The GUI wraps each plan in `QtPlanWidget(re_client, plan, title=None)`, which
takes the IPython session (a `SessionClient`) or the Queue Server client, and the
plan function. The form's title defaults to the function name in title case.

When the user presses **Run Plan**, the widget:

1. checks every field (required fields filled, numbers valid, devices exist and
   have the right type);
2. IPython-session mode: builds the call as text, e.g.
   `RE(height_survey(detectors=[kinetix1], start=0.0, stop=5.0, num=11, exposure_time=0.01, use_shutter=True))`,
   and runs it in the IPython session. Devices are passed by their names in the
   session namespace, and **the plan is looked up by its function name**, which
   is why it must be imported in the profile;
3. Queue Server mode: adds a `BPlan(plan.__name__, **kwargs)` item to the
   queue. The plan must be loaded in the Queue Server worker's namespace and be
   allowed for the user group in the Queue Server's permissions.

(form-fields)=
### How parameters become form fields

| Annotation                                          | Form field                                                                    |
| --------------------------------------------------- | ----------------------------------------------------------------------------- |
| `int`, `float`, `str` (or `... \| None`)           | Text box, validated as a number for `int`/`float`.                             |
| `bool`                                              | Checkbox.                                                                     |
| An `Enum` (e.g. `StrictEnum`)                       | Drop-down list of the enum's values.                                          |
| `list[float]`, `list[int]`, `list[str]`             | Text box taking comma-separated values.                                       |
| A device class, or a union of them, **required**    | Drop-down of the session's devices of that type.                              |
| `list[DeviceClass]` (or `list[A \| B]`)             | Growable list of device drop-downs with a **+** button.                        |
| A device class with default `None`                  | *No field*: the plan resolves it itself (`ensure_available`).                  |
| `*args`, `**kwargs`, anything else                  | *No field*.                                                                   |

Other details:

- Required parameters (no default) show `required`; the form refuses to run
  until they are filled.
- Defaults are pre-filled; a default of `None` shows `optional`, and an empty
  optional field is left out of the call so the plan's default applies.
- Field labels are the parameter names in title case (`exposure_time` →
  "Exposure Time"). Choose parameter names with that in mind.
- Tooltips come from the numpydoc `Parameters` section.
- Device drop-downs update as devices appear in or disappear from the session.

If a plan's natural signature doesn't map well onto a form (for example it
takes `*args` like `bp.scan`), write a thin wrapper with an explicit signature,
as `move_motor` does for `bps.mv`:

```python
def move_motor(motor: AsyncEpicsMotor, value: float, timeout: float | None = None):
    """Move the specified motor to the given position. ..."""
    yield from bps.mv(motor, value, timeout=timeout)
```

### Adding a technique tab

Add a `Technique` to `HEX_GUI.techniques` in `gui_config.py`:

```python
Technique(
    "Imaging",
    plans=[("Height Survey", height_survey)],
    detectors=[DetectorView("kinetix1", "XF:27ID1-BI{Kinetix-Det:1}")],
    combined=[("Dual Cam", ["kinetix1", "kinetix3"])],
),
```

Each `DetectorView(name, prefix)` adds a live viewer for the device called
`name` in the session, reading images from the PV Access prefix `prefix`. It is
only shown if the device exists, so the GUI still starts when a detector is
missing. Each `combined` entry adds a side-by-side view of several detectors.

### Other extension points

All of these are fields of `GuiSpec` (see its docstring in
`src/hextools/gui/spec.py`):

- **Available Devices tab** — `expected_devices` lists the devices the beamline
  should have, with their types; add new devices there.
- **Top-row widgets** — `status_widgets` holds factories called as
  `factory(session)` in IPython-session mode; HEX uses one for its
  shutters box.
- **Extra tabs** — `extra_tabs` holds `(title, factory)` pairs added after the
  built-in tabs; HEX uses one for NXS Export.
- **Proposals** — `proposals` connects Proposal Info's **Change Proposal** to
  the facility's proposal system. Without it, that button is hidden.
- **Date, time and weather** — `site`; the box is omitted without one.
- **Scan Browser** — `run_specs` lists the Tiled specs it offers.
  `QtScanBrowser.add_callback(spec, label, callback)` adds a button (and
  right-click action) for runs of a given spec; `callback` is called with the
  selected run's uid.
- **Menus** — `QtDataAcquisitionWindow._build_menus` in
  `src/hextools/gui/window.py`.

### The GUI and the IPython session

The GUI runs in its own process, so a crash in it cannot stop a plan or its
cleanup. `start_gui()` (in `hextools/gui/session_server.py`) runs in the IPython
session: it starts a small server on a Unix socket only you can open, then
launches the GUI. The GUI talks to the session through a `SessionClient`
(`hextools/gui/session_client.py`), which widgets take where they once took
the RunEngine:

- **Commands** (Run Plan, Resume, Stop, the shutters, Re-run) are sent as code
  and run at the session's prompt, echoed in its terminal. Pause goes straight
  to the RunEngine from the server's thread, since a running plan holds the
  session's main thread.
- **What the GUI shows** is sent by the session: RunEngine state, plans started
  and finished, documents, the narrator's log lines, device progress, the
  devices in the namespace (with their class hierarchy, so device fields match
  type hints exactly), `RE.md` and shutter readings.

A widget that needs something new from the session needs a new message in
both. Never use Qt's blocking `waitFor...` calls in the GUI: PySide6 holds the
GIL while it waits, which stalls every Python thread in the process.

### The GUI for another beamline

Another beamline's tools package writes its own spec and launches the same GUI
with it, e.g. a `pdftools/gui_config.py` (the plan and PV names here are
placeholders):

```python
from hextools.gui.spec import DetectorView, GuiSpec, Technique
from pdftools.plans import pdf_count

PDF_GUI = GuiSpec(
    name="PDF",
    techniques=[
        Technique(
            "PDF",
            plans=[("Count", pdf_count)],
            detectors=[DetectorView("pe1", "XF:28ID1-ES{Det:PE1}")],
        ),
    ],
    logger_name="pdftools",
)
```

and a short `pdftools/gui/__main__.py`:

```python
from hextools.gui.window import run

run("pdftools.gui_config:PDF_GUI")
```

Fields left out keep their defaults: no shutters box, no extra tabs, no Change
Proposal, no weather box.

## Updating the documentation

The documentation is built with Sphinx and MyST Markdown from `docs/`. The GUI
screenshots are generated, not taken by hand; regenerate them after changing
the GUI:

```bash
QT_QPA_PLATFORM=offscreen pixi run -e dev python docs/_scripts/capture_screenshots.py
```

The script runs the profile in mock mode, opens the GUI offscreen, runs two
small plans so the execution views have content, and writes the PNGs to
`docs/_static/screenshots/`.
