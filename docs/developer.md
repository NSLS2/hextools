# Developer guide

This guide covers how the code is organised, how to write a new plan, how to
test it, and how to make it available in the GUI.

## Development setup

```bash
pixi install -e dev
pixi run tests                 # unit tests, no coverage
pixi run gui-tests             # GUI tests (their own process, as in CI)
pixi run all-tests             # both, each in its own process
pixi run tests-cov             # both, with one combined coverage report
pixi run lint                  # pre-commit hooks (ruff, ...)
pixi run type-check            # pyright
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
├── devices/                    # device classes only, no plans
│   ├── machine.py              # storage ring
│   ├── motors.py               # stages, tables, microscope optics
│   ├── shutter.py, slits.py, filters.py, dclm.py   # photon delivery system
│   ├── flyers.py               # PandA fly-scan logic
│   └── detectors/              # Phantom, GeRM, Kinetix, Perkin-Elmer
├── plans/                      # bluesky plans and plan stubs
│   ├── photon_delivery.py      # shutters, slits, filters, beam mode and energy
│   ├── motors.py               # move_motor, step trajectories
│   ├── detectors.py            # preview mode, Proc plugin and queue setup
│   ├── tomography/             # flyscans, step scans, darks/flats, alignment
│   ├── radiography.py          # radiograph series
│   ├── edxd.py                 # energy-dispersive diffraction plans
│   ├── xrd.py                  # angle-dispersive diffraction plans
│   └── settings.py             # store and apply saved device settings
├── settings/                   # saved device settings (YAML data)
└── gui/                        # the HEX Data Acquisition GUI
tests/                          # pytest suite, grouped by area
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
3. **Take supporting devices as required parameters, and fill them in with
   `functools.partial` in the profile.** These are the devices that are always
   there and the user rarely chooses (the PandA, rotation stage, shutters).
   If every device the plan takes is one of these, take them first,
   positionally. Otherwise take them last, as keyword-only parameters (after
   `*`), so the user's own devices and settings come first. The profile then
   binds them once the devices exist:

   ```python
   # src/hextools/profiles/collection.py
   change_beam_mode = partial(change_beam_mode, dclm, fe_shutter)
   take_radiograph = partial(
       take_radiograph, panda=panda, fe_shutter=fe_shutter, photon_shutter=photon_shutter
   )
   ```

   Parameters bound this way get no GUI field, and at the prompt the plan is
   called without them. Other plans and tests call the plan function itself, so
   they pass the devices explicitly.
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
"""src/hextools/plans/tomography/survey.py"""

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import bluesky.preprocessors as bpp
from ophyd_async.epics.adkinetix import KinetixDetector

from hextools.devices.motors import SampleTower
from hextools.devices.shutter import Shutter
from hextools.plans.photon_delivery import (
    ensure_shutter_closed,
    ensure_shutter_open,
)


def height_survey(
    detectors: list[KinetixDetector],
    start: float,
    stop: float,
    num: int,
    exposure_time: float,
    sample_name: str | None = None,
    use_shutter: bool = True,
    *,
    sample_tower: SampleTower,
    fe_shutter: Shutter,
    photon_shutter: Shutter,
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
    sample_tower : SampleTower
        The sample tower.
    fe_shutter : Shutter
        Front-end shutter, which must already be open.
    photon_shutter : Shutter
        Photon shutter.
    """
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
  raising inside the plan reports the error when the plan starts.
- `bp.scan` opens and closes the run. Pass your metadata through its `md`
  argument; anything in `md` is stored in the run's start document and is
  searchable in Tiled.
- The plan returns what `bp.scan` returns (the run uid), so callers composing
  plans can use it.

### Making the plan available

1. Put the plan in the module for its technique (here
   `src/hextools/plans/tomography/`), and export it from the package `__init__.py`
   if the package re-exports its plans.
2. Import it in `src/hextools/profiles/collection.py` so it exists in every
   session, and fill in its supporting devices after the devices are created:

   ```python
   from hextools.plans.tomography.survey import height_survey

   ...
   height_survey = partial(
       height_survey,
       sample_tower=sample_tower,
       fe_shutter=fe_shutter,
       photon_shutter=photon_shutter,
   )
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

from hextools.devices.motors import SampleTower
from hextools.devices.shutter import Shutter
from hextools.plans.tomography.survey import height_survey


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

    RE(
        height_survey(
            [],
            0,
            1,
            3,
            exposure_time=0.01,
            use_shutter=False,
            sample_tower=sample_tower,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
        )
    )

    start = next(doc for name, doc in docs if name == "start")
    assert start["plan_name"] == "height_survey"
    assert sum(name == "event" for name, _ in docs) == 3
```

The `RE` fixture comes from `tests/conftest.py`. Tests call the plan function,
not the profile's partial, so they pass every supporting device explicitly.
See `tests/tomography/` for fixtures that build mock shutters which
respond to open/close commands and mock Kinetix detectors with file writers.

Keep tests fast:

- **Tiled**: use the `tiled_client` fixture. It hands each test its own container
  in one Tiled server shared by the whole run (starting a server takes seconds),
  plus a data directory the server can read.
- **Waits**: a plan that settles for seconds (e.g. 4 s after a shutter moves)
  wastes that time in tests. Add `@pytest.mark.usefixtures("skip_long_sleeps")`
  to make sleeps of a second or more return at once; the fixture yields the
  lengths that were requested, if the test wants to check them.
- **Timing**: don't assert wall-clock times close to how long a step takes; a
  loaded machine runs slower. E.g. `bps.repeat` subtracts an iteration's own
  duration from its delay, so tests use a delay far longer than an iteration.

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

The GUI lives in `src/hextools/gui/`. Each technique tab (Beamline, Tomography,
EDXD, ...) is a class in `src/hextools/gui/__main__.py` with a column of plan
tabs. Each plan tab is a `QtPlanWidget`, which builds its form from the plan's
signature, so adding a plan is usually one line.

### Adding a plan tab

Import the plan at the top of `__main__.py` and add a tab to the technique's
`plan_tabs`, e.g. in `QtTomographyView.__init__`:

```python
from hextools.plans.tomography.survey import height_survey

...
plan_tabs.addTab(QtPlanWidget(re_client, tomo_flyscan), "Flyscan")
plan_tabs.addTab(QtPlanWidget(re_client, height_survey), "Height Survey")
```

`QtPlanWidget(re_client, plan, title=None)` takes the RunEngine (in-process
mode) or Queue Server client, and the plan function. The form's title defaults
to the function name in title case; pass `title=` to override it.

When the user presses **Run Plan**, the widget:

1. checks every field (required fields filled, numbers valid, devices exist and
   have the right type);
2. in-process mode: builds the call as text, e.g.
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
| A parameter bound by `functools.partial` in the profile | *No field*: the profile fills it in.                                      |
| A device class with default `None`                  | *No field*: the plan handles `None` itself.                                   |
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

To add a new technique, create a view class next to the existing ones and add
it to `QtTabbedTechniqueSelector`:

```python
class QtImagingView(QWidget):
    """Imaging view: live camera and imaging plans."""

    def __init__(self, re_client: RunEngineClient | RunEngine, *args, **kwargs):
        super().__init__(*args, **kwargs)

        top = QHBoxLayout()

        detectors = QtTabbedDetectorsWidget(re_client)
        detectors.add_detector("kinetix1", "XF:27ID1-BI{Kinetix-Det:1}")
        top.addWidget(detectors, stretch=3)

        plan_tabs = QTabWidget()
        plan_tabs.addTab(QtPlanWidget(re_client, height_survey), "Height Survey")
        top.addWidget(plan_tabs, stretch=1)

        vbox = QVBoxLayout()
        vbox.addLayout(top, stretch=1)
        self.setLayout(vbox)
```

and in `QtTabbedTechniqueSelector.__init__`:

```python
self._imaging = QtImagingView(self._re_client)
tabs.addTab(self._imaging, "Imaging")
```

`QtTabbedDetectorsWidget.add_detector(name, prefix)` adds a live viewer for the
device called `name` in the session, reading images from the PV Access prefix
`prefix`. It is only shown if the device exists, so the GUI still starts when a
detector is missing. `add_combined(label, [names])` adds a side-by-side view
of several detectors.

### Other extension points

- **Available Devices tab** — lists every device in the RunEngine client's
  device registry, with its type and connection status.
- **Scan Browser actions** — `QtScanBrowser.add_callback(spec, label, callback)`
  adds a button (and right-click action) for runs of a given spec. `callback`
  is called with the selected run's uid.
- **Top-row widgets** — `QtTabbedTechniqueSelector.__init__` builds the shared
  top row; add widgets to `controls`.
- **Menus** — `QtDataAcquisitionWindow._build_menus`.

## Updating the documentation

Generic GUI widgets must not import or branch on HEX device/plan identities,
PVs, specialized labels/defaults, or beamline policy. Configure those through
generic extension points; where none exists, defer the request. For example,
`QtShutterStatus` takes explicit readbacks and a command-builder callback;
`gui/hex_config.py` owns HEX discovery and manual-shutter settling policy.
`QtPlanWidget` has no device-specific overrides. This is a blocking
architectural boundary, not a feature-restoration mechanism.

The documentation is built with Sphinx and MyST Markdown from `docs/`. The GUI
screenshots are generated, not taken by hand; regenerate them after changing
the GUI:

```bash
QT_QPA_PLATFORM=offscreen pixi run -e dev python docs/_scripts/capture_screenshots.py
```

The script runs the profile in mock mode, opens the GUI offscreen, runs two
small plans so the execution views have content, and writes the PNGs to
`docs/_static/screenshots/`.
