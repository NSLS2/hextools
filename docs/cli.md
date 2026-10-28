# Running plans from the command line

The command-line interface is an IPython session that runs the
[collection profile](profile.md). Everything the GUI does can also be done
here, and some things (ad-hoc scans, custom analysis, debugging) are only
possible here.

## Starting a session

```bash
pixi run start          # real hardware
pixi run start-mock     # simulated devices, local Tiled server
```

`start` accepts two arguments: `mock` (`true`/`false`) and `profile` (the name
of a file in `src/hextools/profiles/`, default `collection`), e.g.
`pixi run start true collection`.

On startup the profile prints the package versions, the active proposal and a
connection report for every device:

```text
Version Information
  bluesky: 1.15.2.dev230+g1d0640485
  ophyd_async: 0.21.3.dev48+gc831a174a
  tiled: 0.2.18
  hextools: 0.2.4

Active proposal:
  Proposal title: Mock mode proposal
  Proposal type: Mock Commissioning
  Proposal PI: jwlodek

Connecting to beamline devices...

  fe_shutter .............................. [   OK   ]
  photon_shutter .......................... [   OK   ]
  sample_tower ............................ [ FAILED ] (12/14 signals connected)
  ...

pass-123456 [1]:
```

- `[   OK   ]` means every signal of the device connected.
- `[ FAILED ]` means the device could not be fully connected. When only some
  of its signals connected, the count is shown in parentheses. The session
  still starts, and the failure reasons are printed below the list; plans that
  need a failed device will raise an error when you try to use it.

The prompt shows the active proposal (`pass-123456`). Every run you take is
recorded under that proposal.

## The RunEngine

Plans are Python generators. Calling a plan function builds the generator but
does nothing on its own; you execute it by passing it to the RunEngine, `RE`:

```python
RE(move_motor(sample_tower.x1, 12.5))
```

`RE(...)` blocks until the plan finishes and returns the uids of the runs it
created. Data from runs is written to Tiled as it is collected.

### Interrupting a plan

| Keys                        | Effect                                                                      |
| --------------------------- | --------------------------------------------------------------------------- |
| `Ctrl+C` once               | Deferred pause: the plan pauses at the next checkpoint.                     |
| `Ctrl+C` twice (quickly)    | Immediate pause: the plan pauses now.                                       |

Once paused, choose how to continue:

```python
RE.resume()   # carry on from the last checkpoint
RE.stop()     # end the plan, marking the run as successful
RE.abort()    # end the plan, marking the run as aborted
RE.halt()     # end immediately, skipping cleanup (emergencies only)
```

If the storage-ring current drops below 100 mA, the RunEngine suspends the
plan automatically and resumes once it recovers above 390 mA.

## Beamline setup plans

### Shutters

The front-end (`fe_shutter`) and photon (`photon_shutter`) shutters are
opened and closed with plan stubs that check the state before acting:

```python
from hextools.photon_delivery_system import ensure_shutter_open, ensure_shutter_closed

RE(ensure_shutter_open([fe_shutter, photon_shutter], allow_actuation=True))
RE(ensure_shutter_closed(photon_shutter))
```

`ensure_shutter_open` only actuates a shutter if `allow_actuation=True`;
otherwise it raises if the shutter is not already open. This protects plans
from silently opening a shutter.

### Beam mode and energy

The double crystal Laue monochromator (`dclm`, also available as `mono` and
`monochromator`) switches between white and monochromatic beam:

```python
RE(change_beam_mode("White"))
RE(change_beam_mode("Monochromatic"))

# Monochromatic mode only. auto_tune=True scans crystal 2 pitch to peak the
# intensity on the fluorescence screen camera.
RE(change_energy(70.0, auto_tune=True))
```

### Motors and filters

```python
RE(move_motor(sample_tower.ry2, 90))          # single motor, absolute
RE(bps.mv(sample_tower.x1, 1, sample_tower.y, 2))   # several motors at once
RE(bps.mvr(sample_tower.x1, 0.5))             # relative move
```

Slits (`a_slits`, `f_slits`, `edxd_slits`) move by the center line and gap of each
blade pair; only the values given move, and a center move keeps the gap:

```python
RE(move_slits(a_slits, horizontal_center=0.5))                  # shift the pair
RE(move_slits(a_slits, horizontal_gap=2.0, vertical_gap=1.0))   # open about the center
RE(move_slits(f_slits, vertical_center=-0.1, relative=True))    # nudge by 0.1 mm
```

Front-end Slit1 has fixed apertures chosen by its Y axis (`fe_slit1`):

```python
RE(set_fe_slit1_aperture("Pinhole 0.89 mm"))
RE(set_fe_slit1_aperture(SlitAperture.FULLY_OPEN))   # 24 x 4.55 mm
```

Every filter defined in the filter configuration file is placed in the
namespace by name (e.g. `filter1_upstream`) and can be moved with `bps.mv`.
Filter 1 upstream, filter 1 downstream and filter 2 can be set together by material;
a filter left out does not move:

```python
RE(set_filters(filter1_upstream_setting="6 mm Cu", filter2_setting="1 mm Cu"))
RE(set_filters(filter1_downstream_setting="pass through"))
```

To see what a device contains, print its tree:

```python
from hextools.utils import print_device_tree
print_device_tree(sample_tower)
```

## Acquisition plans

All acquisition plans take their supporting devices (PandA, rotation stage,
shutters, ...) as optional keyword arguments. If you leave them out, the plan
looks up the device of the same name in the session namespace, so the short
form below is usually all you need.

### Tomography

```python
# One flyscan: rotate 0 -> 180 degrees while kinetix1 takes 1800 images
RE(tomo_flyscan([kinetix1], num_images=1801, exposure_time=0.01, sample_name="steel_rod"))

# Repeat the flyscan at 5 heights
RE(tomo_1d_step_scan([kinetix1], sample_tower.z1, 0, 4, 5,
                     num_images=1801, exposure_time=0.01))

# Repeat over a 2D grid
RE(tomo_2d_step_scan([kinetix1], sample_tower.x1, -1, 1, 3, sample_tower.z1, 0, 4, 5,
                     num_images=1801, exposure_time=0.01))

# Rotation-axis alignment scan: 37 projections over 360 degrees
RE(tomo_alignment_scan(kinetix1, exposure_time=0.01))
```

With the default `use_shutter=True`, the flyscan plans first check that the
front-end shutter is already open (they will not open it for you) and then open
the photon shutter for the acquisition.

### Radiography

```python
from hextools.tomography.radiography import take_radiograph

RE(take_radiograph([kinetix1], exposure_time=0.005, num_images=100))
```

### EDXD

The energy-dispersive plans use the GeRM detector (`germ`):

```python
RE(edxd_count(count_time=10, num=3, description="background"))
RE(edxd_scan(...))          # 1D scan
RE(edxd_grid_scan(...))     # 2D grid
```

Use `edxd_count?` (or any plan name followed by `?`) to print its full
documentation, including every parameter.

### Built-in bluesky plans

The standard bluesky plans are available as `bp` (plans) and `bps` (plan
stubs) and work with any HEX device:

```python
RE(bp.count([kinetix1], num=10))
RE(bp.scan([fs_window_cam], dclm.xtal2_pitch, -0.05, 0.05, 41))
RE(bp.grid_scan([germ], sample_tower.x1, 0, 1, 11, sample_tower.z1, 0, 1, 11))
```

The simulated motors `sim_x`, `sim_y` and `sim_z` are always available and are
useful for checking a plan's logic without moving anything:

```python
RE(bp.scan([], sim_x, 0, 2, 5))
```

## Proposals

The active proposal is stored in the RunEngine metadata (`RE.md`). To start or
switch to a beamtime:

```python
from hextools.utils import start_beamtime
start_beamtime(123456)    # prompts for your BNL username and password
```

This runs NSLS-II's `sync-experiment`, which checks that you are allowed to
take data on the proposal and updates the shared beamline metadata.

## Reading data back

In an interactive session the profile creates a Tiled client for the HEX raw
catalog, available as `c` (and `tiled_reading_client`):

```python
run = c[-1]                          # most recent run
run.start                            # start document (metadata)
run["primary"]["kinetix1"].read()    # an image stack, as a numpy array
c.search(...)                        # search by metadata
```
