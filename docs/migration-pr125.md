# Breaking changes after PR #125

These are intentional removals and API changes, not compatibility aliases.
Existing scripts and operator habits must be updated; this follow-up does not
restore the removed features.

## Removed and renamed interfaces

- `set_fe_slit1_aperture` is now `set_fe_aperture`. `FrontEndSlit` and
  `SlitAperture` live in `hextools.photon_delivery_system.slits`, not `fe_slit`.
  The package-level exports expose the current names.
- `PPSShutter.interlock` is now `interlock_ack`. For example,
  `RE(bps.mv(fe_shutter.interlock_ack, True))`.
- `load_filters` and the old `photon_delivery_system/filters.yml` configuration
  are removed. Filters are constructed explicitly by the profile; their
  constants live in `src/hextools/bl_constants.yml`. The unused
  `schemas/filters.yml`, which described the removed PV-and-positions format,
  is removed too; it did not validate the current constants.
- `set_filters` material strings are case-sensitive enum values, including
  **`"Pass Through"`**, not `"pass through"`. Only selected filters need to be
  available. All material selections are validated before device lookup,
  shutter reads or motion.
- `Filter` now requires a keyword-only `settings_key`, independent of `name`:
  `Filter(motor_pv, switch_pv, name="operator-label", settings_key="filter2")`.
  The selected constants must contain exactly `upper_limit`, `pass_through`,
  `pos_1`, `pos_2`, `pos_3`, `pos_4`, and `lower_limit`. Each is either a finite
  number or a mapping with `position` and optional string `description`.
- DCLM constants are now on the frozen `dclm.config` (`DCLMConfig`) rather than
  individual properties, e.g. `dclm.config.bragg_factor`. Missing/unknown keys
  fail construction with `TypeError`; invalid numeric values raise `ValueError`.
  The numeric constants from PR #125 are unchanged. The YAML has three top-level
  mappings: `monochromator`, `filters`, and `fe_aperture`.

## Apertures, labels and exceptions

Current aperture string values are **`"Fully open (24 x 4.55 mm)"`**,
**`"Large (7.13 x 3.56 mm)"`**, **`"Fully closed"`**,
**`"Small (3.56 x 3.56 mm)"`**, and **`"Pinhole (0.89 mm)"`**.
The former dimension-only large/small strings and `"Pinhole 0.89 mm"` are not
accepted. Use `SlitAperture` members instead of embedding strings in scripts.

`FrontEndSlit.aperture` reports the nearest aperture even between positions.
Use `in_position` to test whether it is actually aligned (strictly less than
0.01 mm away). The old `"between apertures"` sentinel and configurable
`tolerance` constructor argument are removed.

`ensure_shutter_state` (and its open/closed wrappers) now raises `ValueError`,
not `RuntimeError`, when a shutter is in the wrong state and actuation is
forbidden. Update exception handling. Invalid enum strings raise `ValueError`;
errors during a device `set` executed through the RunEngine may be wrapped in
Bluesky `FailedStatus`, so inspect its cause.

The `HEXBlueskyMagics` class and its registration, including the custom `%mov`
and `%movr` implementations, are removed with `hextools.utils.magics`.
Use supported plans and device signals; this package does not register
compatibility magics. The removed Sample Small
display-name overrides remain removed. Energy auto-tuning still fits a peak
but no longer creates the removed tuning plot; the unused DCLM figure-factory
helper and its obsolete plotter comment are removed.

Operator requests 5 (prefilled slit readbacks), 9 (special **Two theta** label),
and 11 (TIFF checked by default) remain deferred. The generic plan widget uses
signature-derived labels and defaults, with no device-specific branches.

## Timing is not changed by a profile variable

The former profile-local `DEFAULT_TIMEOUT = 60.0` did not configure ophyd-async.
It has been removed rather than turned into a global timeout change. Shutter
status and interlock-ack waits use ophyd-async's default, **10 seconds in this
lockfile**. Manual shutter buttons retain mutual exclusion and request
`sleep_time=0.0`; acquisition plans retain their normal four-second settling
delay when they actuate shutters. No hardware timings have been measured here.
A slow-shutter timing policy needs separate operator measurements and review,
not an unverified global increase.
