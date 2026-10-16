"""Collection of tiled specs used to denote the various types of datasets produced by bluesky scans at HEX beamline."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from tiled.structures.core import Spec

if TYPE_CHECKING:
    from bluesky_tiled_plugins.clients.bluesky_run import BlueskyRunV3


class SpecValidationError(ValueError):
    """A run does not match the spec it is validated against."""


def _require_plan_name(run: BlueskyRunV3, *names: str) -> None:
    plan_name = run.start.get("plan_name")
    if plan_name not in names:
        raise SpecValidationError(f"plan_name is {plan_name!r}, expected one of {list(names)}.")


def _require_start_keys(run: BlueskyRunV3, *keys: str) -> None:
    missing = [key for key in keys if key not in run.start]
    if missing:
        raise SpecValidationError(f"Start document is missing {missing}.")


def _names(run: BlueskyRunV3, key: str, count: int | None = None) -> list[str]:
    """Return the start document's list of names under ``key`` (``detectors``/``motors``)."""
    names: Any = run.start.get(key)
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise SpecValidationError(f"Start document '{key}' must be a list of names, got {names!r}.")
    if count is not None and len(names) != count:
        raise SpecValidationError(f"Expected {count} name(s) in '{key}', got {names}.")
    return names


def _require_stream(run: BlueskyRunV3, stream: str, *fields: str) -> None:
    if stream not in run:
        raise SpecValidationError(f"Run has no '{stream}' stream.")
    missing = [field for field in fields if field not in run[stream]]
    if missing:
        raise SpecValidationError(f"Stream '{stream}' is missing {missing}.")


def _require_optional_str(run: BlueskyRunV3, key: str) -> None:
    value = run.start.get(key)
    if value is not None and not isinstance(value, str):
        raise SpecValidationError(f"Start document '{key}' must be a string, got {value!r}.")


# Only detector datasets and motor readbacks are named dynamically: ``<det>`` is a
# detector's ``name`` (e.g. ``kinetix1``) and ``<motor>`` a scanned motor's readback
# field. Every stream name and every other dataset name is fixed.
#
# "Standard scan metadata" means the start document keys bluesky's built-in
# plans add: ``detectors``, ``motors``, ``num_points``, ``num_intervals``,
# ``plan_args``, ``plan_pattern*`` and ``hints.dimensions``.
#
# Every run also has a ``baseline`` stream from the profile's SupplementalData;
# it is not part of any spec below.
#
# Tomography dark/flat convention: darks and flats are never part of a tomography
# run. They are taken by a separate ``tomo_dark_flat`` run (TOMO_DARK_FLAT_V1).
# The profile's LastDarkFlatUIDTracker copies that run's uid and scan_id into
# ``RE.md``, so every later run's start document carries ``last_dark_flat_uid``
# and ``last_dark_flat_scan_id`` until the next dark/flat replaces them. Both keys
# are absent if no dark/flat has been taken in the session.


# tomo_dark_flat
#
# Start document:
#   plan_name: "tomo_dark_flat"
#   exposure_time: float             seconds
#   num_dark_images: int
#   num_flat_images: int
#   x_offset: float                  sample tower x1 offset for the flats
#   no standard scan metadata: the run is opened directly, not by a built-in plan
# Streams:
#   dark: a single event, beam off
#     <det>                     (num_dark_images, height, width) image stack, external
#   flat: a single event, sample moved out of the beam by x_offset
#     <det>                     (num_flat_images, height, width) image stack, external
TOMO_DARK_FLAT_V1 = Spec("TomoDarkFlat", version="1")


def validate_tomo_dark_flat(run: BlueskyRunV3, version: str | None = "1") -> None:
    """Raise :class:`SpecValidationError` if ``run`` does not match :data:`TOMO_DARK_FLAT_V1`."""
    _require_plan_name(run, "tomo_dark_flat")
    _require_start_keys(run, "exposure_time", "num_dark_images", "num_flat_images", "x_offset")
    _require_stream(run, "dark")
    # No ``detectors`` key here, so the dark stream defines the detectors.
    dets = [field for field in run["dark"] if not field.startswith("ts_") and field != "seq_num"]
    if not dets:
        raise SpecValidationError("Stream 'dark' has no detector fields.")
    _require_stream(run, "flat", *dets)


# tomo_flyscan (also the per-tomogram run of tomo_flyscan_loop and the
# tomo_1d/2d/nd_step_scan plans, which open one run per tomogram)
#
# Start document:
#   plan_name: "tomo_flyscan"
#   detectors: list[str]
#   num_points: int                  projections per detector
#   exposure_time: float             seconds
#   acquire_period: float | None     seconds
#   time_based: bool
#   start_position, stop_position: float   rotation range, degrees
#   images_to_average: int
#   sample_name: str                 optional
#   last_dark_flat_uid: str          optional, see the tomography dark/flat convention
#   last_dark_flat_scan_id: int      optional, see the tomography dark/flat convention
# Streams:
#   primary: collected while flying
#     <det>                     (num_points, height, width) image stack, external
#     rotation_angle            rotation angle from the PandA, degrees, external. Other
#                               PandA captures enabled in its settings also appear here.
TOMO_FLYSCAN_V1 = Spec("TomoFlyscan", version="1")


def validate_tomo_flyscan(run: BlueskyRunV3, version: str | None = "1") -> None:
    """Raise :class:`SpecValidationError` if ``run`` does not match :data:`TOMO_FLYSCAN_V1`."""
    _require_plan_name(run, "tomo_flyscan")
    _require_start_keys(
        run,
        "num_points",
        "exposure_time",
        "acquire_period",
        "time_based",
        "start_position",
        "stop_position",
        "images_to_average",
    )
    _require_optional_str(run, "last_dark_flat_uid")
    _require_stream(run, "primary", *_names(run, "detectors"), "rotation_angle")


# tomo_alignment_scan
#
# Start document:
#   plan_name: "tomography_alignment_scan"
#   description: "Tomography alignment scan"
#   standard scan metadata (motors: [<motor>], detectors: [<det>])
#   flat_scan_uid: str               optional, user supplied. The alignment analysis
#                                    uses this run's flat stream if the scan has none.
# Streams:
#   flat: optional; the plan never writes it, but the analysis reads it if present
#     <det>                     (num_images, height, width) image stack, external
#   primary: one event per projection angle
#     <det>                     (height, width) projection image, external
#     <motor>                   rotation angle readback, degrees
TOMO_ALIGNMENT_SCAN_V1 = Spec("TomoAlignmentScan", version="1")


def validate_tomo_alignment_scan(run: BlueskyRunV3, version: str | None = "1") -> None:
    """Raise :class:`SpecValidationError` if ``run`` does not match :data:`TOMO_ALIGNMENT_SCAN_V1`."""
    _require_plan_name(run, "tomography_alignment_scan")
    _require_start_keys(run, "description")
    _require_optional_str(run, "flat_scan_uid")
    dets = _names(run, "detectors", count=1)
    motors = _names(run, "motors", count=1)
    if "flat" in run:
        _require_stream(run, "flat", *dets)
    _require_stream(run, "primary", *dets, *motors)


# take_radiograph (one run per iteration, each a bluesky count with num=1)
#
# Start document:
#   plan_name: "take_radiograph"
#   sample_name: str                 optional
#   description: str                 optional
#   standard count metadata (no motors)
# Streams:
#   primary: a single event
#     <det>                     (num_images, height, width) images, each combining
#                               images_to_average exposures, external
RADIOGRAPH_V1 = Spec("Radiograph", version="1")


def validate_radiograph(run: BlueskyRunV3, version: str | None = "1") -> None:
    """Raise :class:`SpecValidationError` if ``run`` does not match :data:`RADIOGRAPH_V1`."""
    _require_plan_name(run, "take_radiograph")
    _require_stream(run, "primary", *_names(run, "detectors"))


# edxd_count, edxd_scan, edxd_grid_scan, edxd_custom_pos_list_grid
# (edxd_scan opens one run per iteration; edxd_custom_pos_list_grid one run per
# outer position)
#
# Start document:
#   plan_name: "edxd_count" | "edxd_scan" | "edxd_grid_scan" | "edxd_custom_pos_list_grid"
#   description: str                 optional
#   standard count/scan/grid_scan/list_scan metadata (0, 1 or 2 motors)
#   outer_position: float            edxd_custom_pos_list_grid only
#   initial_scan_uid: str            edxd_custom_pos_list_grid only, from its second run on
# Streams:
#   primary: one event per point
#     <det>                     (num_elements, num_energy_bins) GeRM spectra,
#                               external HDF5 or TIFF
#     <det>-stats1-total        float, total counts
#     <motor>                   scanned axis readbacks, one field per axis
EDXD_SCAN_V1 = Spec("EdxdScan", version="1")


def validate_edxd_scan(run: BlueskyRunV3, version: str | None = "1") -> None:
    """Raise :class:`SpecValidationError` if ``run`` does not match :data:`EDXD_SCAN_V1`."""
    _require_plan_name(run, "edxd_count", "edxd_scan", "edxd_grid_scan", "edxd_custom_pos_list_grid")
    dets = _names(run, "detectors")
    motors = _names(run, "motors") if "motors" in run.start else []
    if len(motors) > 2:
        raise SpecValidationError(f"Expected at most 2 motors, got {motors}.")
    _require_stream(run, "primary", *dets, *(f"{det}-stats1-total" for det in dets), *motors)


# edxd_calib_scan
#
# Start document:
#   plan_name: "edxd_calib_scan"
#   description: str                 optional
#   no standard scan metadata: the run is opened directly, not by a built-in plan
# Streams:
#   primary: a single event, read once the GeRM count finishes while the sample
#            tower z1 sweeps between start and stop
#     <det>                     (num_elements, num_energy_bins) GeRM spectra, external
#     <det>-stats1-total        float, total counts
EDXD_CALIBRATION_V1 = Spec("EdxdCalibration", version="1")


def validate_edxd_calibration(run: BlueskyRunV3, version: str | None = "1") -> None:
    """Raise :class:`SpecValidationError` if ``run`` does not match :data:`EDXD_CALIBRATION_V1`."""
    _require_plan_name(run, "edxd_calib_scan")
    _require_stream(run, "primary")
    # No ``detectors`` key here, so find detectors through their stats field.
    suffix = "-stats1-total"
    dets = [field.removesuffix(suffix) for field in run["primary"] if field.endswith(suffix)]
    if not dets:
        raise SpecValidationError(f"Stream 'primary' has no '<det>{suffix}' field.")
    _require_stream(run, "primary", *dets)


# xrd_calibration
#
# Start document:
#   plan_name: "xrd_calibration"
#   description: str                 default "Energy-geometry calibration"
#   standard scan metadata (motors: [detector distance motor])
# Streams:
#   primary: one event per detector position (num_steps, spaced by gap)
#     <det>                     (height, width) area detector image, external
#     <motor>                   detector position readback
XRD_CALIBRATION_V1 = Spec("XrdCalibration", version="1")


def validate_xrd_calibration(run: BlueskyRunV3, version: str | None = "1") -> None:
    """Raise :class:`SpecValidationError` if ``run`` does not match :data:`XRD_CALIBRATION_V1`."""
    _require_plan_name(run, "xrd_calibration")
    _require_start_keys(run, "description")
    motors = _names(run, "motors", count=1)
    _require_stream(run, "primary", *_names(run, "detectors"), *motors)


# energy_auto_tune (run twice by change_energy(auto_tune=True): coarse, then fine)
#
# Start document:
#   plan_name: "energy_auto_tune"
#   standard scan metadata (motors: [monochromator crystal 2 pitch])
# Streams:
#   primary: one event per pitch step
#     <det>                     (height, width) fluorescence screen image, external
#     <det>-stats1-mean_value   float, mean screen intensity used to find the peak
#     <motor>                   crystal 2 pitch readback, degrees
ENERGY_AUTO_TUNE_V1 = Spec("EnergyAutoTune", version="1")


def validate_energy_auto_tune(run: BlueskyRunV3, version: str | None = "1") -> None:
    """Raise :class:`SpecValidationError` if ``run`` does not match :data:`ENERGY_AUTO_TUNE_V1`."""
    _require_plan_name(run, "energy_auto_tune")
    dets = _names(run, "detectors")
    motors = _names(run, "motors", count=1)
    _require_stream(run, "primary", *dets, *(f"{det}-stats1-mean_value" for det in dets), *motors)


_VALIDATORS: dict[str, Callable[[BlueskyRunV3, str | None], None]] = {
    spec.name: validator
    for spec, validator in [
        (TOMO_DARK_FLAT_V1, validate_tomo_dark_flat),
        (TOMO_FLYSCAN_V1, validate_tomo_flyscan),
        (TOMO_ALIGNMENT_SCAN_V1, validate_tomo_alignment_scan),
        (RADIOGRAPH_V1, validate_radiograph),
        (EDXD_SCAN_V1, validate_edxd_scan),
        (EDXD_CALIBRATION_V1, validate_edxd_calibration),
        (XRD_CALIBRATION_V1, validate_xrd_calibration),
        (ENERGY_AUTO_TUNE_V1, validate_energy_auto_tune),
    ]
}

#: Every spec defined here.
ALL_SPECS: list[Spec] = [
    TOMO_DARK_FLAT_V1,
    TOMO_FLYSCAN_V1,
    TOMO_ALIGNMENT_SCAN_V1,
    RADIOGRAPH_V1,
    EDXD_SCAN_V1,
    EDXD_CALIBRATION_V1,
    XRD_CALIBRATION_V1,
    ENERGY_AUTO_TUNE_V1,
]


def validate_run(run: BlueskyRunV3, spec: Spec) -> None:
    """Validate ``run`` against ``spec``.

    Raises
    ------
    SpecValidationError
        If the run does not match the spec.
    KeyError
        If there is no validator for ``spec``'s name.
    """
    try:
        validator = _VALIDATORS[spec.name]
    except KeyError:
        raise KeyError(f"No validator for spec {spec.name!r}.") from None
    validator(run, spec.version)
