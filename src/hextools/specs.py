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


def _require_dark_flat(run: BlueskyRunV3, dets: list[str]) -> None:
    """Check the tomography dark/flat convention below."""
    for stream in ["dark", "flat"]:
        if stream in run:
            _require_stream(run, stream, *dets)
        else:
            _require_start_keys(run, f"{stream}_scan_uid")


# Only detector datasets and step-scan motor readbacks are named dynamically:
# ``<det>`` is a detector's ``name`` (e.g. ``kinetix1``) and ``<motor>`` a
# stepped motor's readback field. Every stream name and every other dataset
# name is fixed.
#
# "Standard scan metadata" means the start document keys bluesky's built-in
# plans add: ``detectors``, ``motors``, ``num_points``, ``num_intervals``,
# ``plan_args``, ``plan_pattern*`` and ``hints.dimensions``.
#
# Tomography dark/flat convention: the ``dark`` and ``flat`` streams are optional.
# When a run has no ``dark`` stream, its start document must include
# ``dark_scan_uid``: the uid of the most recent run that did take darks. Likewise
# ``flat_scan_uid`` when it has no ``flat`` stream.
#   dark: dark-field images, beam off
#     <det>                     (num_images, height, width) image stack, external
#   flat: flat-field images, sample out of the beam
#     <det>                     (num_images, height, width) image stack, external


# tomo_flyscan
#
# Start document:
#   plan_name: "tomo_flyscan"
#   detectors: list[str]
#   num_points: int                  images per detector
#   exposure_time: float             seconds
#   acquire_period: float | None     seconds
#   time_based: bool
#   start_position, stop_position: float   rotation range, degrees
#   images_to_average: int
#   sample_name: str                 optional
#   dark_scan_uid: str               required if there is no dark stream
#   flat_scan_uid: str               required if there is no flat stream
# Streams:
#   dark, flat: optional, see the tomography dark/flat convention above
#   primary: one event per image, collected while flying
#     <det>                     (num_points, height, width) image stack, external
#     angle                     (num_points,) rotation angle from the PandA, degrees, external
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
    dets = _names(run, "detectors")
    _require_dark_flat(run, dets)
    _require_stream(run, "primary", *dets, "angle")


# tomo_1d_step_scan, tomo_2d_step_scan, tomo_nd_step_scan
#
# Start document:
#   plan_name: "tomo_1d_step_scan" | "tomo_2d_step_scan" | "tomo_nd_step_scan"
#   standard grid_scan metadata, incl. shape, extents, snaking
#   sample_name: str                 optional
#   dark_scan_uid: str               required if there is no dark stream
#   flat_scan_uid: str               required if there is no flat stream
# Streams:
#   dark, flat: optional, see the tomography dark/flat convention above
#   positions: one event per grid point
#     <motor>                   step axis readbacks, one field per axis
#   primary_<i>[_<j>...]: one stream per grid point, each a full flyscan as in
#     TOMO_FLYSCAN_V1. Indices are each axis's position index, in first-visited
#     order (snaked axes reuse indices).
#     <det>                     (num_images, height, width) image stack, external
#     angle                     (num_images,) rotation angle from the PandA, degrees, external
TOMO_STEP_SCAN_V1 = Spec("TomoStepScan", version="1")


def validate_tomo_step_scan(run: BlueskyRunV3, version: str | None = "1") -> None:
    """Raise :class:`SpecValidationError` if ``run`` does not match :data:`TOMO_STEP_SCAN_V1`."""
    _require_plan_name(run, "tomo_1d_step_scan", "tomo_2d_step_scan", "tomo_nd_step_scan")
    _require_start_keys(run, "shape", "extents", "snaking")
    dets = _names(run, "detectors")
    motors = _names(run, "motors")
    _require_dark_flat(run, dets)
    _require_stream(run, "positions", *motors)
    point_streams = [name for name in run if name.startswith("primary_")]
    if not point_streams:
        raise SpecValidationError("Run has no 'primary_<i>' streams.")
    for stream in point_streams:
        _require_stream(run, stream, *dets, "angle")


# tomo_alignment_scan
#
# Start document:
#   plan_name: "tomography_alignment_scan"
#   description: "Tomography alignment scan"
#   standard scan metadata (motors: [<motor>], detectors: [<det>])
#   flat_scan_uid: str               optional. If set, use flat stream from different run.
# Streams:
#   flat: optional, see the tomography dark/flat convention above
#   primary: one event per projection angle
#     <det>                     (height, width) projection image, external
#     <motor>                   rotation angle readback, degrees
TOMO_ALIGNMENT_SCAN_V1 = Spec("TomoAlignmentScan", version="1")


def validate_tomo_alignment_scan(run: BlueskyRunV3, version: str | None = "1") -> None:
    """Raise :class:`SpecValidationError` if ``run`` does not match :data:`TOMO_ALIGNMENT_SCAN_V1`."""
    _require_plan_name(run, "tomography_alignment_scan")
    _require_start_keys(run, "description")
    dets = _names(run, "detectors", count=1)
    motors = _names(run, "motors", count=1)
    if "flat" in run:
        _require_stream(run, "flat", *dets)
    _require_stream(run, "primary", *dets, *motors)


# take_radiograph
#
# Start document:
#   plan_name: "take_radiograph"
#   sample_name: str                 optional
#   description: str                 optional
#   standard count metadata (no motors)
# Streams:
#   primary: one event per acquisition
#     <det>                     (num_images, height, width) images, each the sum of
#                               num_exposures exposures, external
RADIOGRAPH_V1 = Spec("Radiograph", version="1")


def validate_radiograph(run: BlueskyRunV3, version: str | None = "1") -> None:
    """Raise :class:`SpecValidationError` if ``run`` does not match :data:`RADIOGRAPH_V1`."""
    _require_plan_name(run, "take_radiograph")
    _require_stream(run, "primary", *_names(run, "detectors"))


# edxd_count, edxd_scan, edxd_grid_scan, edxd_custom_pos_list_grid
#
# Start document:
#   plan_name: "edxd_count" | "edxd_scan" | "edxd_grid_scan" | "edxd_custom_pos_list_grid"
#   description: str                 optional
#   standard count/scan/grid_scan metadata (0, 1 or 2 motors)
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
#     <det>-stats1-total        float, total screen intensity used to find the peak
#     <motor>                   crystal 2 pitch readback, degrees
ENERGY_AUTO_TUNE_V1 = Spec("EnergyAutoTune", version="1")


def validate_energy_auto_tune(run: BlueskyRunV3, version: str | None = "1") -> None:
    """Raise :class:`SpecValidationError` if ``run`` does not match :data:`ENERGY_AUTO_TUNE_V1`."""
    _require_plan_name(run, "energy_auto_tune")
    dets = _names(run, "detectors")
    motors = _names(run, "motors", count=1)
    _require_stream(run, "primary", *dets, *(f"{det}-stats1-total" for det in dets), *motors)


_VALIDATORS: dict[str, Callable[[BlueskyRunV3, str | None], None]] = {
    spec.name: validator
    for spec, validator in [
        (TOMO_FLYSCAN_V1, validate_tomo_flyscan),
        (TOMO_STEP_SCAN_V1, validate_tomo_step_scan),
        (TOMO_ALIGNMENT_SCAN_V1, validate_tomo_alignment_scan),
        (RADIOGRAPH_V1, validate_radiograph),
        (EDXD_SCAN_V1, validate_edxd_scan),
        (EDXD_CALIBRATION_V1, validate_edxd_calibration),
        (XRD_CALIBRATION_V1, validate_xrd_calibration),
        (ENERGY_AUTO_TUNE_V1, validate_energy_auto_tune),
    ]
}


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
