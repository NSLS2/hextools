"""Collection of tiled specs used to denote the various types of datasets produced by bluesky scans at HEX beamline."""

from tiled.structures.core import Spec

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


# tomo_alignment_scan
#
# Start document:
#   plan_name: "tomography_alignment_scan"
#   description: "Tomography alignment scan"
#   standard scan metadata (motors: [rotation motor])
#   dark_scan_uid: str               required if there is no dark stream
#   flat_scan_uid: str               required if there is no flat stream
# Streams:
#   dark, flat: optional, see the tomography dark/flat convention above
#   primary: one event per projection angle
#     <det>                     (height, width) projection image, external
#     <motor>                   rotation angle readback, degrees
TOMO_ALIGNMENT_SCAN_V1 = Spec("TomoAlignmentScan", version="1")


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
