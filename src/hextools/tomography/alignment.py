"""Tomography alignment tools for HEX beamline."""

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import algotom.util.calibration as calib
from bluesky_tiled_plugins import CatalogOfBlueskyRuns
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from bluesky import plan_stubs as bps, plans as bp, preprocessors as bpp
from bluesky_tiled_plugins.clients.bluesky_run import BlueskyRunV3
from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor
from skimage import measure, segmentation
from skimage.measure._regionprops import RegionProperties

from hextools.detectors.phantom import PhantomDetector
from hextools.motors import RotationMotor
from hextools.photon_delivery_system import Shutter
from hextools.photon_delivery_system.shutter import ShutterStatus, ensure_shutter_closed, ensure_shutter_open
from hextools.utils import ensure_available
from hextools.specs import TOMO_ALIGNMENT_SCAN_V1


Image = np.ndarray[tuple[int, int], np.dtype[np.uint16] | np.dtype[np.uint8]]
BinaryImage = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
ImageDataset = np.ndarray[
    tuple[int, int, int], np.dtype[np.uint16] | np.dtype[np.uint8]
]
FloatImage = np.ndarray[tuple[int, int], np.dtype[np.float64]]
FloatImageDataset = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
#: Sphere center coordinates, one per projection.
Points = np.ndarray[tuple[int], np.dtype[np.float32]]


def ensure_run_is_valid(
    run: BlueskyRunV3,
    det_names: list[str],
    motor_name: str,
    proj_stream: str = "primary",
    ff_stream: str | None = None,
) -> None:
    """Check that a run holds the streams and datasets needed for alignment analysis.

    Parameters
    ----------
    run : BlueskyRunV3
        The run to check.
    det_names : list[str]
        Detectors whose images must be in every checked stream.
    motor_name : str
        Rotation motor whose readback must be in the projection stream.
    proj_stream : str, optional
        Stream holding the projections, by default "primary".
    ff_stream : str | None, optional
        Stream holding flat-field images, if it must be checked too.

    Raises
    ------
    KeyError
        If a stream, detector, or motor dataset is not available.
    """

    def _check_stream(stream_name: str, requires_motor: bool):
        if stream_name not in run:
            raise KeyError(f"Stream '{stream_name}' not found in the run.")
        data_stream = run[stream_name]
        for det_name in det_names:
            if det_name not in data_stream:
                raise KeyError(
                    f"Detector '{det_name}' not found in the stream '{stream_name}'."
                )
        if requires_motor and motor_name not in data_stream:
            raise KeyError(f"Motor '{motor_name}' not found in the stream '{stream_name}'.")

    _check_stream(proj_stream, requires_motor=True)
    if ff_stream is not None:
        _check_stream(ff_stream, requires_motor=False)


def check_crop_values_valid(
    projection_width: int,
    projection_height: int,
    left_crop: int,
    right_crop: int,
    top_crop: int,
    bottom_crop: int,
) -> bool:
    """Check if the crop values are valid for the given projection dimensions."""
    if left_crop < 0 or right_crop < 0 or top_crop < 0 or bottom_crop < 0:
        raise ValueError("Crop values must be non-negative.")
    if left_crop + right_crop >= projection_width:
        raise ValueError(
            "The sum of left and right crops must be less than the projection width."
        )
    if top_crop + bottom_crop >= projection_height:
        raise ValueError(
            "The sum of top and bottom crops must be less than the projection height."
        )
    return True


def clean_image(binary_image: BinaryImage, size_threshold: int = 100) -> BinaryImage:
    """Clean binary image."""
    # Clear objects connected to the border and fill holes
    binary_image = segmentation.clear_border(binary_image)
    binary_image = cast(BinaryImage, ndi.binary_opening(binary_image, iterations=2))
    binary_image = np.asarray(ndi.binary_fill_holes(binary_image), dtype=np.bool_)

    # Label connected regions in the binary image
    label_image = measure.label(binary_image)
    properties: list[RegionProperties] = measure.regionprops(label_image)

    # Initialize mask to keep objects larger than the size threshold
    size_mask = np.zeros_like(binary_image, dtype=bool)

    # Filter objects based on size
    for prop in properties:
        if prop.area >= size_threshold:
            size_mask[label_image == prop.label] = True
    filtered_image = np.logical_and(binary_image, size_mask)
    return filtered_image


class TomoAlignMethod(StrEnum):
    """Methods for tomography alignment."""

    ELLIPSE = "ellipse"
    LINEAR = "linear"


def crop_and_flatfield_correction(
    projection_data: ImageDataset,
    flatfield: FloatImage | None = None,
    top_crop: int = 500,
    bottom_crop: int = 500,
    left_crop: int = 0,
    right_crop: int = 0,
    ratio: float = 1.0,
    figsize: tuple[int, int] = (14, 7),
) -> tuple[FloatImageDataset, Points, Points]:
    """Crop and flat-field correct each projection, and find the sphere's center in it.

    Parameters
    ----------
    projection_data : ImageDataset
        The 3D array of projection images to be processed.
    flatfield : FloatImage, optional
        The flat-field image used for correction. If None, no correction is applied.
    top_crop : int
        Number of pixels to crop from the top of each image.
    bottom_crop : int
        Number of pixels to crop from the bottom of each image.
    left_crop : int
        Number of pixels to crop from the left of each image.
    right_crop : int
        Number of pixels to crop from the right of each image.
    ratio : float
        Ratio for thresholding during binarization.
    figsize : tuple[int, int]
        Size of the figure shown when no sphere can be found.

    Returns
    -------
    tuple[FloatImageDataset, Points, Points]
        The corrected images, and the sphere's x and y centers in each (y measured
        upwards from the bottom of the cropped image).

    Raises
    ------
    ValueError
        If no sphere can be found in a projection.
    """
    cropped_and_normalized: list[FloatImage] = []
    x_centers: list[float] = []
    y_centers: list[float] = []
    for i, proj_img in enumerate(projection_data):
        bottom = proj_img.shape[0] - bottom_crop
        right = proj_img.shape[1] - right_crop
        cropped = np.asarray(proj_img[top_crop:bottom, left_crop:right], dtype=np.float64)
        if flatfield is not None:
            cropped = cropped / flatfield[top_crop:bottom, left_crop:right]
        # Denoise
        mat = cast(FloatImage, ndi.gaussian_filter(cropped, 5))
        threshold = calib.calculate_threshold(mat, bgr="bright")
        mat_bin0 = cast(
            BinaryImage,
            np.asarray(
                calib.binarize_image(mat, threshold=ratio * threshold, bgr="bright"),
                dtype=np.bool_,
            ),
        )
        mat_bin0 = clean_image(mat_bin0)
        if np.sum(mat_bin0) < 20.0:
            # Show the image so the field of view or ratio can be adjusted.
            plt.figure(figsize=figsize)
            plt.imshow(mat, cmap="gray")
            plt.show()
            raise ValueError(
                f"No sphere detected in projection {i} (ratio {ratio}, threshold "
                f"{threshold}). Adjust the ratio or the field of view."
            )
        # Keep the sphere only
        sphere_size = calib.get_dot_size(mat_bin0, size_opt="max")
        mat_bin = calib.select_dot_based_size(mat_bin0, sphere_size)
        y_cen, x_cen = np.asarray(ndi.center_of_mass(mat_bin), dtype=np.float64)
        x_centers.append(float(x_cen))
        y_centers.append(mat.shape[0] - float(y_cen))
        cropped_and_normalized.append(mat)
    images: FloatImageDataset = np.stack(cropped_and_normalized).astype(np.float64)
    return (
        images,
        np.asarray(x_centers, dtype=np.float32),
        np.asarray(y_centers, dtype=np.float32),
    )


def fit_points_to_ellipse(
    x: Points,
    y: Points,
) -> tuple[float, float, float, float, float]:
    """Fit points to an ellipse and return the roll and tilt angles."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length!!!")
    a = np.array([x**2, x * y, y**2, x, y, np.ones_like(x)]).T
    vh = np.linalg.svd(a, full_matrices=False)[-1]
    a0, b0, c0, d0, e0, f0 = vh.T[:, -1]
    denom = b0**2 - 4 * a0 * c0
    msg = "Can't fit to an ellipse!!!"
    if denom == 0:
        raise ValueError(msg)
    xc: float = (2 * c0 * d0 - b0 * e0) / denom
    yc: float = (2 * a0 * e0 - b0 * d0) / denom
    roll_angle: float = np.rad2deg(
        np.arctan2(c0 - a0 - np.sqrt((a0 - c0) ** 2 + b0**2), b0)
    )
    if roll_angle > 90.0:
        roll_angle = -(180 - roll_angle)
    if roll_angle < -90.0:
        roll_angle = 180 + roll_angle
    a_term = (
        2
        * (a0 * e0**2 + c0 * d0**2 - b0 * d0 * e0 + denom * f0)
        * (a0 + c0 + np.sqrt((a0 - c0) ** 2 + b0**2))
    )
    if a_term < 0.0:
        raise ValueError(msg)
    a_major: float = -2 * np.sqrt(a_term) / denom
    b_term = (
        2
        * (a0 * e0**2 + c0 * d0**2 - b0 * d0 * e0 + denom * f0)
        * (a0 + c0 - np.sqrt((a0 - c0) ** 2 + b0**2))
    )
    if b_term < 0.0:
        raise ValueError(msg)
    b_minor: float = -2 * np.sqrt(b_term) / denom
    if a_major < b_minor:
        a_major, b_minor = b_minor, a_major
        if roll_angle < 0.0:
            roll_angle = 90 + roll_angle
        else:
            roll_angle = -90 + roll_angle
    return roll_angle, a_major, b_minor, xc, yc


def identify_sign_tilt_angle(
    x: Points,
    y: Points,
) -> int:
    """Find the two furthest-apart points and linear-fit through them."""
    data_points = np.asarray(list(zip(x, y, strict=True)))
    max_dist = 0
    index1, index2 = 0, 0
    for i in range(len(data_points)):
        for j in range(i + 1, len(data_points)):
            dist = np.linalg.norm(data_points[i] - data_points[j])
            if dist > max_dist:
                max_dist = dist
                index1, index2 = i, j
    # Perform a linear fit using the two furthest points
    x_furthest = [data_points[index1][0], data_points[index2][0]]
    y_furthest = [data_points[index1][1], data_points[index2][1]]
    coeffs = np.polyfit(x_furthest, y_furthest, 1)
    slope, intercept = coeffs

    min_index, max_index = min(index1, index2), max(index1, index2)
    y_dis = []
    for i in range(min_index, max_index + 1):
        x_i = data_points[i, 0]
        y_i = data_points[i, 1]
        y_fit = slope * x_i + intercept
        y_dis.append(y_i - y_fit)

    y_median = np.median(np.asarray(y_dis))
    if y_median < 0:
        angle_sign = 1
    else:
        angle_sign = -1

    return angle_sign


@dataclass
class AlignmentResult:
    """Roll and tilt of the rotation axis found from one detector's projections."""

    detector: str
    roll_angle: float
    tilt_angle: float
    method: TomoAlignMethod
    x_centers: Points
    y_centers: Points
    overlay: FloatImage
    #: ``(major_axis, minor_axis, xc, yc)`` for an ellipse fit, ``(slope, intercept)`` for linear.
    fit: tuple[float, ...]


def ellipse_fit(x: Points, y: Points) -> tuple[float, float, tuple[float, float, float, float]]:
    """Fit the sphere centers to an ellipse.

    Returns the roll and (unsigned) tilt angles, in degrees, and the ellipse's
    ``(major_axis, minor_axis, xc, yc)``.

    Raises
    ------
    ValueError
        If the points lie too close to a line, or don't fit an ellipse.
    """
    (a, b) = np.polyfit(x, y, 1)[:2]
    dist_list = np.abs(a * x - y + b) / np.sqrt(a**2 + 1)
    dist_list = ndi.gaussian_filter1d(dist_list, 2)
    if np.max(dist_list) < 1.0:
        raise ValueError("Distances of points to a fitted line is small.")

    try:
        roll_angle, major_axis, minor_axis, xc, yc = fit_points_to_ellipse(x, y)
    except ValueError as e:
        raise ValueError("Failed to fit points to an ellipse: " + str(e)) from e
    tilt_angle = np.rad2deg(np.arctan2(minor_axis, major_axis))
    return roll_angle, tilt_angle, (major_axis, minor_axis, xc, yc)


def linear_fit(x: Points, y: Points) -> tuple[float, float, tuple[float, float]]:
    """Fit the sphere centers to a line.

    Returns the roll and (unsigned) tilt angles, in degrees, and the line's
    ``(slope, intercept)``.
    """
    (a, b) = np.polyfit(x, y, 1)[:2]
    dist_list = np.abs(a * x - y + b) / np.sqrt(a**2 + 1)
    appr_major = np.max(
        np.asarray(
            [
                np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                for i in range(len(x))
                for j in range(i + 1, len(x))
            ]
        )
    )
    dist_list = ndi.gaussian_filter1d(dist_list, 2)
    appr_minor = 2.0 * np.max(dist_list)
    tilt_angle = np.rad2deg(np.arctan2(appr_minor, appr_major))
    roll_angle = np.rad2deg(np.arctan(a))
    return roll_angle, tilt_angle, (a, b)


def _mean_image(images: ImageDataset) -> FloatImage:
    """Average a stack of images (any leading dims) into one, with no zero pixels."""
    stack = np.asarray(images, dtype=np.float64)
    mean = stack.reshape(-1, *stack.shape[-2:]).mean(axis=0)
    # Zero pixels would divide by zero in the flat-field correction.
    mean[mean == 0.0] = np.mean(mean)
    return mean


def _load_flatfield(
    run: BlueskyRunV3,
    det_name: str,
    flat_run: BlueskyRunV3 | None,
    tiled_client: CatalogOfBlueskyRuns | None,
) -> FloatImage:
    """Find the flat field for ``det_name``.

    Uses the ``flat`` stream of ``flat_run`` if given, else the run's own ``flat``
    stream, else that of the run named by ``flat_scan_uid`` in its start document.
    """
    if flat_run is None:
        if "flat" in run and det_name in run["flat"]:
            return _mean_image(run["flat"].read()[det_name])
        flat_scan_uid = run.start.get("flat_scan_uid")
        if not isinstance(flat_scan_uid, str):
            raise ValueError("No flat field available and 'flat_scan_uid' is not specified.")

        catalog = ensure_available(CatalogOfBlueskyRuns, c=tiled_client)
        found = catalog[flat_scan_uid]
        if not isinstance(found, BlueskyRunV3):
            raise KeyError(f"No run found for flat_scan_uid {flat_scan_uid}.")
        flat_run = found

    if "flat" not in flat_run or det_name not in flat_run["flat"]:
        raise KeyError(f"The flat-field run has no '{det_name}' flat-field images.")
    return _mean_image(flat_run["flat"].read()[det_name])


def plot_alignment(result: AlignmentResult, figsize: tuple[int, int] = (14, 7)) -> None:
    """Show the projection overlay, and the sphere centers with the fitted path.
    Parameters
    ----------
    result : AlignmentResult
        The result of the alignment check containing overlay image, sphere centers, and fit.
    figsize : tuple[int, int], optional
        The size of the figure to display, by default (14, 7)
    """
    
    height, width = result.overlay.shape
    plt.figure(f"{result.detector}: projection overlay", figsize=figsize)
    plt.imshow(result.overlay, cmap="gray", extent=(0, width, 0, height))
    plt.tight_layout(rect=(0, 0, 1, 1))

    plt.figure(f"{result.detector}: sphere centers", figsize=figsize)
    x, y = result.x_centers, result.y_centers
    for i in range(len(x)):
        plt.plot(x[i], y[i], "o", markersize=10, color="cyan")
        plt.text(
            x[i], y[i], str(i), fontsize=10, fontweight="bold", ha="center", va="center", color="red"
        )
    plt.title(
        f"{result.detector} - Roll : {result.roll_angle:2.4f}; "
        f"Tilt : {result.tilt_angle:2.4f} (degree)"
    )
    if result.method == TomoAlignMethod.ELLIPSE:
        major_axis, minor_axis, xc, yc = result.fit
        angle = np.radians(result.roll_angle)
        theta = np.linspace(0, 2 * np.pi, 100)
        x_fit = (
            xc
            + 0.5 * major_axis * np.cos(theta) * np.cos(angle)
            - 0.5 * minor_axis * np.sin(theta) * np.sin(angle)
        )
        y_fit = (
            yc
            + 0.5 * major_axis * np.cos(theta) * np.sin(angle)
            + 0.5 * minor_axis * np.sin(theta) * np.cos(angle)
        )
        plt.plot(x_fit, y_fit, color="red")
    else:
        slope, intercept = result.fit
        plt.plot(x, slope * x + intercept, color="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()


def fetch_alignment_data(
    alignment_scan: BlueskyRunV3 | str | int,
    motor: RotationMotor,
    flat_scan: BlueskyRunV3 | None = None,
    tiled_client: CatalogOfBlueskyRuns | None = None,
) -> tuple[str, ImageDataset, FloatImage]:
    """Load the projections and flat field from a sphere alignment scan.

    Parameters
    ----------
    alignment_scan : BlueskyRunV3 | str | int
        The alignment run, or its uid or scan id in ``tiled_client``. Its start
        document's ``detectors`` must name exactly one detector.
    motor : RotationMotor
        The rotation motor whose readback must be in the projection stream.
    flat_scan : BlueskyRunV3, optional
        A run whose ``flat`` stream holds the flat-field images. If not given, the
        alignment run's own ``flat`` stream is used, else the run named by its
        ``flat_scan_uid``.
    tiled_client : CatalogOfBlueskyRuns, optional
        Catalog to look runs up in, only needed for a uid/scan id or a
        ``flat_scan_uid`` lookup. Defaults to ``c`` in the IPython namespace.

    Returns
    -------
    tuple[str, ImageDataset, FloatImage]
        The detector name, its projections, and the flat field.
    """
    if isinstance(alignment_scan, BlueskyRunV3):
        run = alignment_scan
    else:
        catalog = ensure_available(CatalogOfBlueskyRuns, c=tiled_client)
        found = catalog[alignment_scan]
        if not isinstance(found, BlueskyRunV3):
            raise ValueError(f"No run found for {alignment_scan!r}.")
        run = found

    det_names = run.start.get("detectors")
    if not isinstance(det_names, list) or len(det_names) != 1 or not isinstance(det_names[0], str):
        raise ValueError(
            f"Expected exactly one detector in the run's 'detectors' metadata, got {det_names}."
        )
    det_name = det_names[0]

    ensure_run_is_valid(run, [det_name], motor.name)

    projections: ImageDataset = np.asarray(run["primary"].read()[det_name])
    flatfield = _load_flatfield(run, det_name, flat_scan, tiled_client)
    return det_name, projections, flatfield


def check_alignment(
    projections: ImageDataset,
    flatfield: FloatImage | None = None,
    detector: str = "detector",
    left_crop: int = 0,
    right_crop: int = 0,
    top_crop: int = 500,
    bottom_crop: int = 500,
    method: TomoAlignMethod = TomoAlignMethod.ELLIPSE,
    ratio: float = 1.0,
    show_plots: bool = True,
) -> AlignmentResult:
    """Find the roll and tilt of the rotation axis from sphere projections.

    The projections (e.g. from :func:`fetch_alignment_data`) show a dense sphere
    over a full rotation. The sphere's center traces an ellipse whose
    orientation and eccentricity give the axis roll and tilt.

    Parameters
    ----------
    projections : ImageDataset
        The projection images, at least 36 of them.
    flatfield : FloatImage, optional
        The flat field to correct the projections with. If None, none is applied.
    detector : str, optional
        Detector name, used to label the result and plots.
    left_crop, right_crop, top_crop, bottom_crop : int, optional
        Pixels to crop from each image border before finding the sphere.
    method : TomoAlignMethod, optional
        Fit to an ellipse (default) or a line. An ellipse fit falls back to a line
        fit if the points are nearly collinear or don't fit an ellipse.
    ratio : float, optional
        Scales the binarization threshold used to find the sphere.
    show_plots : bool, optional
        Whether to plot the projection overlay and the fitted sphere path.

    Returns
    -------
    AlignmentResult
        The fitted roll and tilt, with the data used to compute them.
    """
    depth, height, width = projections.shape
    if depth < 36:
        raise ValueError(
            f"There are {depth} projections from '{detector}'; "
            "at least 36 are needed for a reliable fit."
        )
    check_crop_values_valid(width, height, left_crop, right_crop, top_crop, bottom_crop)

    images, x, y = crop_and_flatfield_correction(
        projections,
        flatfield=flatfield,
        top_crop=top_crop,
        bottom_crop=bottom_crop,
        left_crop=left_crop,
        right_crop=right_crop,
        ratio=ratio,
    )

    used_method = TomoAlignMethod(method)
    fit: tuple[float, ...] = ()
    roll_angle = tilt_angle = 0.0
    if used_method == TomoAlignMethod.ELLIPSE:
        try:
            roll_angle, tilt_angle, fit = ellipse_fit(x, y)
        except ValueError:
            # Nearly collinear or non-elliptical centers: a line fit still works.
            used_method = TomoAlignMethod.LINEAR
    if used_method == TomoAlignMethod.LINEAR:
        roll_angle, tilt_angle, fit = linear_fit(x, y)
    tilt_angle = abs(tilt_angle) * identify_sign_tilt_angle(x, y)

    result = AlignmentResult(
        detector=detector,
        roll_angle=float(roll_angle),
        tilt_angle=float(tilt_angle),
        method=used_method,
        x_centers=x,
        y_centers=y,
        overlay=np.mean(images, axis=0),
        fit=tuple(float(v) for v in fit),
    )

    if show_plots:
        plot_alignment(result)
        plt.show()

    return result


def tomo_alignment_scan(
    det: KinetixDetector | PhantomDetector,
    exposure_time: float,
    num_projections: int = 37,
    init_angle: float = 0.0,
    stop_angle: float = 360.0,
    base_x_offset: float = 0.0,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
    rot_motor: RotationMotor | None = None,
    sample_stage_x: AsyncEpicsMotor | None = None,
):
    """Tomography alignment scan.

    Parameters
    ----------
    det : KinetixDetector | PhantomDetector
        The detector to use for the scan.
    rotation_stage : RotationMotor
        The rotation stage motor.
    front_end_shutter : Shutter
        The front-end shutter.
    photon_shutter : Shutter
        The photon shutter.
    exposure_time : float
        Exposure time for each projection in seconds.
    num_projections : int, optional
        Number of projections to acquire, by default 37.
    init_angle : float, optional
        Initial angle for the rotation stage, by default 0.0.
    stop_angle : float, optional
        Final angle for the rotation stage, by default 360.0.
    base_x_offset : float, optional
        Base X offset for the sample stage, by default 0.0.
    sample_stage_x : AsyncEpicsMotor | None, optional
        The sample stage X motor, by default None.
    """

    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)
    rot_motor = ensure_available(RotationMotor, rot_motor=rot_motor)
    take_flat = abs(base_x_offset) > 0.0 and sample_stage_x is not None

    @bpp.reset_positions_decorator([rot_motor.velocity] + ([sample_stage_x] if take_flat else []))
    def _body():

        yield from ensure_shutter_open(fe_shutter)
        yield from ensure_shutter_open(photon_shutter, allow_actuation=True)

        # Set the rotation stage to the maximum velocity before starting the scan
        max_velocity = yield from bps.rd(rot_motor.max_velocity)
        yield from bps.mv(rot_motor.velocity, max_velocity)
        yield from bps.mv(rot_motor, init_angle)

        yield from bps.mv(det.driver.acquire_time, exposure_time)
        yield from bps.mv(
            det.driver.acquire_period, exposure_time + 0.002
        )  # TODO: Don't hard code this

        _md = {
            "description": "Tomography alignment scan",
            "plan_name": "tomography_alignment_scan",
            "tiled_specs": [TOMO_ALIGNMENT_SCAN_V1],
        }

        def _take_flat():
            yield from bps.mvr(sample_stage_x, base_x_offset)
            yield from bps.trigger_and_read([det], name="flat")
            yield from bps.mvr(sample_stage_x, -base_x_offset)

        def _insert_flat_after_open_run(msg):
            # bp.scan opens the run itself; splice the flat into it right after.
            if take_flat and msg.command == "open_run":
                return None, _take_flat()
            return None, None

        yield from bpp.plan_mutator(
            bp.scan([det], rot_motor, init_angle, stop_angle, num_projections, md=_md),
            _insert_flat_after_open_run,
        )

    def _cleanup():
        yield from ensure_shutter_closed(photon_shutter)

    return (bpp.finalize_wrapper(_body(), _cleanup()))
