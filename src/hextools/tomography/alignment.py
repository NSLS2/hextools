"""Tomography alignment tools for HEX beamline."""

from enum import StrEnum

import algotom.util.calibration as calib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from bluesky import plan_stubs as bps
from bluesky import plans as bp
from bluesky_tiled_plugins.clients.bluesky_run import BlueskyRunV3
from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor
from skimage import measure, segmentation
from skimage.measure._regionprops import RegionProperties

from hextools.detectors.phantom import PhantomDetector
from hextools.motors import RotationMotor
from hextools.photon_delivery_system import Shutter

Image = np.ndarray[tuple[int, int], np.dtype[np.uint16] | np.dtype[np.uint8]]
BinaryImage = np.ndarray[tuple[int, int], np.dtype[bool]]
ImageDataset = np.ndarray[
    tuple[int, int, int], np.dtype[np.uint16] | np.dtype[np.uint8]
]


def ensure_run_is_valid(
    run: BlueskyRunV3,
    det_names: list[str],
    motor_name: str,
    proj_stream: str = "primary",
    ff_stream: str | None = None,
) -> None:
    """Check if a BlueskyRunV3 object is valid for tomography analysis.

    Parameters
    ----------
    run : BlueskyRunV3
        The BlueskyRunV3 object to check.

    Raises
    ------
    KeyError
        If stream, detector, or motor datasets are not available
    """

    def _check_stream_exists_and_contains_detector(
        stream_name: str, requires_motor: bool = True
    ):
        if stream_name not in run:
            raise KeyError(f"Stream '{stream_name}' not found in the run.")
        data_stream = run[stream_name]
        for det_name in det_names:
            if det_name not in data_stream:
                raise KeyError(
                    f"Detector '{det_name}' not found in the stream '{stream_name}'."
                )
        if requires_motor and motor_name not in data_stream:
            raise KeyError(
                f"Motor '{motor_name}' not found in the stream '{stream_name}'."
            )

    _check_stream_exists_and_contains_detector(proj_stream)

    if ff_stream is not None:
        _check_stream_exists_and_contains_detector(ff_stream, requires_motor=False)


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


def clean_image(binary_image, size_threshold=100):
    """Clean binary image."""
    # Clear objects connected to the border and fill holes
    binary_image = segmentation.clear_border(binary_image)
    binary_image = ndi.binary_opening(binary_image, iterations=2)
    binary_image = ndi.binary_fill_holes(binary_image)

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
    projection_angles: list[float],
    flatfield: Image | None = None,
    top_crop: int = 500,
    bottom_crop: int = 500,
    left_crop: int = 0,
    right_crop: int = 0,
    ratio: float = 1.0,
    figsize: tuple[int, int] = (14, 7),
) -> tuple[ImageDataset, list[float], list[float]]:
    """Crop the projection data and apply flat-field correction.

    Parameters
    ----------
    projection_data : ImageDataset
        The 3D array of projection images to be processed.
    flatfield : Image
        The flat-field image used for correction.
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
        Size of the figure for displaying images.
    """
    cropped_and_normalized = []
    x_centers = []
    y_centers = []
    for i, proj_img in enumerate(projection_data):
        # Crop image and perform flat-field correction
        mat = proj_img[
            top_crop : proj_img.shape[0] - bottom_crop,
            left_crop : proj_img.shape[1] - right_crop,
        ]
        if flatfield is not None:
            mat = (
                mat
                / flatfield[
                    top_crop : flatfield.shape[0] - bottom_crop,
                    left_crop : flatfield.shape[1] - right_crop,
                ]
            )
        # Denoise
        mat = ndi.gaussian_filter(mat, 5)
        # Normalize the background.
        # Optional
        threshold = calib.calculate_threshold(mat, bgr="bright")
        # Binarize the image
        mat_bin0 = calib.binarize_image(mat, threshold=ratio * threshold, bgr="bright")
        mat_bin0 = clean_image(mat_bin0)
        nmean = np.sum(mat_bin0)
        if nmean < 20.0:
            print("\n******************************************************")
            print("Adjust ratio of threshold or the field of view to get the sphere!")
            print(f"Current used ratio: {ratio} and threshold: {threshold} ")
            print("********************************************************")
            plt.figure(figsize=figsize)
            plt.imshow(mat, cmap="gray")
            plt.show()
            raise ValueError("No binary sphere detected! Please adjust parameters!")
        # Keep the sphere only
        sphere_size = calib.get_dot_size(mat_bin0, size_opt="max")
        mat_bin = calib.select_dot_based_size(mat_bin0, sphere_size)
        (y_cen, x_cen) = ndi.center_of_mass(mat_bin)
        x_centers.append(x_cen)
        y_centers.append((bottom_crop - top_crop) - y_cen)
        cropped_and_normalized.append(mat)
        print(
            f"  ---> Done image: {i:2} "
            f"| Angle: {projection_angles[i]:3.1f} "
            f"| Center X: {x_cen:4.2f} | Center Y: {y_cen:4.2f}"
        )
        # plt.figure(0)
        # plt.imshow(mat, cmap="gray")
        # plt.figure(1)
        # plt.imshow(mat_bin, cmap="gray")
    # plt.show()
    return np.asarray(cropped_and_normalized)


def fit_points_to_ellipse(
    x: np.ndarray[tuple[int], np.dtype[np.int32]],
    y: np.ndarray[tuple[int], np.dtype[np.int32]],
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
    x: np.ndarray[tuple[int], np.dtype[np.float32]],
    y: np.ndarray[tuple[int], np.dtype[np.float32]],
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


def ellipse_fit(x, y):
    """Fit points to an ellipse and return the roll and tilt angles."""
    (a, b) = np.polyfit(x, y, 1)[:2]
    dist_list = np.abs(a * x - y + b) / np.sqrt(a**2 + 1)
    dist_list = ndi.gaussian_filter1d(dist_list, 2)
    if np.max(dist_list) < 1.0:
        raise ValueError("Distances of points to a fitted line is small.")

    try:
        roll_angle, major_axis, minor_axis, xc, yc = fit_points_to_ellipse(x, y)
        tilt_angle = np.rad2deg(np.arctan2(minor_axis, major_axis))
    except ValueError as e:
        raise ValueError("Failed to fit points to an ellipse: " + str(e)) from e
    return roll_angle, tilt_angle


def linear_fit(x, y):
    """Perform a linear fit to the given x and y data points."""
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
    return roll_angle, tilt_angle


def check_alignment(
    alignment_scan: BlueskyRunV3,
    dets: list[KinetixDetector | PhantomDetector],
    motor: RotationMotor,
    left_crop: int = 0,
    right_crop: int = 0,
    top_crop: int = 500,
    bottom_crop: int = 500,
    method: TomoAlignMethod = TomoAlignMethod.ELLIPSE,
    ratio: float = 1.0,
    proj_stream: str = "primary",
    flatfield_stream_name: str = "primary",
):
    """Check the alignment of a tomography scan."""
    if any(crop < 0 for crop in [left_crop, right_crop, top_crop, bottom_crop]):
        raise ValueError("Crop values must be non-negative integers.")

    if len(dets) == 0:
        raise ValueError("At least one detector must be provided.")

    ensure_run_is_valid(
        alignment_scan,
        [det.name for det in dets],
        motor.name,
        proj_stream,
        flatfield_stream_name,
    )

    data = alignment_scan[proj_stream].read()
    for det in dets:
        depth, height, width = data[det.name].shape
        if depth < 36:
            raise ValueError(
                "The alignment scan must contain at least"
                "36 projections for a reliable fit."
            )

    check_crop_values_valid(width, height, left_crop, right_crop, top_crop, bottom_crop)


def tomo_alignment_scan(
    dets: list[KinetixDetector | PhantomDetector],
    rotation_stage: RotationMotor,
    front_end_shutter: Shutter,
    photon_shutter: Shutter,
    exposure_time: float,
    num_projections: int = 37,
    init_angle: float = 0.0,
    stop_angle: float = 360.0,
    base_x_offset: float = 0.0,
    sample_stage_x: AsyncEpicsMotor | None = None,
):
    """Tomography alignment scan.

    Parameters
    ----------
    dets : list[KinetixDetector | PhantomDetector]
        List of detectors to use for the scan.
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
    # Check the shutter statuses
    fe_shutter_open = yield from bps.rd(front_end_shutter.status)
    photon_shutter_open = yield from bps.rd(photon_shutter.status)

    # FE shutter must already be open. If not, raise an error.
    # If the photon shutter is closed, open it.
    if not fe_shutter_open:
        raise ValueError(
            "Front-end shutter is closed. Please open it before starting the scan."
        )
    if not photon_shutter_open:
        yield from bps.mv(photon_shutter, True)

    # Set the rotation stage to the maximum velocity before starting the scan
    max_velocity = yield from bps.rd(rotation_stage.max_velocity)
    yield from bps.mv(rotation_stage.velocity, max_velocity)
    yield from bps.mv(rotation_stage, init_angle)

    for det in dets:
        yield from bps.mv(det.driver.acquire_time, exposure_time)
        yield from bps.mv(
            det.driver.acquire_period, exposure_time + 0.002
        )  # TODO: Don't hard code this

    # Optionally, take a single flat image
    flat_uid = None
    if abs(base_x_offset) > 0.0 and sample_stage_x is not None:
        yield from bps.mvr(sample_stage_x, base_x_offset)
        flat_uid = yield from bp.count(
            dets, md={"description": "Flat-field image for tomography alignment"}
        )
        yield from bps.mvr(sample_stage_x, -base_x_offset)

    _md = {
        "description": "Tomography alignment scan",
    }
    if flat_uid is not None:
        _md["flat_uid"] = flat_uid
    yield from bp.scan(
        dets, rotation_stage, init_angle, stop_angle, num_projections, md=_md
    )
