from ophyd_async.epics.adkinetix import KinetixReadoutMode

_KINETIX_MAX_SENSOR_SIZE = 3200
_KINETIX_VALID_BINNING = {1, 2, 4}

# Max frame rates (Hz) for PCIe connection, indexed by ROI height.
# Each list is sorted by height ascending, with (height, framerate) tuples.
_KINETIX_FRAMERATES: dict[KinetixReadoutMode, list[tuple[int, float]]] = {
    KinetixReadoutMode.DYNAMIC_RANGE: [
        (2, 88900),
        (4, 53300),
        (8, 29600),
        (16, 15700),
        (32, 8100),
        (64, 4100),
        (1600, 166),
        (2048, 130),
        (2304, 115),
        (3200, 83),
    ],
    KinetixReadoutMode.SPEED: [
        (2, 107200),
        (4, 99400),
        (8, 80000),
        (16, 57100),
        (32, 36400),
        (64, 21100),
        (1600, 996),
        (2048, 778),
        (2304, 691),
        (3200, 498),
    ],
    KinetixReadoutMode.SENSITIVITY: [
        (2, 77200),
        (4, 47200),
        (8, 28300),
        (16, 15700),
        (32, 8300),
        (64, 4300),
        (1600, 176),
        (2048, 138),
        (2304, 122),
        (3200, 88),
    ],
    KinetixReadoutMode.SUB_ELECTRON: [
        (2, 2700),
        (4, 2100),
        (8, 1400),
        (16, 800),
        (32, 500),
        (64, 200),
        (1600, 10.4),
        (2048, 8.1),
        (2304, 7.2),
        (3200, 5.2),
    ],
}


def _interpolate_framerate(data: list[tuple[int, float]], height: int) -> float:
    """Linearly interpolate the max framerate for a given ROI height."""
    if height <= data[0][0]:
        return data[0][1]
    if height >= data[-1][0]:
        return data[-1][1]

    for i in range(len(data) - 1):
        h0, f0 = data[i]
        h1, f1 = data[i + 1]
        if h0 <= height <= h1:
            t = (height - h0) / (h1 - h0)
            return f0 + t * (f1 - f0)

    return data[-1][1]


def compute_max_kinetix_framerate(
    readout_mode: KinetixReadoutMode,
    roi_height: int = _KINETIX_MAX_SENSOR_SIZE,
) -> float:
    """Given the readout mode, ROI size, and binning of the Kinetix detector, return its maximum framerate in Hz.

    The Kinetix uses digital binning (post-readout), so binning does not affect
    the readout speed. The frame rate is determined by roi_height (rows read from
    the sensor). Width does not affect frame rate due to column-parallel readout.
    """
    if readout_mode not in _KINETIX_FRAMERATES:
        raise ValueError(f"Unknown readout mode: {readout_mode}")
    if not 1 <= roi_height <= _KINETIX_MAX_SENSOR_SIZE:
        raise ValueError(
            f"roi_height must be between 1 and {_KINETIX_MAX_SENSOR_SIZE}, got {roi_height}"
        )

    return _interpolate_framerate(_KINETIX_FRAMERATES[readout_mode], roi_height)


def calculate_scan_time(
    num_images: int,
    exposure_time: float,
    acquire_period: float = 0.0,
    overhead: float = 0.005,
    max_framerate: float | None = None,
    max_velocity: float | None = None,
    travel_distance: float | None = None,
) -> float:
    """Calculate the total scan time based on the number of images, exposure time, and acquire period.

    Parameters
    ----------
    num_images : int
        The total number of images to be acquired.
    exposure_time : float
        The exposure time for each image in seconds.
    acquire_period : float, optional
        The time between the start of one acquisition and the start of the next, in seconds. If 0, it will be set to exposure_time + overhead.

    Returns
    -------
    float
        The total scan time in seconds.
    """
    if (max_velocity is None) == (travel_distance is None):
        raise ValueError(
            "Both max_velocity and travel_distance must be provided together or not at all."
        )
    elif max_velocity is not None and travel_distance is not None:
        scan_time_max_velocity = travel_distance / max_velocity
    else:
        scan_time_max_velocity = 0

    if (acquire_period + overhead) < exposure_time or acquire_period == 0.0:
        acquire_period = exposure_time + overhead

    base_scan_time = (num_images - 1) * acquire_period + exposure_time

    scan_time_max_framerate = (
        (num_images - 1) / max_framerate if max_framerate is not None else 0
    )

    return max(base_scan_time, scan_time_max_velocity, scan_time_max_framerate)


def calculate_zero_encoder_value(
    current_encoder: int, current_deg: float, counts_per_rev: int
):
    """Calculate the zero encoder value based on the current encoder value, current degree, and counts per revolution.

    Parameters
    ----------
    current_encoder : int
        The current encoder value.
    current_deg : float
        The current degree position of the motor.
    counts_per_rev : int
        The number of encoder counts per revolution.

    Returns
    -------
    int
        The calculated zero encoder value.
    """
    zero_encoder = (
        current_encoder - (current_deg / 360.0) * counts_per_rev
    ) % counts_per_rev
    return int(zero_encoder)
