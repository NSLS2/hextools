"""X-Ray Diffraction (XRD) tools for HEX beamline."""

import logging
from collections.abc import MutableMapping
from pathlib import PureWindowsPath
from typing import Any

from bluesky import plan_stubs as bps
from bluesky import plans as bp
from bluesky import preprocessors as bpp
from bluesky.utils import MsgGenerator
from ophyd_async.core import TriggerInfo
from ophyd_async.epics.adcore import (
    NDCircularBuffIO,
    NDPluginBaseIO,
    NDPluginFileIO,
    NDProcessFilterType,
    NDProcessIO,
)
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor
from ophyd_async.sim import SimMotor

from hextools.detectors.perkin_elmer import PerkinElmerDetector
from hextools.detectors.utils import enable_preview_mode
from hextools.motors import DiffractionTable
from hextools.photon_delivery_system.shutter import (
    Shutter,
    ensure_shutter_closed,
    ensure_shutter_open,
)
from hextools.specs import XRD_CALIBRATION_V1, XRD_DARK_REFERENCE_V1, XRD_SCAN_V1
from hextools.utils import ensure_available

logger = logging.getLogger("hextools")


def _use_cb_plugin_as_img_source(detector: PerkinElmerDetector) -> MsgGenerator[None]:
    """Configure the detector to use the Circular Buffer plugin as the image source.

    Parameters
    ----------
    detector : PerkinElmerDetector
        The detector to configure.
    """
    logger.info(
        f"Configuring {detector} to use Circular Buffer plugin as image source."
    )
    trans_plugin = detector.get_plugin("trans", NDPluginBaseIO)
    cb_plugin = detector.get_plugin("cb", NDCircularBuffIO)

    cb_plugin_array_port = yield from bps.rd(cb_plugin.port_name)
    yield from bps.mv(trans_plugin.nd_array_port, cb_plugin_array_port)


def _change_exposure_time(
    detector: PerkinElmerDetector, exposure_time: float, frames_to_wait: int = 20
) -> MsgGenerator[None]:
    """Change the exposure time of the detector if it differs from the current setting.

    Parameters
    ----------
    detector : PerkinElmerDetector
        The detector to configure.
    exposure_time : float
        The desired exposure time.
    frames_to_wait : int, optional
        The number of frames to wait after changing the exposure time, by default 20.
    """
    current_exposure_time = yield from bps.rd(detector.driver.acquire_time)
    if current_exposure_time != exposure_time:
        yield from bps.abs_set(detector.driver.acquire, False, wait=True)
        yield from bps.mv(detector.driver.acquire_time, exposure_time)
        yield from bps.sleep(1)
    acquiring = yield from bps.rd(detector.driver.acquire)
    if not acquiring:
        yield from bps.abs_set(detector.driver.acquire, True, wait=False)
        yield from bps.sleep(frames_to_wait * exposure_time)


def dark_reference(
    det: PerkinElmerDetector,
    exposure_time: float,
    imgs_to_average: int,
    frames_to_wait: int = 20,
    filter_type: NDProcessFilterType = NDProcessFilterType.AVERAGE,
    photon_shutter: Shutter | None = None,
) -> MsgGenerator[None]:
    """Take a dark reference reading with a given exposure time.

    Parameters
    ----------
    det : PerkinElmerDetector
        The detector to use for the dark reference.
    exposure_time : float
        The exposure time for the dark reference.
    imgs_to_average : int
        The number of images to average for the dark reference.
    filter_type : NDProcessFilterType, optional
        The filter type to use for processing, by default NDProcessFilterType.AVERAGE.
    photon_shutter : Shutter | None, optional
        The photon shutter to use, by default None.
    frames_to_wait : int, optional
        The number of frames to wait after changing the exposure time, by default 20.
    """
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)
    yield from ensure_shutter_closed(photon_shutter)

    def _body():

        yield from _change_exposure_time(
            det, exposure_time, frames_to_wait=frames_to_wait
        )
        yield from _use_cb_plugin_as_img_source(det)

        det.save_as_tiff()
        proc_plugin = det.get_plugin("proc", NDProcessIO)
        proc_tiff = proc_plugin.tiff
        if proc_tiff is None:
            raise RuntimeError(
                "Failed to get the TIFF plugin from the processing plugin."
            )

        tiff_plugin = det.get_plugin("tiff", NDPluginFileIO)
        yield from bps.mv(
            proc_plugin.enable_background, False, proc_plugin.filter_type, filter_type
        )

        yield from bps.prepare(
            det,
            TriggerInfo(
                livetime=exposure_time,
                collections_per_event=1,
                exposures_per_collection=imgs_to_average,
            ),
        )

        _md = {
            "num_averaged": imgs_to_average,
            "exposure_time": exposure_time,
            "plan_name": "dark_reference",
            "tiled_specs": [
                {
                    "name": XRD_DARK_REFERENCE_V1.name,
                    "version": XRD_DARK_REFERENCE_V1.version,
                }
            ],
        }
        yield from bp.count([det], md=_md)

        file_path = yield from bps.rd(tiff_plugin.full_file_name)
        file_path = PureWindowsPath(file_path)

        yield from bps.mv(
            proc_tiff.file_path,
            str(file_path.parent),
            proc_tiff.file_name,
            file_path.name,
            proc_tiff.file_number,
            0,
            proc_tiff.file_template,
            "%s%s",
        )

        yield from bps.trigger(proc_plugin.read_background_tiff_seq)
        yield from bps.sleep(1)  # Give some time for the background to be loaded
        yield from bps.mv(proc_plugin.enable_background, True)

    def _cleanup():
        det.save_as_hdf()
        yield from enable_preview_mode(det)

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))


class LastDarkReferenceTracker:
    """RunEngine subscription that tracks the last dark-flat scan UID and scan ID."""

    def __init__(self, _metadata_dict: MutableMapping[str, Any]):
        self._metadata_dict = _metadata_dict

    def __call__(self, name: str, doc: dict[str, Any]):
        if (
            name == "start"
            and doc.get("uid", None)
            and doc.get("plan_name", None) == "dark_reference"
        ):
            self._metadata_dict["last_dark_reference"] = {
                **{
                    key: doc.get(key, None)
                    for key in ("uid", "scan_id", "num_averaged", "exposure_time")
                }
            }


def xrd_scan_nd(
    detector: PerkinElmerDetector,
    exposure_time: float,
    *args,
    images_to_average: int = 1,
    filter_type: NDProcessFilterType = NDProcessFilterType.AVERAGE,
    num_iterations: int = 1,
    sleep_time: float = 0.0,
    use_shutter: bool = True,
    snake_axes: bool | list = False,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
    description: str = "N-Dimensional XRD scan",
    md: dict | None = None,
) -> MsgGenerator[None]:
    """Perform an XRD scan with the given parameters.

    Parameters
    ----------
    detector : PerkinElmerDetector
        The detector to use for the scan.
    exposure_time : float
        The exposure time for each image.
    *args
        Positional arguments for the grid scan.
    images_to_average : int, optional
        The number of images to average, by default 1.
    filter_type : NDProcessFilterType, optional
        The filter type to use for processing, by default NDProcessFilterType.AVERAGE.
    num_iterations : int, optional
        The number of times to repeat the scan, by default 1.
    sleep_time : float, optional
        The time to sleep between iterations, by default 0.0.
    use_shutter : bool, optional
        Whether to use the shutters, by default True.
    snake_axes : bool | list, optional
        Whether to snake the scan axes, by default False.
    fe_shutter : Shutter | None, optional
        The front-end shutter to use, by default None.
    photon_shutter : Shutter | None, optional
        The photon shutter to use, by default None.
    description : str, optional
        A description for the scan, by default "N-Dimensional XRD scan".
    md : dict | None, optional
        Metadata for the scan, by default None.
    """
    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)

    # Extract the step axes from the positional arguments.
    if len(args) % 4:
        raise ValueError("Step axes must be given as (motor, start, stop, num) groups.")
    axes = [args[i : i + 4] for i in range(0, len(args), 4)]
    motors = [axis[0] for axis in axes]

    @bpp.reset_positions_decorator(motors)
    def _body():
        if use_shutter:
            yield from ensure_shutter_open(fe_shutter)
            yield from ensure_shutter_open(photon_shutter, allow_actuation=True)

        proc_plugin = detector.get_plugin("proc", NDProcessIO)
        yield from bps.mv(proc_plugin.filter_type, filter_type)

        yield from _change_exposure_time(detector, exposure_time)
        yield from _use_cb_plugin_as_img_source(detector)

        yield from bps.prepare(
            detector,
            TriggerInfo(
                livetime=exposure_time,
                exposures_per_collection=images_to_average,
            ),
        )

        _md = {
            "description": description,
            "plan_name": "xrd_scan_nd",
            "exposure_time": exposure_time,
            "num_averaged": images_to_average,
            "tiled_specs": [
                {
                    "name": XRD_SCAN_V1.name,
                    "version": XRD_SCAN_V1.version,
                }
            ],
        }
        _md.update(md or {})
        previous_uid = None

        for iteration in range(num_iterations):
            _md["iteration"] = iteration + 1
            if previous_uid is not None:
                _md["previous_uid"] = previous_uid
            previous_uid = yield from bp.grid_scan(
                [detector], *args, snake_axes=snake_axes, md=_md
            )
            if iteration < num_iterations - 1:
                yield from bps.sleep(sleep_time)

    def _cleanup():
        if use_shutter:
            yield from ensure_shutter_closed(photon_shutter)

        yield from enable_preview_mode(detector)

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))


def xrd_scan_2d(
    detector: PerkinElmerDetector,
    exposure_time: float,
    outer_motor: AsyncEpicsMotor | SimMotor,
    outer_start: float,
    outer_stop: float,
    outer_num_points: int,
    inner_motor: AsyncEpicsMotor | SimMotor,
    inner_start: float,
    inner_stop: float,
    inner_num_points: int,
    images_to_average: int = 1,
    filter_type: NDProcessFilterType = NDProcessFilterType.AVERAGE,
    num_iterations: int = 1,
    sleep_time: float = 0.0,
    use_shutter: bool = True,
    snake_axes: bool = False,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
    description: str = "2D XRD scan",
) -> MsgGenerator[None]:
    """Perform a 2D XRD scan with the given parameters.

    Parameters
    ----------
    detector : PerkinElmerDetector
        The detector to use for the scan.
    exposure_time : float
        The exposure time for each image.
    outer_motor : AsyncEpicsMotor | SimMotor
        The motor controlling the outer scan axis.
    outer_start : float
        The starting position for the outer motor.
    outer_stop : float
        The stopping position for the outer motor.
    outer_num_points : int
        The number of points to scan along the outer axis.
    inner_motor : AsyncEpicsMotor | SimMotor
        The motor controlling the inner scan axis.
    inner_start : float
        The starting position for the inner motor.
    inner_stop : float
        The stopping position for the inner motor.
    inner_num_points : int
        The number of points to scan along the inner axis.
    images_to_average : int, optional
        The number of images to average, by default 1.
    filter_type : NDProcessFilterType, optional
        The filter type to use for processing, by default NDProcessFilterType.AVERAGE.
    num_iterations : int, optional
        The number of times to repeat the scan, by default 1.
    sleep_time : float, optional
        The time to sleep between iterations, by default 0.0.
    use_shutter : bool, optional
        Whether to use the shutters, by default True.
    snake_axes : bool | list, optional
        Whether to snake the scan axes, by default False.
    fe_shutter : Shutter | None, optional
        The front-end shutter to use, by default None.
    photon_shutter : Shutter | None, optional
        The photon shutter to use, by default None.
    description : str, optional
        A description for the scan, by default "2D XRD scan".
    """
    return (
        yield from xrd_scan_nd(
            detector,
            exposure_time,
            outer_motor,
            outer_start,
            outer_stop,
            outer_num_points,
            inner_motor,
            inner_start,
            inner_stop,
            inner_num_points,
            images_to_average=images_to_average,
            filter_type=filter_type,
            num_iterations=num_iterations,
            sleep_time=sleep_time,
            use_shutter=use_shutter,
            snake_axes=snake_axes,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
            description=description,
            md={"description": description, "plan_name": "xrd_scan_2d"},
        )
    )


def xrd_scan(
    detector: PerkinElmerDetector,
    exposure_time: float,
    motor: AsyncEpicsMotor | SimMotor,
    start: float,
    stop: float,
    num_points: int,
    images_to_average: int = 1,
    filter_type: NDProcessFilterType = NDProcessFilterType.AVERAGE,
    num_iterations: int = 1,
    sleep_time: float = 0.0,
    use_shutter: bool = True,
    snake_axes: bool | list = False,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
    description: str = "1D XRD scan",
) -> MsgGenerator[None]:
    """Perform a 1D XRD scan with the given parameters.

    Parameters
    ----------
    detector : PerkinElmerDetector
        The detector to use for the scan.
    exposure_time : float
        The exposure time for each image.
    motor : AsyncEpicsMotor | SimMotor
        The motor controlling the scan axis.
    start : float
        The starting position for the motor.
    stop : float
        The stopping position for the motor.
    num_points : int
        The number of points to scan along the axis.
    images_per_point : int, optional
        The number of images to collect per point, by default 1.
    images_to_average : int, optional
        The number of images to average, by default 1.
    filter_type : NDProcessFilterType, optional
        The filter type to use for processing, by default NDProcessFilterType.AVERAGE.
    num_iterations : int, optional
        The number of times to repeat the scan, by default 1.
    sleep_time : float, optional
        The time to sleep between iterations, by default 0.0.
    use_shutter : bool, optional
        Whether to use the shutters, by default True.
    snake_axes : bool | list, optional
        Whether to snake the scan axes, by default False.
    fe_shutter : Shutter | None, optional
        The front-end shutter to use, by default None.
    photon_shutter : Shutter | None, optional
        The photon shutter to use, by default None.
    description : str, optional
        A description for the scan, by default "1D XRD scan".
    """
    return (
        yield from xrd_scan_nd(
            detector,
            exposure_time,
            motor,
            start,
            stop,
            num_points,
            images_to_average=images_to_average,
            filter_type=filter_type,
            num_iterations=num_iterations,
            sleep_time=sleep_time,
            use_shutter=use_shutter,
            snake_axes=snake_axes,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
            description=description,
            md={"description": description, "plan_name": "xrd_scan"},
        )
    )


def xrd_calibration(
    detector: PerkinElmerDetector,
    exposure_time: float,
    num_points: int = 3,
    gap: float = 200.0,
    start_position: float | None = None,
    images_to_average: int = 1,
    filter_type: NDProcessFilterType = NDProcessFilterType.AVERAGE,
    use_shutter: bool = True,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
    diff_table: DiffractionTable | None = None,
    description: str = "Energy-geometry calibration",
) -> MsgGenerator[None]:
    """Perform an XRD calibration scan.

    Parameters
    ----------
    detector : PerkinElmerDetector
        The detector to use for the calibration scan.
    exposure_time : float
        The exposure time for each step of the scan.
    num_points : int, optional
        The number of steps in the scan (default is 3).
    gap : float, optional
        The gap between steps in the scan (default is 200.0).
    start_position : float | None, optional
        The starting position of the diffraction table.
        Default is None, which uses the current position.
    use_shutter : bool, optional
        Whether to use the shutters during the scan (default is True).
    fe_shutter : Shutter | None, optional
        The front-end shutter to use.
    photon_shutter : Shutter | None, optional
        The photon shutter to use.
    diff_table : DiffractionTable | None, optional
        The diffraction table to use.
    description : str, optional
        A description for the scan.
    """
    diff_table = ensure_available(DiffractionTable, diff_table=diff_table)
    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)
    diff_z = diff_table.z

    if start_position is None:
        start_position = float((yield from bps.rd(diff_z)))

    stop_position = start_position + gap * (num_points - 1)
    if stop_position > 2000.0:
        raise ValueError(
            f"Stop position exceeds the maximum allowed value of 2000: {stop_position}."
        )
    elif stop_position < 650.0:
        raise ValueError(
            f"Stop position is below the minimum allowed value of 700: {stop_position}."
        )

    return (
        yield from xrd_scan_nd(
            detector,
            exposure_time,
            diff_z,
            start_position,
            stop_position,
            num_points,
            images_to_average=images_to_average,
            filter_type=filter_type,
            use_shutter=use_shutter,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
            md={
                "description": description,
                "plan_name": "xrd_calibration",
                "tiled_specs": [
                    {
                        "name": XRD_CALIBRATION_V1.name,
                        "version": XRD_CALIBRATION_V1.version,
                    }
                ],
            },
        )
    )


__all__ = [
    "xrd_calibration",
    "xrd_scan",
    "xrd_scan_2d",
    "dark_reference",
    "LastDarkReferenceTracker",
]
