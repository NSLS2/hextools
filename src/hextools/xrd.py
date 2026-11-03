"""X-Ray Diffraction (XRD) tools for HEX beamline."""

from typing import Any

from matplotlib import use
from ophyd_async.core import TriggerInfo
from hextools.detectors.utils import enable_background_subtraction
from hextools.motors import DiffractionTable, NDAxis, construct_nd_trajectory

from hextools.photon_delivery_system.shutter import Shutter, ensure_shutter_closed, ensure_shutter_open
from bluesky import Msg
from bluesky.utils import MsgGenerator
from hextools.detectors.perkin_elmer import PerkinElmerDetector
from bluesky import plans as bp, plan_stubs as bps, preprocessors as bpp
from bluesky import plan_patterns
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor

from hextools.utils import ensure_available

def _update_exposure_time(detector: PerkinElmerDetector, exposure_time: float) -> MsgGenerator[None]:
    """Update the exposure time of the detector if it differs from the current setting."""

    current_exposure_time = yield from bps.rd(detector.driver.acquire_time)
    if current_exposure_time != exposure_time:
        yield from bps.abs_set(detector.driver.acquire, False, wait=True)
        yield from bps.mv(detector.driver.acquire_time, exposure_time)
        yield from bps.abs_set(detector.driver.acquire, True, wait=False)
        yield from bps.sleep(exposure_time)  # Wait for at least the first frame
        yield from enable_background_subtraction(detector)


def xrd_scan_nd(
    detector: PerkinElmerDetector,
    exposure_time: float,
    *args,
    images_to_average: int = 1,
    num_iterations: int = 1,
    sleep_time: float = 0.0,
    use_shutter: bool = True,
    snake_axes: bool | list = False,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
    description: str = "N-Dimensional XRD scan",
    md: dict | None = None,
) -> MsgGenerator[None]:
    """Perform an XRD scan with the given parameters."""

    fe_shutter = ensure_available(Shutter, fe_shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, photon_shutter=photon_shutter)

    def _body():
        if use_shutter:
            yield from ensure_shutter_open(fe_shutter)
            yield from ensure_shutter_open(photon_shutter, allow_actuation=True)
        
        yield from _update_exposure_time(detector, exposure_time)
        yield from bps.prepare(
            detector,
            TriggerInfo(
                livetime=exposure_time,
                collections_per_event=1,
                exposures_per_collection=images_to_average,
            )
        )

        _md = {
            "description": description,
        }
        _md.update(md or {})
        yield from bp.grid_scan([detector], args, snake_axes=snake_axes, md=_md)

    def _cleanup():
        if use_shutter:
            yield from ensure_shutter_closed(photon_shutter)

    return (
        yield from bps.repeat(
            lambda: bpp.finalize_wrapper(_body(), _cleanup()),
            num=num_iterations,
            delay=sleep_time,
        )
    )

def xrd_scan_2d(
        detector: PerkinElmerDetector,
        exposure_time: float,
        outer_motor: AsyncEpicsMotor,
        outer_start: float,
        outer_stop: float,
        outer_num_points: int,
        inner_motor: AsyncEpicsMotor,
        inner_start: float,
        inner_stop: float,
        inner_num_points: int,
        images_to_average: int = 1,
        num_iterations: int = 1,
        sleep_time: float = 0.0,
        use_shutter: bool = True,
        snake_axes: bool | list = False,
        fe_shutter: Shutter | None = None,
        photon_shutter: Shutter | None = None,
        description: str = "2D XRD scan",
) -> MsgGenerator[None]:
    """Perform a 2D XRD scan with the given parameters."""
    return (yield from xrd_scan_nd(
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
        num_iterations=num_iterations,
        sleep_time=sleep_time,
        use_shutter=use_shutter,
        snake_axes=snake_axes,
        fe_shutter=fe_shutter,
        photon_shutter=photon_shutter,
        description=description,
        md={"description": description, "plan_name": "xrd_scan_2d"},
    ))

def xrd_scan(
        detector: PerkinElmerDetector,
        exposure_time: float,
        motor: AsyncEpicsMotor,
        start: float,
        stop: float,
        num_points: int,
        images_to_average: int = 1,
        num_iterations: int = 1,
        sleep_time: float = 0.0,
        use_shutter: bool = True,
        snake_axes: bool | list = False,
        fe_shutter: Shutter | None = None,
        photon_shutter: Shutter | None = None,
        description: str = "1D XRD scan",
) -> MsgGenerator[None]:
    """Perform a 1D XRD scan with the given parameters."""
    return (yield from xrd_scan_nd(
        detector,
        exposure_time,
        motor,
        start,
        stop,
        num_points,
        images_to_average=images_to_average,
        num_iterations=num_iterations,
        sleep_time=sleep_time,
        use_shutter=use_shutter,
        snake_axes=snake_axes,
        fe_shutter=fe_shutter,
        photon_shutter=photon_shutter,
        description=description,
        md={"description": description, "plan_name": "xrd_scan"},
    ))

def xrd_calibration(
    detector: PerkinElmerDetector,
    exposure_time: float,
    num_points: int = 3,
    gap: float = 200.0,
    start_position: float | None = None,
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
        raise ValueError(f"Stop position exceeds the maximum allowed value of 2000: {stop_position}.")

    return (
        yield from xrd_scan(
            detector,
            exposure_time,
            diff_z,
            start_position,
            stop_position,
            num_points=num_points,
            use_shutter=use_shutter,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
            description=description,
        )
    )