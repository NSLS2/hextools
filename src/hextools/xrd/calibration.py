"""XRD calibration tools."""

from collections.abc import Generator

from bluesky import Msg
from bluesky import plan_stubs as bps
from bluesky import plans as bp
from ophyd_async.epics.adcore import AreaDetector
from ophyd_async.epics.adcore._io import ADBaseIOT
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor

from hextools.photon_delivery_system import Shutter


def xrd_calibration(
    detector: AreaDetector[ADBaseIOT],
    exposure_time: float,
    motor: AsyncEpicsMotor,
    photon_shutter: Shutter,
    num_steps: int = 3,
    gap: float = 200.0,
    start_position: float | None = None,
    description: str = "Energy-geometry calibration",
) -> Generator[Msg, None, None]:
    """Perform an XRD calibration scan."""
    if start_position is None:
        start_position: float = yield from bps.rd(motor.user_readback)

    yield from bps.abs_set(
        detector.driver.acquire_time,
        exposure_time,
        detector.driver.acquire_period,
        0.0,
        detector.driver.num_images,
        1,
        wait=True,
    )

    try:
        yield from bps.mv(photon_shutter, True)

        _md = {
            "description": description,
            "plan_name": "xrd_calibration",
        }
        yield from bp.scan(
            [detector],
            motor,
            start_position,
            start_position + gap * (num_steps - 1),
            num_steps,
            md=_md,
        )
    finally:
        yield from bps.mv(photon_shutter, False)
        yield from bps.mv(motor, start_position)
