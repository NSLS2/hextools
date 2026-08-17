"""XRD calibration tools."""

from collections.abc import Generator

import bluesky.plan_stubs as bps
from bluesky import Msg
from ophyd_async.epics.adcore import AreaDetector
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor

from hextools.photon_delivery_system import Shutter


def xrd_calibration(
    detector: AreaDetector,
    exposure_time: float,
    motor: AsyncEpicsMotor,
    photon_shutter: Shutter,
    num_distances: int = 3,
    gap: float = 200.0,
    start_position: float | None = None,
    description: str = "Energy-geometry calibration",
) -> Generator[Msg, None, None]:

    if start_position is None:
        start_position = yield from bps.rd(motor)
