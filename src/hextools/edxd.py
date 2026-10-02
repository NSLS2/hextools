

from bluesky.preprocessors import finalize_wrapper, reset_positions_decorator

from hextools.detectors.germ import GeRMDetector
from hextools.photon_delivery_system import ensure_shutter_open, Shutter
from hextools.photon_delivery_system.shutter import ensure_shutter_closed
from hextools.utils import ensure_available
from bluesky import plan_stubs as bps, plans as bp
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor

def edxd_scan(
    movable: AsyncEpicsMotor,
    start: float,
    stop: float,
    num_points: int,
    count_time: int,
    description: str | None,
    save_as_tiff: bool = False,
    reset_position: bool = True,
    germ: GeRMDetector | None = None,
    fe_shutter: Shutter | None = None,
    photon_shutter: Shutter | None = None,
):

    fe_shutter = ensure_available(Shutter, shutter=fe_shutter)
    photon_shutter = ensure_available(Shutter, shutter=photon_shutter)
    germ = ensure_available(GeRMDetector, germ=germ)

    def _body():
        yield from ensure_shutter_open(fe_shutter, allow_actuation=True)
        yield from ensure_shutter_open(photon_shutter, allow_actuation=True)
        yield from bps.mv(germ.driver.acquire_time, count_time)
        if save_as_tiff:
            germ.save_as_tiff()

        _md = {
            "plan_name": "edxd_scan",
        }
        if description is not None:
            _md["description"] = description
        yield from bp.scan([germ], movable, start, stop, num_points, md=_md)

    def _cleanup():
        yield from ensure_shutter_closed(fe_shutter)
        yield from ensure_shutter_closed(photon_shutter)
        if reset_position:
            yield from bps.mv(movable, start)

    return (yield from finalize_wrapper(_body(), _cleanup()))
