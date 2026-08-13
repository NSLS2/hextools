"""
Live-view flat-field correction plan for HEX beamline.

Equivalent of the old pyepics script:
    hex-acq-pyepics/techniques/tomography/kinetix/util/enable_flat_correction_areadetector.py

Enable mode: opens the shutter, moves the sample out by *flat_x_offset*,
captures a flat-field reference into the Proc plugin (SaveFlatField),
enables flat-field normalization, routes the live view (PVA plugin) through
Proc, and moves the sample back — the live view then shows
flat-corrected images.

Disable mode: routes the live view back to the default upstream port and
turns the correction off.

Usage
-----
    RE(enable_flat_correction(
        kinetix1, sample_x, ph_open_cmd, ph_close_cmd,
        exposure_time=0.05, flat_x_offset=5.0, enable=True,
    ))
"""

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from ophyd_async.core import EnableDisable

from hextools.plans.shutter import close_photon_shutter, open_photon_shutter


def enable_flat_correction(
    detector,
    sample_x,
    ph_open_cmd,
    ph_close_cmd,
    *,
    exposure_time: float,
    flat_x_offset: float,
    enable: bool = True,
    keep_shutter_open: bool = False,
    settle_time: float = 1.0,
):
    """
    Enable/disable on-the-fly flat-field correction of the live view.

    Parameters
    ----------
    detector : HEXKinetixDetector
        Built by ``hextools.detectors.kinetix.make_kinetix``.
    sample_x : Motor
        Sample X-axis, moved by *flat_x_offset* to capture the flat.
    ph_open_cmd / ph_close_cmd : signal
        Photon-shutter command signals.
    exposure_time : float
        Live-view exposure while capturing the flat reference.
    flat_x_offset : float
        Relative X move (mm) that takes the sample out of the beam.
    enable : bool, optional
        True (default) = capture reference + enable correction;
        False = disable and restore normal live view.
    keep_shutter_open : bool, optional
        Leave the shutter open afterwards. Default False.
    settle_time : float, optional
        Pause between plugin writes (the plugin needs frames flowing
        between steps; matches the legacy 1 s sleeps).
    """
    proc = detector.proc
    default_port = getattr(detector, "default_source_port", None)

    if not enable:
        print("Disabling flat-field correction...")
        if default_port is None:
            default_port = yield from bps.rd(detector.driver.port_name)
        yield from bps.mv(detector.pva.nd_array_port, default_port)
        yield from bps.mv(proc.enable_callbacks, EnableDisable.DISABLE)
        yield from bps.mv(proc.enable_flat_field, "Disable")
        print("Live view back to normal mode.")
        return

    # ---- enable path ----
    # Live view running with the requested exposure.  Period = exposure
    # rather than the legacy 0: a real Kinetix clamps period 0 to its
    # readout time, but a simulated camera free-runs unbounded and floods
    # the plugin chain.
    yield from bps.mv(detector.driver.acquire_period, exposure_time)
    yield from bps.mv(detector.driver.acquire_time, exposure_time)
    # wait_for takes awaitable FACTORIES (callables returning a
    # coroutine) so the RE creates the coroutine in its own loop —
    # hence the bound method, not a call.
    yield from bps.wait_for([detector.start_live_view])

    moved = {"out": False}

    def _body():
        yield from open_photon_shutter(ph_open_cmd)
        print(f"Moving sample out ({flat_x_offset:+g}) to capture the flat...")
        yield from bps.mvr(sample_x, flat_x_offset)
        moved["out"] = True

        # Proc must see frames: route it off the current live-view source.
        source = yield from bps.rd(detector.pva.nd_array_port)
        proc_port = yield from bps.rd(proc.port_name)
        if source == proc_port:
            source = yield from bps.rd(proc.nd_array_port)
        yield from bps.mv(proc.nd_array_port, source)
        yield from bps.mv(proc.enable_callbacks, EnableDisable.ENABLE)
        yield from bps.sleep(settle_time)
        yield from bps.mv(detector.pva.nd_array_port, proc_port)
        yield from bps.sleep(settle_time)
        yield from bps.mv(proc.save_flat_field, "Yes")
        yield from bps.sleep(settle_time)
        yield from bps.mv(proc.save_flat_field, "Yes")
        yield from bps.sleep(settle_time)
        valid = yield from bps.rd(proc.valid_flat_field)
        if valid != "Valid":
            raise RuntimeError(
                f"Flat-field reference not captured (ValidFlatField={valid}) "
                "— is the camera acquiring?"
            )
        yield from bps.mv(proc.enable_flat_field, "Enable")
        print("Live view is now flat-corrected.")

    def _cleanup():
        if moved["out"]:
            print("Moving sample back...")
            yield from bps.mvr(sample_x, -flat_x_offset)
        if not keep_shutter_open:
            yield from close_photon_shutter(ph_close_cmd)

    return (yield from bpp.finalize_wrapper(_body(), _cleanup()))
