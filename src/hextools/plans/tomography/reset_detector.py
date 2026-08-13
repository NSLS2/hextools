"""
Detector/stage recovery plan for HEX beamline.

Equivalent of the old pyepics script:
    hex-acq-pyepics/techniques/tomography/kinetix/util/reset_detector.py

Puts everything back into a known good state after an interrupted or
misbehaving scan: PandA disarmed and its capture stopped, rotation stage
stopped with the reset velocity, camera acquisition and HDF capture
stopped, the Proc averaging filter disabled, the HDF/PVA plugins re-pointed
at the default upstream port, and the live view restarted.

Note: the ported scan plans restore their own state in finalizers, so this
is a RECOVERY tool (crashed sessions, manual poking), not part of normal
operation.

Usage
-----
    RE(reset_detector(kinetix1, panda=panda1, rot_stage=rot_stage))
"""

import bluesky.plan_stubs as bps
from ophyd_async.core import EnableDisable

RESET_VELOCITY = 30.0  # deg/s, same as the legacy script


def reset_detector(
    detector,
    *,
    panda=None,
    rot_stage=None,
    source_port: str | None = None,
    live_view_period: float = 0.05,
):
    """
    Reset the detector (and optionally PandA + rotation stage) to defaults.

    Parameters
    ----------
    detector : HEXKinetixDetector
        Built by ``hextools.detectors.kinetix.make_kinetix``.
    panda : HDFPanda, optional
        When given: disarm PCAP and stop the PandA HDF capture.
    rot_stage : Motor, optional
        When given: stop it and restore the reset velocity.
    source_port : str, optional
        Upstream port for the HDF/PVA plugins.  Defaults to the detector's
        ``default_source_port`` (TRANS1 for Det:1, KTX1 for Det:3 — the
        legacy defaults), falling back to the camera's own port name.
    live_view_period : float, optional
        Acquire period for the restarted live view.  The legacy script
        used 0 (driver-limited); a real Kinetix clamps that to its readout
        time, but a simulated camera free-runs unbounded and floods the
        plugin chain — so a sane explicit default (0.05 s) is used.
    """
    if source_port is None:
        source_port = getattr(detector, "default_source_port", None)
    if source_port is None:
        source_port = yield from bps.rd(detector.driver.port_name)

    if panda is not None:
        print("Disarming PandA...")
        yield from bps.abs_set(panda.pcap.arm, False, wait=True)
        yield from bps.abs_set(panda.data.capture, False, wait=True)

    if rot_stage is not None:
        print("Stopping rotation stage, restoring velocity...")
        yield from bps.stop(rot_stage)
        yield from bps.mv(rot_stage.velocity, RESET_VELOCITY)

    print("Stopping camera acquisition and HDF capture...")
    yield from bps.abs_set(detector.driver.acquire, 0, wait=True)
    yield from bps.abs_set(detector.hdf.capture, 0, wait=True)

    print("Disabling the Proc averaging filter...")
    proc = detector.proc
    yield from bps.mv(proc.enable_filter, "Disable")
    yield from bps.mv(proc.num_filter, 1)
    yield from bps.mv(proc.queue_size, 1)
    yield from bps.mv(proc.enable_callbacks, EnableDisable.DISABLE)

    print(f"Re-pointing HDF/PVA plugins at {source_port}...")
    yield from bps.mv(detector.hdf.nd_array_port, source_port)
    yield from bps.mv(detector.pva.nd_array_port, source_port)

    print("Restarting live view...")
    yield from bps.mv(detector.driver.acquire_period, live_view_period)
    # wait_for takes awaitable FACTORIES (callables returning a
    # coroutine) so the RE creates the coroutine in its own loop —
    # hence the bound method, not a call.
    yield from bps.wait_for([detector.start_live_view])
    print("Reset done.")
