"""
Photon-shutter plan helpers shared by the tomography plans.

(Mirrors open_ph_shutter / close_ph_shutter from the profile's 03-motors.py;
previously private to plans/tomography/alignment_scan.py — moved here once a
second plan needed them.)

The old scripts' front-end status check is NOT here yet — it arrives with the
shared beam-check helpers port from lib_device_control; until then the
operator confirms the front end.
"""

import bluesky.plan_stubs as bps


def open_photon_shutter(ph_open_cmd, sleep_time: float = 3.0):
    """Open the photon shutter and wait for it to settle."""
    print("Opening photon shutter...")
    yield from bps.abs_set(ph_open_cmd, 1, wait=True)
    yield from bps.sleep(sleep_time)
    print("Photon shutter open.")


def close_photon_shutter(ph_close_cmd, sleep_time: float = 3.0):
    """Close the photon shutter and wait for it to settle."""
    print("Closing photon shutter...")
    yield from bps.abs_set(ph_close_cmd, 1, wait=True)
    yield from bps.sleep(sleep_time)
    print("Photon shutter closed.")
