"""A tomography run's dark/flat must come from its own proposal and cycle.

RE.md is shared by the whole beamline and keeps ``last_dark_flat_uid`` across a
proposal switch. At HEX on 2026-10-22, tomo scans 249-256 in pass-312143 were
exported with the dark/flat of scan 246 from pass-314022.
"""

from unittest.mock import Mock

from bluesky import RunEngine
from bluesky_tiled_plugins.clients.bluesky_run import BlueskyRunV3

from hextools.exporters.tomography import resolve_dark_flat_run
from hextools.utils import DARK_FLAT_MD_KEYS, clear_last_dark_flat, reset_scan_id


def _run(data_session: str, cycle: str = "2026-3") -> Mock:
    run = Mock(spec=BlueskyRunV3)
    run.start = {
        "uid": f"uid-{data_session}",
        "data_session": data_session,
        "cycle": cycle,
    }
    return run


def test_clear_last_dark_flat(RE: RunEngine):
    RE.md.update({"last_dark_flat_uid": "uid-246", "last_dark_flat_scan_id": 246})
    RE(clear_last_dark_flat(RE=RE))
    assert not set(DARK_FLAT_MD_KEYS) & RE.md.keys()
    RE(clear_last_dark_flat(RE=RE))  # nothing left to clear is fine


def test_reset_scan_id_forgets_the_last_dark_flat(RE: RunEngine):
    RE.md.update({"scan_id": 256, "last_dark_flat_uid": "uid-246"})
    RE(reset_scan_id(RE=RE))
    assert RE.md["scan_id"] == 0
    assert not set(DARK_FLAT_MD_KEYS) & RE.md.keys()


def test_dark_flat_from_another_proposal_is_not_used():
    client = {"uid-246": _run("pass-314022")}
    start = {
        "data_session": "pass-312143",
        "cycle": "2026-3",
        "last_dark_flat_uid": "uid-246",
    }
    assert resolve_dark_flat_run(client, start) is None


def test_dark_flat_from_an_earlier_cycle_is_not_used():
    client = {"uid-246": _run("pass-312143", cycle="2026-2")}
    start = {
        "data_session": "pass-312143",
        "cycle": "2026-3",
        "last_dark_flat_uid": "uid-246",
    }
    assert resolve_dark_flat_run(client, start) is None


def test_dark_flat_from_the_same_proposal_is_used():
    dark_flat = _run("pass-312143")
    client = {"uid-260": dark_flat}
    start = {
        "data_session": "pass-312143",
        "cycle": "2026-3",
        "last_dark_flat_uid": "uid-260",
    }
    assert resolve_dark_flat_run(client, start) is dark_flat


def test_unresolvable_dark_flat_is_skipped_not_raised():
    start = {"data_session": "pass-312143", "last_dark_flat_uid": "gone"}
    assert resolve_dark_flat_run({}, start) is None


def test_no_dark_flat_recorded():
    assert resolve_dark_flat_run({}, {"data_session": "pass-312143"}) is None


def test_explicit_dark_flat_id_from_another_proposal_is_not_used():
    client = {"uid-246": _run("pass-314022")}
    start = {"data_session": "pass-312143", "cycle": "2026-3"}
    assert resolve_dark_flat_run(client, start, dark_flat_id="uid-246") is None


def test_explicit_dark_flat_id_overrides_the_start_document():
    dark_flat = _run("pass-312143")
    client = {"uid-260": dark_flat, "uid-246": _run("pass-314022")}
    start = {
        "data_session": "pass-312143",
        "cycle": "2026-3",
        "last_dark_flat_uid": "uid-246",
    }
    assert resolve_dark_flat_run(client, start, dark_flat_id="uid-260") is dark_flat


def test_unresolvable_explicit_dark_flat_id_is_skipped_not_raised():
    start = {"data_session": "pass-312143", "cycle": "2026-3"}
    assert resolve_dark_flat_run({}, start, dark_flat_id="gone") is None
