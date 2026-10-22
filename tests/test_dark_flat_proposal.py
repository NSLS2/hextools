"""A tomography run's dark/flat must come from its own proposal.

RE.md is shared by the whole beamline and keeps ``last_dark_flat_uid`` across a
proposal switch. At HEX on 2026-10-22, tomo scans 249-256 in pass-312143 were
exported with the dark/flat of scan 246 from pass-314022.
"""

from unittest.mock import Mock

import pytest
from bluesky_tiled_plugins.clients.bluesky_run import BlueskyRunV3

from hextools.exporters.tomography import resolve_dark_flat_run
from hextools.utils import DARK_FLAT_MD_KEYS, switch_proposal


def _run(data_session: str) -> Mock:
    run = Mock(spec=BlueskyRunV3)
    run.start = {"uid": f"uid-{data_session}", "data_session": data_session}
    return run


@pytest.fixture
def mock_mode(monkeypatch):
    monkeypatch.setenv("HEXTOOLS_RUNNING_IN_CI", "YES")


def test_switching_proposal_forgets_the_previous_proposals_dark_flat(mock_mode):
    md = {
        "data_session": "pass-314022",
        "last_dark_flat_uid": "uid-246",
        "last_dark_flat_scan_id": 246,
    }
    switch_proposal(312143, "user", "pw", md=md)
    assert md["data_session"] == "pass-312143"
    assert not set(DARK_FLAT_MD_KEYS) & md.keys()


def test_resyncing_the_same_proposal_keeps_its_dark_flat(mock_mode):
    md = {"data_session": "pass-314022", "last_dark_flat_uid": "uid-246"}
    switch_proposal(314022, "user", "pw", md=md)
    assert md["last_dark_flat_uid"] == "uid-246"


def test_dark_flat_from_another_proposal_is_not_used():
    client = {"uid-246": _run("pass-314022")}
    start = {"data_session": "pass-312143", "last_dark_flat_uid": "uid-246"}
    assert resolve_dark_flat_run(client, start) is None


def test_dark_flat_from_the_same_proposal_is_used():
    dark_flat = _run("pass-312143")
    client = {"uid-260": dark_flat}
    start = {"data_session": "pass-312143", "last_dark_flat_uid": "uid-260"}
    assert resolve_dark_flat_run(client, start) is dark_flat


def test_unresolvable_dark_flat_is_skipped_not_raised():
    start = {"data_session": "pass-312143", "last_dark_flat_uid": "gone"}
    assert resolve_dark_flat_run({}, start) is None


def test_no_dark_flat_recorded():
    assert resolve_dark_flat_run({}, {"data_session": "pass-312143"}) is None


def test_explicit_dark_flat_id_is_honoured_across_proposals():
    dark_flat = _run("pass-314022")
    client = {"uid-246": dark_flat}
    start = {"data_session": "pass-312143"}
    assert resolve_dark_flat_run(client, start, dark_flat_id="uid-246") is dark_flat
