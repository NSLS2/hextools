from collections.abc import Iterator

import pytest
from bluesky.run_engine import RunEngine
from ophyd_async.core import init_devices, set_mock_value

from hextools.machine import NSLS2OpsMode, NSLS2StorageRing
from hextools.suspenders import SuspendFloorUnlessOpsMode

SKIP = (NSLS2OpsMode.MAINTENANCE, NSLS2OpsMode.SHUTDOWN)


@pytest.fixture
def ring(RE: RunEngine) -> NSLS2StorageRing:
    with init_devices(mock=True):
        storage_ring = NSLS2StorageRing()
    return storage_ring


@pytest.fixture
def suspender(
    RE: RunEngine, ring: NSLS2StorageRing
) -> Iterator[SuspendFloorUnlessOpsMode]:
    # Installed whatever the mode at startup: the mode is checked each time it trips.
    susp = SuspendFloorUnlessOpsMode(
        ring.beam_current,
        100,
        resume_thresh=390,
        ops_mode=ring.operating_mode,
        skip_modes=SKIP,
    )
    RE.install_suspender(susp)
    yield susp
    RE.remove_suspender(susp)


def test_trips_on_low_beam_in_operations(ring, suspender):
    set_mock_value(ring.beam_current, 50)
    assert suspender.tripped


@pytest.mark.parametrize("mode", SKIP)
def test_ignores_low_beam_in_maintenance_or_shutdown(ring, suspender, mode):
    set_mock_value(ring.operating_mode, mode)
    set_mock_value(ring.beam_current, 50)
    assert not suspender.tripped


def test_releases_when_ring_enters_maintenance_while_tripped(ring, suspender):
    set_mock_value(ring.beam_current, 50)
    assert suspender.tripped
    set_mock_value(ring.operating_mode, NSLS2OpsMode.MAINTENANCE)
    assert not suspender.tripped


def test_trips_when_operations_resume_with_beam_still_low(ring, suspender):
    set_mock_value(ring.operating_mode, NSLS2OpsMode.SHUTDOWN)
    set_mock_value(ring.beam_current, 0)
    assert not suspender.tripped
    set_mock_value(ring.operating_mode, NSLS2OpsMode.OPERATIONS)
    assert suspender.tripped


def test_resumes_on_beam_recovery_in_operations(ring, suspender):
    set_mock_value(ring.beam_current, 50)
    set_mock_value(ring.beam_current, 395)
    assert not suspender.tripped
