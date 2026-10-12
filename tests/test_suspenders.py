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


@pytest.mark.parametrize("legacy", ["beam", "mode", "both"])
def test_legacy_ophyd_signals(RE: RunEngine, ring: NSLS2StorageRing, legacy):
    # SuspendFloor also takes ophyd signals, whose callbacks pass value=...
    # rather than a reading; both sides must work in either style (PR 93 review).
    from ophyd import Signal

    beam = Signal(name="beam", value=450.0) if legacy != "mode" else ring.beam_current
    mode = (
        Signal(name="mode", value=NSLS2OpsMode.OPERATIONS)
        if legacy != "beam"
        else ring.operating_mode
    )

    def set_beam(v):
        beam.put(v) if isinstance(beam, Signal) else set_mock_value(beam, v)

    def set_mode(m):
        mode.put(m) if isinstance(mode, Signal) else set_mock_value(mode, m)

    susp = SuspendFloorUnlessOpsMode(
        beam, 100, resume_thresh=390, ops_mode=mode, skip_modes=SKIP
    )
    RE.install_suspender(susp)
    try:
        set_beam(50)
        assert susp.tripped
        set_mode(NSLS2OpsMode.MAINTENANCE)
        assert not susp.tripped
        set_mode(NSLS2OpsMode.OPERATIONS)
        assert susp.tripped
    finally:
        RE.remove_suspender(susp)


def test_beam_update_during_mode_change_is_not_lost(RE: RunEngine):
    # PR 93 review: a beam reading arriving on another thread while a mode
    # change re-evaluates must not be overwritten by the stale replayed value.
    import threading

    from ophyd import Signal

    beam = Signal(name="beam", value=0.0)
    mode = Signal(name="mode", value=NSLS2OpsMode.OPERATIONS)

    class Racing(SuspendFloorUnlessOpsMode):
        armed = False
        reads = 0

        @property
        def _last_value(self):
            value = self.__dict__.get("_lv")
            if self.armed and threading.current_thread() is threading.main_thread():
                self.reads += 1
                if self.reads == 2:  # after the None check, before the replay
                    self.armed = False
                    other = threading.Thread(target=beam.put, args=(450.0,))
                    other.start()
                    other.join(timeout=0.5)
                    self._racer = other
            return value

        @_last_value.setter
        def _last_value(self, value):
            self.__dict__["_lv"] = value

    susp = Racing(beam, 100, resume_thresh=390, ops_mode=mode, skip_modes=SKIP)
    RE.install_suspender(susp)
    try:
        beam.put(50.0)
        assert susp.tripped
        susp.armed = True
        mode.put(NSLS2OpsMode.MAINTENANCE)
        susp._racer.join()
        mode.put(NSLS2OpsMode.OPERATIONS)
        assert not susp.tripped, "judged against the stale beam current"
    finally:
        RE.remove_suspender(susp)
