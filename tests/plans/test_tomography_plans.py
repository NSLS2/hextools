"""
Mock-tier tests for the tomography plan family (hextools.plans.tomography).

A representative slice of the full behavioral suites, which live with the
simulated beamline in NSLS2/hex-ob (tests/plans_mock_test.py and the
sim-tier files there): each plan here runs end-to-end under the RunEngine
against mock devices, asserting per-stream event counts and the restore
behaviors the plans promise.
"""

import asyncio
from collections import Counter
from pathlib import Path

import bluesky.plan_stubs as bps
import pytest
from bluesky.run_engine import RunEngine
from ophyd_async.core import (
    callback_on_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.core import epics_signal_rw
from ophyd_async.epics.motor import Motor

from hextools.detectors.kinetix import make_kinetix
from hextools.plans.tomography import (
    alignment_scan,
    run_multiple_scans,
    scan_1d,
    take_dark_flat,
    take_radiograph,
)


class MockBeamline:
    """Mock kinetix + motors + shutter signals, wired so plans run
    end-to-end (one frame per acquisition start, moves complete
    immediately).  Port of hex-ob tests/mock_beamline.py, bound to the
    session RE fixture."""

    def __init__(self, RE: RunEngine):
        self.RE = RE
        with init_devices(mock=True):
            detector = make_kinetix(1)
            rot_stage = Motor("XF:MOCK{MC:5-Ax:4}Mtr", name="rot_stage")
            sample_x = Motor("XF:MOCK{SMPL:1-Ax:X1}Mtr", name="sample_x")
            ph_open_cmd = epics_signal_rw(int, "XF:MOCK{Sh}Opn", name="ph_open_cmd")
            ph_close_cmd = epics_signal_rw(int, "XF:MOCK{Sh}Cls", name="ph_close_cmd")
        self.detector = detector
        self.rot_stage = rot_stage
        self.sample_x = sample_x
        self.ph_open_cmd = ph_open_cmd
        self.ph_close_cmd = ph_close_cmd

        set_mock_value(detector.hdf.file_path_exists, True)
        self._frames = 0

        def on_capture(value, **kwargs):
            if value:
                self._frames = 0
                set_mock_value(detector.hdf.num_captured, 0)

        def on_acquire(value, **kwargs):
            if value:
                self._frames += 1
                set_mock_value(detector.hdf.num_captured, self._frames)

        callback_on_mock_put(detector.hdf.capture, on_capture)
        callback_on_mock_put(detector.driver.acquire, on_acquire)
        for motor in (rot_stage, sample_x):
            callback_on_mock_put(
                motor.user_setpoint,
                lambda value, *, m=motor, **kw: set_mock_value(m.user_readback, value),
            )

        self.docs: list[tuple[str, dict]] = []
        RE.subscribe(lambda name, doc: self.docs.append((name, doc)))

    def events_by_stream(self) -> dict[str, int]:
        names = {
            doc["uid"]: doc["name"]
            for kind, doc in self.docs
            if kind == "descriptor"
        }
        return dict(
            Counter(
                names[doc["descriptor"]]
                for kind, doc in self.docs
                if kind == "event"
            )
        )

    def readback(self, motor) -> float:
        return asyncio.run(motor.user_readback.get_value())


@pytest.fixture
def bl(RE: RunEngine) -> MockBeamline:
    return MockBeamline(RE)


def test_take_radiograph(bl: MockBeamline, tmp_path: Path):
    result = bl.RE(take_radiograph(
        bl.detector, bl.ph_open_cmd, bl.ph_close_cmd,
        output_dir=str(tmp_path), exposure_time=0.01, num_images=3,
    ))
    assert result.exit_status == "success"
    assert bl.events_by_stream() == {"primary": 3}


def test_scan_1d_leaves_motor_at_stop(bl: MockBeamline, tmp_path: Path):
    result = bl.RE(scan_1d(
        bl.detector, bl.rot_stage, bl.ph_open_cmd, bl.ph_close_cmd,
        output_dir=str(tmp_path), exposure_time=0.01,
        start=0.0, stop=9.0, num_points=4,
    ))
    assert result.exit_status == "success"
    assert bl.events_by_stream() == {"primary": 4}
    assert bl.readback(bl.rot_stage) == 9.0


def test_take_dark_flat_restores_sample(bl: MockBeamline, tmp_path: Path):
    result = bl.RE(take_dark_flat(
        bl.detector, bl.sample_x, bl.ph_open_cmd, bl.ph_close_cmd,
        output_dir=str(tmp_path), exposure_time=0.01,
        num_dark=2, num_flat=3, flat_x_offset=5.0,
    ))
    assert result.exit_status == "success"
    assert bl.events_by_stream() == {"dark": 2, "flat": 3}
    assert bl.readback(bl.sample_x) == 0.0


def test_alignment_scan_restores_rotation(bl: MockBeamline, tmp_path: Path):
    bl.RE(bps.mv(bl.rot_stage, 3.0))
    result = bl.RE(alignment_scan(
        bl.detector, bl.rot_stage, bl.sample_x,
        bl.ph_open_cmd, bl.ph_close_cmd,
        output_dir=str(tmp_path), exposure_time=0.01,
        num_projections=5, start_angle=0.0, stop_angle=90.0,
        flat_x_offset=1.0, num_flats=2,
    ))
    assert result.exit_status == "success"
    assert bl.events_by_stream() == {"primary": 5, "flat": 2}
    assert bl.readback(bl.rot_stage) == 3.0


def test_run_multiple_scans_orchestration(bl: MockBeamline, tmp_path: Path):
    calls: list[tuple[str, int]] = []

    def stub(kind):
        def _plan(output_dir, index):
            calls.append((kind, index))
            yield from bps.null()
        return _plan

    bl.RE(bps.mv(bl.sample_x, 0.0))
    result = bl.RE(run_multiple_scans(
        bl.detector, None, bl.rot_stage, bl.sample_x,
        output_base_dir=str(tmp_path),
        exposure_time=0.01, num_projections=61, flat_x_offset=5.0,
        scan_motor=bl.sample_x, start=-1.0, stop=1.0, num_points=3,
        dark_flat_every=2,
        ph_open_cmd=bl.ph_open_cmd, ph_close_cmd=bl.ph_close_cmd,
        scan_plan=stub("scan"), dark_flat_plan=stub("df"),
    ))
    assert result.exit_status == "success"
    assert [c[0] for c in calls] == ["scan", "scan", "df", "scan"]
    assert bl.readback(bl.sample_x) == 0.0
