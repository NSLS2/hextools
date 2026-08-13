"""Tests for the HEX Kinetix detector module (hextools.detectors.kinetix)."""

import asyncio
from pathlib import Path

import bluesky.plans as bp
import pytest
from bluesky.run_engine import RunEngine
from ophyd_async.core import (
    callback_on_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.adcore import ADImageMode

from hextools.detectors.kinetix import (
    HEXKinetixDetector,
    SettablePathProvider,
    make_kinetix,
    set_output_dir,
)


@pytest.fixture
def kinetix(RE: RunEngine) -> HEXKinetixDetector:
    """Mock kinetix1 wired the way the real IOC behaves: directory exists,
    one frame lands per acquisition start, NumCaptured resets per capture
    session."""
    with init_devices(mock=True):
        detector = make_kinetix(1)

    set_mock_value(detector.hdf.file_path_exists, True)

    frames = {"count": 0}

    def on_capture(value, **kwargs):
        if value:
            frames["count"] = 0
            set_mock_value(detector.hdf.num_captured, 0)

    def on_acquire(value, **kwargs):
        if value:
            frames["count"] += 1
            set_mock_value(detector.hdf.num_captured, frames["count"])

    callback_on_mock_put(detector.hdf.capture, on_capture)
    callback_on_mock_put(detector.driver.acquire, on_acquire)
    return detector


def test_make_kinetix_rejects_unknown_id():
    with pytest.raises(ValueError, match="1 or 3"):
        make_kinetix(2)


def test_default_source_port_per_camera(RE: RunEngine):
    # Det:1 routes through the transform plugin, Det:3 straight off the camera
    with init_devices(mock=True):
        det1 = make_kinetix(1)
        det3 = make_kinetix(3)
    assert det1.default_source_port == "TRANS1"
    assert det3.default_source_port == "KTX1"


def test_settable_path_provider_retargets(tmp_path: Path):
    provider = SettablePathProvider()
    provider.set(tmp_path / "scan_00001", "proj")
    info = provider()
    assert info.directory_path == tmp_path / "scan_00001"
    assert info.filename == "proj"
    # the writer may create the scan_NNNNN levels under the proposal
    assert info.create_dir_depth == -4


def test_set_output_dir_requires_settable_provider():
    class NotAKinetix:
        name = "imposter"

    with pytest.raises(TypeError, match="make_kinetix"):
        set_output_dir(NotAKinetix(), "/tmp/nowhere")


def test_count_produces_events_and_restores_live_view(
    RE: RunEngine, kinetix: HEXKinetixDetector, tmp_path: Path
):
    docs: list[tuple[str, dict]] = []
    RE.subscribe(lambda name, doc: docs.append((name, doc)))
    set_output_dir(kinetix, tmp_path, "img")
    result = RE(bp.count([kinetix], num=3))
    assert result.exit_status == "success"
    assert len([d for n, d in docs if n == "event"]) == 3

    # HEX workarounds observable under mock: SWMR forced off at prepare;
    # unstage returns the camera to Continuous free-run (live view resumes)
    assert asyncio.run(kinetix.hdf.swmr_mode.get_value()) is False
    assert (
        asyncio.run(kinetix.driver.image_mode.get_value()) == ADImageMode.CONTINUOUS
    )
