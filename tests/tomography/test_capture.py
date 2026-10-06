from pathlib import Path

import pytest
from bluesky.run_engine import RunEngine
from ophyd_async.core import StaticPathProvider, UUIDFilenameProvider, init_devices
from ophyd_async.sim import PatternGenerator, SimBlobDetector

from hextools.tomography.flyscans import capture


@pytest.fixture
def blob(RE: RunEngine, tmp_path: Path) -> SimBlobDetector:
    path_provider = StaticPathProvider(UUIDFilenameProvider(), tmp_path)
    with init_devices():
        blob = SimBlobDetector(path_provider, PatternGenerator())
    return blob


# capture is for any detector, not only the Phantom (PR 92 review).
def test_capture_takes_one_reading_of_num_images_frames(RE: RunEngine, blob):
    docs: dict[str, list] = {}
    RE(
        capture([blob], num_images=3, exposure_time=0.01),
        lambda name, doc: docs.setdefault(name, []).append(doc),
    )

    assert docs["start"][0]["plan_name"] == "capture"
    assert docs["start"][0]["num_images"] == 3
    assert len(docs["event"]) == 1
    shapes = docs["descriptor"][0]["data_keys"]["blob"]["shape"]
    assert shapes[0] == 3, f"expected 3 frames in the one reading, got {shapes}"
