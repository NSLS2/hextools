"""Exporting a tomography flyscan that did not finish.

At HEX on 2026-10-26, scan 14 (3601 projections x 4 images) was aborted after 570
projections. The run recorded 570 x 4 PandA captures, but the PandA kept capturing
until it was disarmed and its file held 2335, so Tiled refused to serve the capture
(422 "Can not reshape (2335,) array data to (570, 4)") and the auto-export failed.
"""

import csv
from pathlib import Path
from unittest.mock import MagicMock, Mock

import h5py
import numpy as np
import pytest
from bluesky_tiled_plugins import CatalogOfBlueskyRuns
from bluesky_tiled_plugins.clients.bluesky_run import BlueskyRunV3

from hextools.exporters import auto_exporter, tomography
from hextools.exporters.utils import read_recorded_captures
from hextools.specs import TOMO_FLYSCAN_V1

RECORDED, IMAGES, IN_FILE = 570, 4, 2335


def _run(
    exit_status: str, rotation_angle: Mock | None = None, recorded: int = RECORDED
) -> MagicMock:
    run = MagicMock(spec=BlueskyRunV3)
    run.start = {
        "uid": "uid-14",
        "scan_id": 14,
        "plan_name": "tomo_flyscan",
        "data_session": "pass-319162",
        "cycle": "2026-3",
        "num_points": 3601,
        "images_to_average": IMAGES,
    }
    run.stop = {"exit_status": exit_status, "num_events": {"primary": recorded}}
    run.specs = [TOMO_FLYSCAN_V1]
    run.__getitem__.side_effect = {"primary": {"rotation_angle": rotation_angle}}.get
    return run


def _csv_rows(csv_path: Path) -> list[tuple[str, str]]:
    rows = csv.DictReader(csv_path.read_text().splitlines())
    return [(r["status"], r["export_status"]) for r in rows]


@pytest.mark.parametrize("exit_status", ["abort", "stop", "fail"])
def test_auto_export_skips_a_run_that_did_not_finish(
    monkeypatch, tmp_path: Path, exit_status: str
):
    exporter = Mock(__name__="export_tomo_flyscan_to_h5")
    key = (TOMO_FLYSCAN_V1.name, TOMO_FLYSCAN_V1.version)
    monkeypatch.setitem(auto_exporter.EXPORTERS, key, exporter)
    csv_path = tmp_path / "exports.csv"

    auto_exporter.auto_export_run(_run(exit_status), Mock(), csv_path=csv_path)

    exporter.assert_not_called()
    assert _csv_rows(csv_path) == [(exit_status, "skipped")]


def test_auto_export_runs_exporters_for_a_finished_run(monkeypatch, tmp_path: Path):
    exporter = Mock(__name__="export_tomo_flyscan_to_h5")
    key = (TOMO_FLYSCAN_V1.name, TOMO_FLYSCAN_V1.version)
    monkeypatch.setitem(auto_exporter.EXPORTERS, key, exporter)
    csv_path = tmp_path / "exports.csv"

    auto_exporter.auto_export_run(
        _run("success", recorded=3601), Mock(), csv_path=csv_path
    )

    exporter.assert_called_once()
    assert _csv_rows(csv_path) == [("success", "success")]


def _panda_file(path: Path) -> np.ndarray:
    angles = np.arange(IN_FILE, dtype=float)
    with h5py.File(path, "w") as f:
        f["/Angle"] = angles
    return angles


def test_reads_only_the_captures_the_run_recorded(tmp_path: Path):
    angles = _panda_file(tmp_path / "panda.h5")
    values = read_recorded_captures(tmp_path / "panda.h5", "/Angle", RECORDED, IMAGES)
    np.testing.assert_array_equal(
        values, angles[: RECORDED * IMAGES].reshape(RECORDED, IMAGES)
    )


def test_refuses_a_file_shorter_than_the_run(tmp_path: Path):
    _panda_file(tmp_path / "panda.h5")
    with pytest.raises(ValueError, match="fewer"):
        read_recorded_captures(tmp_path / "panda.h5", "/Angle", 600, IMAGES)


@pytest.fixture
def export_dir(monkeypatch, tmp_path: Path) -> Path:
    """Export under tmp_path, with the dark/flat, baseline and json steps stubbed."""
    monkeypatch.setattr(tomography, "Path", lambda p: tmp_path / str(p).lstrip("/"))
    monkeypatch.setattr(tomography, "resolve_dark_flat_run", lambda *a: None)
    monkeypatch.setattr(tomography, "create_baseline_dataset", lambda *a: None)
    monkeypatch.setattr(tomography, "run_metadata_to_json", lambda *a: None)
    scan_dir = tmp_path / "nsls2/data/hex/proposals/2026-3/pass-319162"
    path = scan_dir / "tomography/metadata/scan_00014"
    path.mkdir(parents=True)
    return path


def _export(monkeypatch, run: MagicMock, export_dir: Path) -> np.ndarray:
    datasets = {
        "kinetix-det1": ("det.h5", "/entry/data/data", 4),
        "rotation_angle": ("../panda.h5", "/Angle", 2),
    }
    monkeypatch.setattr(tomography, "find_hdf_datasets", lambda *a: datasets)
    tomography.export_tomo_flyscan_to_h5(run, MagicMock(spec=CatalogOfBlueskyRuns))
    with h5py.File(export_dir / "scan_00014.nxs") as f:
        return f["/entry/data/rotation_angle"][()]


def test_exports_the_recorded_projections_of_an_aborted_run(
    monkeypatch, export_dir: Path
):
    angles = _panda_file(export_dir.parent / "panda.h5")
    rotation_angle = Mock(shape=(RECORDED, IMAGES))
    rotation_angle.read.side_effect = RuntimeError(
        "422: Can not reshape (2335,) array data to (570, 4)"
    )

    exported = _export(monkeypatch, _run("abort", rotation_angle), export_dir)

    expected = angles[: RECORDED * IMAGES].reshape(RECORDED, IMAGES).mean(axis=1)
    np.testing.assert_allclose(exported, expected)


def test_a_finished_run_is_still_read_through_tiled(monkeypatch, export_dir: Path):
    captures = np.arange(3601 * IMAGES, dtype=float).reshape(3601, IMAGES)
    rotation_angle = Mock(shape=captures.shape)
    rotation_angle.read.return_value = captures

    run = _run("success", rotation_angle, recorded=3601)
    exported = _export(monkeypatch, run, export_dir)

    np.testing.assert_allclose(exported, captures.mean(axis=1))
