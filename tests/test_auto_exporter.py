"""The auto-exporter picks an exporter from the run's Tiled spec."""

import csv
import logging
from types import SimpleNamespace
from typing import cast

import pytest
from bluesky_tiled_plugins.clients.bluesky_run import BlueskyRunV3

from hextools.exporters.auto_exporter import (
    CSV_COLUMNS,
    EXPORTERS,
    _record_run,
    auto_export_run,
)
from hextools.exporters.edxd import export_edxd_run_to_h5
from hextools.specs import EDXD_CALIBRATION_V1, EDXD_SCAN_V1


def _run(**start) -> BlueskyRunV3:
    start = {"uid": "new-uid", "time": 0.0, "scan_id": 2, "plan_name": "count"} | start
    run = SimpleNamespace(start=start, stop={"exit_status": "success"}, specs=[])
    return cast(BlueskyRunV3, run)


# Before #139 the columns were scan_id, plan_name, uid, status, export_status.
def test_record_run_starts_a_new_file_when_the_header_differs(tmp_path):
    csv_path = tmp_path / "plan_history.csv"
    csv_path.write_text(
        "scan_id,plan_name,uid,status,export_status\n1,count,old-uid,success,success\n"
    )
    _record_run(csv_path, _run(), succeeded=True)

    with csv_path.open(newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == list(CSV_COLUMNS)
    assert rows[1][CSV_COLUMNS.index("uid")] == "new-uid"
    [old] = [p for p in tmp_path.iterdir() if p != csv_path]
    assert "old-uid" in old.read_text()


def test_record_run_warns_without_traceback_when_the_directory_is_missing(
    tmp_path, caplog
):
    csv_path = tmp_path / "no-such-proposal" / "plan_history.csv"
    with caplog.at_level(logging.WARNING, logger="hextools.exporters.auto_exporter"):
        _record_run(csv_path, _run(), succeeded=True)

    assert not csv_path.parent.exists()
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert record.exc_info is None


def test_run_without_proposal_fields_warns_without_traceback(caplog):
    with caplog.at_level(logging.WARNING, logger="hextools.exporters.auto_exporter"):
        auto_export_run(_run(), catalog=None)  # type: ignore[arg-type]

    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert record.exc_info is None


# Calibration runs logged "no exporter registered" (2026-10-28); they export
# with the same layout as EDXD scans.
@pytest.mark.parametrize("spec", [EDXD_SCAN_V1, EDXD_CALIBRATION_V1])
def test_edxd_runs_use_the_edxd_exporter(spec):
    assert EXPORTERS[(spec.name, spec.version)] is export_edxd_run_to_h5
