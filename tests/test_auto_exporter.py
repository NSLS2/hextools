"""The auto-exporter picks an exporter from the run's Tiled spec."""

import pytest

from hextools.exporters.auto_exporter import EXPORTERS
from hextools.exporters.edxd import export_edxd_run_to_h5
from hextools.specs import EDXD_CALIBRATION_V1, EDXD_SCAN_V1


# Calibration runs logged "no exporter registered" (2026-10-28); they export
# with the same layout as EDXD scans.
@pytest.mark.parametrize("spec", [EDXD_SCAN_V1, EDXD_CALIBRATION_V1])
def test_edxd_runs_use_the_edxd_exporter(spec):
    assert EXPORTERS[(spec.name, spec.version)] is export_edxd_run_to_h5
