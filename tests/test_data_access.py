from __future__ import annotations

from pathlib import Path
from pprint import pprint

import pytest

from hextools.germ.export import (
    GERM_DETECTOR_KEYS,
    get_detector_parameters_from_tiled,
    nx_export,
)


@pytest.mark.tiled()
def test_detector_params_from_tiled(tiled_run_with_germ_data):
    run = tiled_run_with_germ_data
    det_params = get_detector_parameters_from_tiled(
        run, det_name="GeRM", keys=GERM_DETECTOR_KEYS
    )
    pprint(det_params)


@pytest.mark.tiled()
def test_nx_export_germ_detector(tiled_run_with_germ_data):
    run = tiled_run_with_germ_data
    nx_file_path = nx_export(run, det_name="GeRM")
    assert Path.exists(Path(nx_file_path))
