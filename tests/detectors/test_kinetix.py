from pathlib import Path

import pytest
from ophyd_async.core import (
    StaticFilenameProvider,
    StaticPathProvider,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.adcore import NDProcessIO

from hextools.detectors.kinetix import kinetix_factory


@pytest.fixture
def kinetix(RE, tmp_path: Path):
    with init_devices(mock=True):
        det = kinetix_factory(
            1,
            StaticPathProvider(StaticFilenameProvider("scan"), tmp_path),
            name="kinetix-det1",
        )
    set_mock_value(det.hdf.file_path_exists, True)
    return det


@pytest.mark.parametrize("num", (1, 2, 3, 4))
def test_kinetix_factory_uses_the_hex_pv_prefix(RE, tmp_path: Path, num: int):
    """The detector number selects the camera; the rest of the prefix is fixed."""
    with init_devices(mock=True):
        det = kinetix_factory(
            num,
            StaticPathProvider(StaticFilenameProvider("scan"), tmp_path),
            name=f"kinetix-det{num}",
        )
    assert f"XF:27ID1-BI{{Kinetix-Det:{num}}}cam1:Acquire" in det.driver.acquire.source


def test_kinetix_factory_wires_a_proc_plugin(kinetix):
    """The HDF writer sits behind Proc1 so frames can be averaged.

    Without the process plugin the detector cannot squash exposures, which is
    what the HEX tomography plans rely on.
    """
    assert isinstance(kinetix.proc, NDProcessIO)
    assert "XF:27ID1-BI{Kinetix-Det:1}Proc1:NumFilter" in kinetix.proc.num_filter.source


def test_kinetix_factory_keeps_the_name_it_was_given(kinetix):
    """The name becomes the asset directory, so a mangled one misfiles data."""
    assert kinetix.name == "kinetix-det1"
