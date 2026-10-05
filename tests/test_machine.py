import pytest
from ophyd_async.core import init_devices, set_mock_value

from hextools.machine import NSLS2StorageRing


@pytest.fixture
def storage_ring(RE) -> NSLS2StorageRing:
    with init_devices(mock=True):
        ring = NSLS2StorageRing()
    return ring


def test_storage_ring_reads_the_facility_dcct(storage_ring: NSLS2StorageRing):
    """The beam current comes from the facility DCCT, not a beamline PV.

    This is the signal the beam-drop suspender watches, so the PV is part of
    the contract with the accelerator rather than something HEX can rename.
    """
    assert "SR:OPS-BI{DCCT:1}I:Real-I" in storage_ring.beam_current.source


async def test_storage_ring_reports_beam_current_as_configuration(
    storage_ring: NSLS2StorageRing,
):
    set_mock_value(storage_ring.beam_current, 398.5)
    config = await storage_ring.read_configuration()
    assert any(v["value"] == 398.5 for v in config.values())
