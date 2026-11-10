import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from bluesky import RunEngine  # noqa: E402
from ophyd_async.core import init_devices  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from hextools.gui.device_status import QtBeamlineStatus  # noqa: E402
from hextools.gui.hex_config import status_sections  # noqa: E402
from hextools.machine import NSLS2StorageRing  # noqa: E402
from hextools.motors import SampleTower  # noqa: E402
from hextools.photon_delivery_system.dclm import DCLM  # noqa: E402
from hextools.photon_delivery_system.filters import Filter  # noqa: E402
from hextools.photon_delivery_system.shutter import Shutter  # noqa: E402
from hextools.photon_delivery_system.slits import FrontEndSlit, Slits  # noqa: E402


def test_status_sections_cover_beamline_devices():
    app = QApplication.instance() or QApplication([])
    RE = RunEngine()
    with init_devices(mock=True):
        storage_ring = NSLS2StorageRing()
        fe_slit1 = FrontEndSlit("FE:", name="fe-slit1")
        filter1_upstream = Filter("F1U:", "F1U:Sw", name="filter1_upstream")
        filter1_downstream = Filter("F1D:", "F1D:Sw", name="filter1_downstream")
        filter2 = Filter("F2:", "F2:Sw", name="filter2")
        photon_shutter = Shutter("PH:", name="photon-shutter")
        monochromator = DCLM("MONO:", name="monochromator")
        a_slits = Slits("A:", name="a-slits")
        edxd_slits = Slits("E:", name="edxd-slits")
        sample_tower = SampleTower("TOWER:", name="sample-tower")
    namespace = {
        "storage_ring": storage_ring,
        "fe_slit1": fe_slit1,
        "filter2": filter2,
        "filter1_downstream": filter1_downstream,
        "filter1_upstream": filter1_upstream,
        "photon_shutter": photon_shutter,
        "monochromator": monochromator,
        "mono": monochromator,
        "a_slits": a_slits,
        "edxd_slits": edxd_slits,
        "sample_tower": sample_tower,
        "_hidden": sample_tower,
    }
    widget = QtBeamlineStatus(status_sections(namespace, RE.loop), RE.loop)
    try:
        sections = {
            group.title: [s.label_widget.text() for s in group._statuses]
            for group in widget._device_statuses
        }
        slit_fields = ["H Gap:", "H Center:", "V Gap:", "V Center:"]
        assert sections == {
            "Storage Ring": ["Current:", "Mode:"],
            "Front-End Slit": ["Aperture:"],
            "Filters": ["1 Upstream:", "1 Downstream:", "2:"],
            "Monochromator": ["Beam Mode:", "Energy:"],
            "A Slits": slit_fields,
            "EDXD Slits": slit_fields,
            "Sample Tower": ["y:", "pitch:", "roll:"],
        }
    finally:
        widget.close()
        app.processEvents()


def test_status_sections_skip_missing_devices():
    assert status_sections({}, RunEngine().loop) == []
