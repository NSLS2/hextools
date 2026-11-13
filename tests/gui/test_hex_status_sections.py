import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from bluesky import RunEngine  # noqa: E402
from ophyd_async.core import init_devices  # noqa: E402
from qtpy.QtWidgets import QApplication, QComboBox  # noqa: E402

from hextools.catalog.generate import type_expr  # noqa: E402
from hextools.catalog.model import DeviceRegistry  # noqa: E402
from hextools.gui import _ipython  # noqa: E402
from hextools.gui.hex_config import (  # noqa: E402
    filter_rows,
    slits_entries,
    status_sections,
)
from hextools.gui.movable_group import QtMovableGroup  # noqa: E402
from hextools.gui.run_engine_clients import InProcessRunEngineClient  # noqa: E402
from hextools.gui.signal_status_group import QtBeamlineStatus  # noqa: E402
from hextools.machine import NSLS2StorageRing  # noqa: E402
from hextools.motors import SampleTower  # noqa: E402
from hextools.photon_delivery_system.dclm import DCLM  # noqa: E402
from hextools.photon_delivery_system.filters import (  # noqa: E402
    Filter,
    Filter1UpstreamPosition,
    Filter2Position,
)
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
        "RE": RE,
    }
    client = InProcessRunEngineClient(namespace)
    widget = QtBeamlineStatus(status_sections(client), client)
    try:
        sections = {
            group.title: [s.label_widget.text() for s in group._statuses]
            for group in widget._groups
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
    assert status_sections(InProcessRunEngineClient({"RE": RunEngine()})) == []


def test_filter_rows_offer_only_filters_one_and_two():
    RunEngine()
    names = ["filter1_upstream", "filter1_downstream", "filter2", "filter3", "filter4"]
    with init_devices(mock=True):
        namespace = {n: Filter(f"{n}:", f"{n}:Sw", name=n) for n in names}
    rows, _ = filter_rows(DeviceRegistry.from_namespace(namespace))
    assert list(rows.values()) == names[:3]


def test_aliases_resolve_to_the_name_the_registry_keeps():
    RunEngine()
    with init_devices(mock=True):
        filter2 = Filter("F2:", "F2:Sw", name="filter2")
        a_slits = Slits("A:", name="a_slits")
    namespace = {"alias": filter2, "filter2": filter2, "a": a_slits, "a_slits": a_slits}
    registry = DeviceRegistry.from_namespace(namespace)
    rows, _ = filter_rows(registry)
    assert list(rows.values()) == ["filter2"]
    assert slits_entries(registry) == [("a_slits", "A Slits")]


def test_filter_group_batches_material_settings_into_move_filters(monkeypatch):
    app = QApplication.instance() or QApplication([])
    RE = RunEngine()
    with init_devices(mock=True):
        filter2 = Filter("F2:", "F2:Sw", name="filter2")
        filter1_upstream = Filter("F1U:", "F1U:Sw", name="filter1_upstream")
    namespace = {"filter2": filter2, "filter1_upstream": filter1_upstream, "RE": RE}
    client = InProcessRunEngineClient(namespace)
    rows, value_types = filter_rows(client.device_registry)
    assert rows == {"Filter 1 Upstream": "filter1_upstream", "Filter 2": "filter2"}
    assert value_types == {
        "filter1_upstream": type_expr(Filter1UpstreamPosition),
        "filter2": type_expr(Filter2Position),
    }

    commands: list[str] = []
    monkeypatch.setattr(_ipython, "run_in_ipython", commands.append)
    group = QtMovableGroup(
        rows,
        client,
        value_types=value_types,
        move_plan="move_filters",
        tweak_plan=None,
        batch=True,
    )
    try:
        assert group._batch.isChecked()
        upstream, f2 = group.controls
        assert upstream._tweak is None
        for control, member in (
            (upstream, list(Filter1UpstreamPosition)[1]),
            (f2, list(Filter2Position)[0]),
        ):
            combo = control.findChild(QComboBox)
            assert combo is not None
            combo.activated.emit(combo.findData(member.value))
        group._on_move_clicked()
        app.processEvents()
        assert commands == [
            "RE(move_filters(filter1_upstream, "
            f"{list(Filter1UpstreamPosition)[1].value!r}, "
            f"filter2, {list(Filter2Position)[0].value!r}))"
        ]
    finally:
        group.close()
        app.processEvents()
