import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from bluesky import RunEngine  # noqa: E402
from ophyd_async.core import StandardReadable, init_devices, soft_signal_rw  # noqa: E402
from ophyd_async.core import StandardReadableFormat as Format  # noqa: E402
from ophyd_async.sim import SimMotor  # noqa: E402
from qtpy.QtWidgets import QApplication, QGroupBox  # noqa: E402

from hextools.gui.reading_sources import NamespaceReadingSource  # noqa: E402
from hextools.gui.signal_status_group import (  # noqa: E402
    QtBeamlineStatus,
    QtDeviceStatus,
    QtStatusGroup,
    field_label,
    hinted_fields,
)


class _Stage(StandardReadable):
    def __init__(self, name=""):
        with self.add_children_as_readables():
            self.x = SimMotor()
        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.temperature = soft_signal_rw(float, units="degC")
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.mode = soft_signal_rw(str)
        super().__init__(name=name)


@pytest.fixture
def RE():
    return RunEngine()


@pytest.fixture
def stage(RE):
    with init_devices(mock=True):
        stage = _Stage(name="sample-stage")
    return stage


@pytest.fixture
def source(RE, stage):
    return NamespaceReadingSource({"stage": stage}, RE.loop)


def test_hinted_fields_map_to_attribute_paths(stage):
    fields = hinted_fields(stage)
    assert set(fields) == set(stage.hints["fields"])
    assert fields["sample-stage-temperature"] == "temperature"
    assert fields["sample-stage-x"] == "x.user_readback"


@pytest.mark.parametrize(
    "field, expected",
    [
        ("sample-stage-x", "x"),
        ("sample-stage-temperature", "temperature"),
        ("sample-stage", "sample-stage"),
        ("other", "other"),
    ],
)
def test_field_label_strips_device_name(field, expected):
    assert field_label("sample-stage", field) == expected


def test_device_status_titles_box_and_labels_rows(source):
    app = QApplication.instance() or QApplication([])
    widget = QtDeviceStatus("stage", source)
    try:
        assert widget.title == "sample-stage"
        labels = sorted(s.label_widget.text() for s in widget._statuses)
        assert labels == ["temperature:", "x:"]
    finally:
        widget.close()
        app.processEvents()


def test_device_status_uses_display_name(source):
    app = QApplication.instance() or QApplication([])
    widget = QtDeviceStatus("stage", source, display_name="Sample Stage")
    try:
        assert widget.title == "Sample Stage"
    finally:
        widget.close()
        app.processEvents()


def test_device_status_rejects_unknown_name(source):
    if QApplication.instance() is None:
        QApplication([])
    with pytest.raises(KeyError):
        QtDeviceStatus("no_such_device", source)


def test_beamline_status_packs_sections_into_one_box(RE):
    app = QApplication.instance() or QApplication([])
    with init_devices(mock=True):
        stage0 = _Stage(name="stage0")
        stage1 = _Stage(name="stage1")
        stage2 = _Stage(name="stage2")
    namespace = {"stage0": stage0, "stage1": stage1, "stage2": stage2}
    widget = QtBeamlineStatus(
        list(namespace), NamespaceReadingSource(namespace, RE.loop), columns=2
    )
    try:
        assert [s.title for s in widget._groups] == [
            "stage0",
            "stage1",
            "stage2",
        ]
        boxes = widget.findChildren(QGroupBox)
        assert [box.title() for box in boxes] == ["Beamline Status"]
    finally:
        widget.close()
        app.processEvents()


def test_long_sections_split_fields_over_two_columns(RE):
    app = QApplication.instance() or QApplication([])
    signals = {f"f{i}": soft_signal_rw(float, name=f"f{i}") for i in range(7)}
    group = QtStatusGroup(
        "Many",
        {name: name for name in signals},
        NamespaceReadingSource(signals, RE.loop),
    )
    try:
        assert group.rows == 4
    finally:
        group.close()
        app.processEvents()


def test_beamline_status_uses_mapping_keys_as_display_names(source):
    app = QApplication.instance() or QApplication([])
    widget = QtBeamlineStatus({"Sample Stage": "stage"}, source)
    try:
        assert widget._groups[0].title == "Sample Stage"
    finally:
        widget.close()
        app.processEvents()
