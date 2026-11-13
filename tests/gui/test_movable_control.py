import os
from typing import cast

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from bluesky import RunEngine  # noqa: E402
from bluesky.run_engine import call_in_bluesky_event_loop  # noqa: E402
from ophyd_async.core import (  # noqa: E402
    StrictEnum,
    init_devices,
    soft_signal_rw,
)
from ophyd_async.sim import SimMotor  # noqa: E402
from qtpy.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from hextools.gui import movable_control, movable_group  # noqa: E402
from hextools.gui.movable_control import QtMovableControl, set_type  # noqa: E402
from hextools.gui.movable_group import QtMovableGroup  # noqa: E402
from hextools.gui.slits_control import QtSlitsControl  # noqa: E402
from hextools.photon_delivery_system.slits import Slits  # noqa: E402


class _Mode(StrictEnum):
    FAST = "Fast"
    SLOW = "Slow"


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def RE():
    return RunEngine()


@pytest.fixture
def commands(monkeypatch):
    ran: list[str] = []
    for module in (movable_control, movable_group):
        monkeypatch.setattr(module, "run_in_ipython", ran.append)
    return ran


@pytest.fixture
def namespace(RE):
    with init_devices(mock=True):
        motor = SimMotor(name="motor")
        slits = Slits("SLT:", name="slits")
    return {
        "motor": motor,
        "slits": slits,
        "count": soft_signal_rw(int, name="count"),
        "label": soft_signal_rw(str, name="label"),
        "mode": soft_signal_rw(_Mode, name="mode"),
        "flag": soft_signal_rw(bool, name="flag"),
    }


def _control(namespace, RE, path):
    return QtMovableControl(path, RE, namespace=namespace)


def _edit(widget: QWidget) -> QLineEdit:
    """The first line edit in ``widget``: a control's setpoint."""
    edit = widget.findChild(QLineEdit)
    assert edit is not None
    return edit


def _combo(widget: QWidget) -> QComboBox:
    combo = widget.findChild(QComboBox)
    assert combo is not None
    return combo


def _tweak(control: QtMovableControl) -> QLineEdit:
    assert control._tweak is not None
    return control._tweak


def test_set_type_reads_signal_datatype_and_set_hints(namespace):
    assert set_type(namespace["count"]) is int
    assert set_type(namespace["mode"]) is _Mode
    assert set_type(namespace["motor"]) is float


def test_setpoint_runs_mv_and_tweaks_run_mvr(app, RE, namespace, commands):
    control = _control(namespace, RE, "motor")
    try:
        setpoint = _edit(control)
        setpoint.setText("2.5")
        setpoint.returnPressed.emit()
        app.processEvents()
        _tweak(control).setText("0.1")
        control._on_tweak(-1)
        app.processEvents()
        assert commands == ["RE(mv(motor, 2.5))", "RE(mvr(motor, -0.1))"]
    finally:
        control.close()


def test_invalid_setpoint_is_not_sent(app, RE, namespace, commands):
    control = _control(namespace, RE, "count")
    try:
        setpoint = _edit(control)
        setpoint.setText("1.5")
        setpoint.returnPressed.emit()
        app.processEvents()
        assert commands == []
        assert setpoint.styleSheet()
    finally:
        control.close()


@pytest.mark.parametrize("path, has_tweak", [("count", True), ("label", False)])
def test_tweaks_only_for_numbers(app, RE, namespace, path, has_tweak):
    control = _control(namespace, RE, path)
    try:
        assert (control._tweak is not None) is has_tweak
    finally:
        control.close()


def test_string_setpoint_is_quoted(app, RE, namespace, commands):
    control = _control(namespace, RE, "label")
    try:
        setpoint = _edit(control)
        setpoint.setText("it's")
        setpoint.returnPressed.emit()
        app.processEvents()
        assert commands == ['RE(mv(label, "it\'s"))']
    finally:
        control.close()


@pytest.mark.parametrize(
    "path, choice, command",
    [
        ("mode", 1, "RE(mv(mode, 'Slow'))"),
        ("flag", 1, "RE(mv(flag, False))"),
    ],
)
def test_enum_and_bool_use_a_dropdown(
    app, RE, namespace, commands, path, choice, command
):
    control = _control(namespace, RE, path)
    try:
        combo = control.findChild(QComboBox)
        assert combo is not None and control._tweak is None
        combo.activated.emit(choice)
        app.processEvents()
        assert commands == [command]
    finally:
        control.close()


def test_rejects_unknown_paths_and_unsupported_types(app, RE, namespace):
    with pytest.raises(ValueError):
        _control(namespace, RE, "motor; import os")
    with pytest.raises(TypeError):
        _control(namespace, RE, "slits")


def test_explicit_value_type_covers_untyped_ophyd_v1_movables(
    app, RE, namespace, commands
):
    ophyd = pytest.importorskip("ophyd")
    namespace["axis"] = ophyd.Signal(name="axis", value=1.0)
    with pytest.raises(TypeError):
        _control(namespace, RE, "axis")
    control = QtMovableControl("axis", RE, namespace=namespace, value_type=float)
    try:
        assert control._tweak is not None
        _enter(control, "2.5")
        app.processEvents()
        assert commands == ["RE(mv(axis, 2.5))"]
    finally:
        control.close()


def test_plans_are_named_for_single_tweak_and_batch_moves(app, RE, namespace, commands):
    namespace["other"] = soft_signal_rw(float, name="other")
    group = QtMovableGroup(
        ["motor", "other"],
        RE,
        namespace=namespace,
        move_plan="custom_move",
        tweak_plan="bps.mvr",
    )
    try:
        motor, other = group.controls
        _enter(motor, "1")
        app.processEvents()
        _tweak(motor).setText("0.5")
        motor._on_tweak(1)
        app.processEvents()
        group.set_batch(True)
        _enter(motor, "2")
        _enter(other, "3")
        group._on_move_clicked()
        app.processEvents()
        assert commands == [
            "RE(custom_move(motor, 1.0))",
            "RE(bps.mvr(motor, 0.5))",
            "RE(custom_move(motor, 2.0, other, 3.0))",
        ]
    finally:
        group.close()


def test_no_tweak_plan_disables_tweaks_for_numbers(app, RE, namespace):
    control = QtMovableControl("motor", RE, namespace=namespace, tweak_plan=None)
    try:
        assert control._tweak is None
    finally:
        control.close()


@pytest.mark.parametrize("name", ["mv; import os", "mv(x)", "for", ""])
def test_plan_names_must_be_python_names(app, RE, namespace, name):
    with pytest.raises(ValueError):
        QtMovableControl("motor", RE, namespace=namespace, move_plan=name)


def test_explicit_enum_value_type_gives_a_dropdown(app, RE, namespace, commands):
    control = QtMovableControl("label", RE, namespace=namespace, value_type=_Mode)
    try:
        combo = _combo(control)
        assert [combo.itemText(i) for i in range(combo.count())] == ["Fast", "Slow"]
        combo.activated.emit(0)
        app.processEvents()
        assert commands == ["RE(mv(label, 'Fast'))"]
    finally:
        control.close()


async def _noop():
    pass


def _process_until(app, condition):
    for _ in range(200):
        app.processEvents()
        if condition():
            return True
        call_in_bluesky_event_loop(_noop())
    return condition()


def test_setpoint_starts_at_the_readback(app, RE, namespace):
    namespace["temperature"] = soft_signal_rw(float, 1.5, name="temperature")
    control = _control(namespace, RE, "temperature")
    try:
        setpoint = _edit(control)
        assert _process_until(app, lambda: setpoint.text() == "1.500")
    finally:
        control.close()


async def _set(signal, value):
    await signal.set(value)


def _readback_text(control: QtMovableControl) -> str:
    assert control._readback is not None
    value = control._readback._value
    assert isinstance(value, QLabel)
    return value.text()


def test_setpoint_does_not_follow_later_readbacks(app, RE, namespace):
    signal = soft_signal_rw(float, 1.5, name="temperature")
    namespace["temperature"] = signal
    control = _control(namespace, RE, "temperature")
    try:
        setpoint = _edit(control)
        assert _process_until(app, lambda: setpoint.text() == "1.500")
        call_in_bluesky_event_loop(_set(signal, 2.0))
        assert _process_until(app, lambda: "2.000" in _readback_text(control))
        assert setpoint.text() == "1.500"
    finally:
        control.close()


def test_dropdown_starts_at_the_readback(app, RE, namespace):
    namespace["mode"] = soft_signal_rw(_Mode, _Mode.SLOW, name="mode")
    control = _control(namespace, RE, "mode")
    try:
        combo = _combo(control)
        assert _process_until(app, lambda: combo.currentText() == "Slow")
    finally:
        control.close()


def test_slits_control_groups_blades_gaps_and_centers(app, RE, namespace):
    widget = QtSlitsControl("slits", RE, namespace=namespace)
    try:
        assert [[c.path for c in g.controls] for g in widget.groups] == [
            ["slits.inboard", "slits.outboard", "slits.bottom", "slits.top"],
            ["slits.horizontal_gap", "slits.vertical_gap"],
            ["slits.horizontal_center", "slits.vertical_center"],
        ]
        assert all(c._tweak is not None for g in widget.groups for c in g.controls)
        assert all(c._readback is not None for g in widget.groups for c in g.controls)
    finally:
        widget.close()


def _enter(control, text):
    setpoint = _edit(control)
    setpoint.setText(text)
    setpoint.returnPressed.emit()


def test_batch_mode_stages_setpoints_and_moves_them_together(
    app, RE, namespace, commands
):
    namespace["other"] = soft_signal_rw(float, name="other")
    group = QtMovableGroup(["motor", "other"], RE, namespace=namespace)
    try:
        group.set_batch(True)
        motor, other = group.controls
        _enter(motor, "1.5")
        _enter(other, "2")
        app.processEvents()
        assert commands == []
        assert group.staged == [("motor", 1.5), ("other", 2.0)]
        assert not _tweak(motor).isEnabled()

        group._on_move_clicked()
        app.processEvents()
        assert commands == ["RE(mv(motor, 1.5, other, 2.0))"]
        assert group.staged == []
    finally:
        group.close()


def _type_and_leave(control, text):
    """Type ``text`` into the setpoint, then move focus away without Enter."""
    setpoint = _edit(control)
    setpoint.setText(text)
    setpoint.setModified(True)
    setpoint.editingFinished.emit()


def test_batch_mode_stages_on_leaving_the_field(app, RE, namespace, commands):
    group = QtMovableGroup(["motor"], RE, namespace=namespace)
    try:
        group.set_batch(True)
        (motor,) = group.controls
        _type_and_leave(motor, "1.5")
        assert group.staged == [("motor", 1.5)]
        assert group._summary.text() == "(1 staged)"
        assert commands == []
    finally:
        group.close()


def test_leaving_the_field_does_not_move_outside_batch_mode(
    app, RE, namespace, commands
):
    group = QtMovableGroup(["motor"], RE, namespace=namespace)
    try:
        (motor,) = group.controls
        _type_and_leave(motor, "1.5")
        app.processEvents()
        assert commands == []
    finally:
        group.close()


def test_setpoint_changed_back_to_its_original_value_is_unstaged(
    app, RE, namespace, commands
):
    namespace["temperature"] = soft_signal_rw(float, 1.5, name="temperature")
    group = QtMovableGroup(["temperature", "mode"], RE, namespace=namespace)
    try:
        temperature, mode = group.controls
        setpoint = _edit(temperature)
        assert _process_until(app, lambda: setpoint.text() == "1.500")
        group.set_batch(True)

        _type_and_leave(temperature, "2")
        assert group.staged == [("temperature", 2.0)]
        _type_and_leave(temperature, "1.5")
        assert group.staged == []
        assert not setpoint.styleSheet()

        combo = _combo(mode)
        combo.setCurrentIndex(0)
        mode.capture_original()
        combo.activated.emit(1)
        assert group.staged == [("mode", _Mode.SLOW)]
        combo.activated.emit(0)
        assert group.staged == []
    finally:
        group.close()


def test_editing_a_staged_setpoint_unstages_it(app, RE, namespace, commands):
    group = QtMovableGroup(["motor"], RE, namespace=namespace)
    try:
        group.set_batch(True)
        (motor,) = group.controls
        _enter(motor, "1.5")
        _edit(motor).textEdited.emit("1.")
        assert group.staged == []
    finally:
        group.close()


def test_leaving_batch_mode_discards_staged_and_moves_directly(
    app, RE, namespace, commands
):
    group = QtMovableGroup(["motor"], RE, namespace=namespace)
    try:
        group.set_batch(True)
        (motor,) = group.controls
        _enter(motor, "1.5")
        group.set_batch(False)
        assert group.staged == []
        _enter(motor, "3")
        app.processEvents()
        assert commands == ["RE(mv(motor, 3.0))"]
    finally:
        group.close()


def test_tweaking_does_not_pass_focus_to_the_next_row(app, RE, namespace, commands):
    group = QtMovableGroup(["motor", "count"], RE, namespace=namespace)
    try:
        group.show()
        group.activateWindow()
        motor, count = group.controls
        right = motor._tweak_widgets[-1]
        assert isinstance(right, QPushButton)
        right.setFocus()
        app.processEvents()
        right.click()
        assert not _edit(count).hasFocus()
    finally:
        group.close()


def test_toggling_batch_mode_keeps_the_group_size(app, RE, namespace):
    group = QtMovableGroup(["motor"], RE, namespace=namespace)
    try:
        group.show()
        app.processEvents()
        before = group.sizeHint()
        group.set_batch(True)
        app.processEvents()
        assert group.sizeHint() == before
    finally:
        group.close()


def test_slits_control_rejects_non_slits(app, RE, namespace):
    with pytest.raises(ValueError):
        QtSlitsControl("motor", RE, namespace=namespace)


@pytest.mark.parametrize(
    "stacked, cells",
    [(False, [(0, 0), (0, 1), (1, 1)]), (True, [(0, 0), (1, 0), (2, 0)])],
)
def test_slits_control_layout(app, RE, namespace, stacked, cells):
    widget = QtSlitsControl("slits", RE, namespace=namespace, stacked=stacked)
    try:
        parent = widget.groups[0].parentWidget()
        assert parent is not None
        grid = parent.layout()
        assert isinstance(grid, QGridLayout)
        positions = [
            cast(tuple[int, int, int, int], grid.getItemPosition(grid.indexOf(g)))[:2]
            for g in widget.groups
        ]
        assert positions == cells
    finally:
        widget.close()
