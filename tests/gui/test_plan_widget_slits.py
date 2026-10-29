import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

from bluesky import RunEngine  # noqa: E402
from IPython.core.interactiveshell import InteractiveShell  # noqa: E402
from ophyd_async.core import init_devices, set_mock_value  # noqa: E402
from qtpy.QtWidgets import QApplication, QCheckBox, QLineEdit  # noqa: E402

from hextools.gui.plan_widget import QtPlanWidget  # noqa: E402
from hextools.photon_delivery_system import Slits, move_slits  # noqa: E402


def _wait(app, predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert predicate()


def test_selected_slits_load_actual_values():
    app = QApplication.instance() or QApplication([])
    shell = InteractiveShell.instance()
    re = RunEngine({})
    with init_devices(mock=True):
        slits = Slits("S:", name="operator_slits")
    for motor, value in zip(
        (slits.inboard, slits.outboard, slits.bottom, slits.top),
        (1.0, 5.0, -4.0, 2.0),
        strict=True,
    ):
        set_mock_value(motor.user_readback, value)
    shell.user_ns["operator_slits"] = slits
    widget = QtPlanWidget(re, move_slits)
    try:
        combo = widget._device_fields[0]["widget"]
        combo.setCurrentIndex(combo.findData("operator_slits"))
        fields = {
            n: w for n, _, w, _ in widget._scalar_fields if isinstance(w, QLineEdit)
        }
        _wait(app, lambda: fields["horizontal_gap"].text() != "")
        assert float(fields["horizontal_gap"].text()) == 4.0
        assert float(fields["horizontal_center"].text()) == 3.0
        assert float(fields["vertical_gap"].text()) == 6.0
        assert float(fields["vertical_center"].text()) == -1.0
    finally:
        widget.close()
        shell.user_ns.pop("operator_slits")
        app.processEvents()


@pytest.mark.parametrize("old_error", [False, True])
def test_slit_reads_are_nonblocking_and_ignore_obsolete_responses(
    monkeypatch, old_error
):
    from concurrent.futures import Future

    app = QApplication.instance() or QApplication([])
    shell = InteractiveShell.instance()
    re = RunEngine({})
    with init_devices(mock=True):
        first = Slits("A:", name="operator_first")
        second = Slits("B:", name="operator_second")
    shell.user_ns.update(operator_first=first, operator_second=second)
    requests = []

    def defer(coro, loop):
        coro.close()
        future = Future()
        future.set_running_or_notify_cancel()
        requests.append(future)
        return future

    monkeypatch.setattr(
        "hextools.gui.plan_widget.asyncio.run_coroutine_threadsafe", defer
    )
    widget = QtPlanWidget(re, move_slits)
    combo = widget._device_fields[0]["widget"]
    fields = {n: w for n, _, w, _ in widget._scalar_fields if isinstance(w, QLineEdit)}
    try:
        combo.setCurrentIndex(combo.findData("operator_first"))
        assert not widget._submit_button.isEnabled()
        assert all(not w.isEnabled() and not w.text() for w in fields.values())
        app.processEvents()  # Selection must return without waiting for EPICS.
        combo.setCurrentIndex(combo.findData("operator_second"))
        requests[1].set_result(dict.fromkeys(fields, 2.0))
        app.processEvents()
        assert all(float(w.text()) == 2.0 for w in fields.values())
        if old_error:
            requests[0].set_exception(RuntimeError("obsolete read"))
        else:
            requests[0].set_result(dict.fromkeys(fields, 1.0))
        app.processEvents()
        assert all(float(w.text()) == 2.0 for w in fields.values())
        assert widget._submit_button.isEnabled()
        combo.setCurrentIndex(combo.findData("operator_first"))
        requests[2].set_exception(RuntimeError("read timed out"))
        app.processEvents()
        assert "read timed out" in widget._status_label.text()
        assert not widget._submit_button.isEnabled()
        assert all(not w.text() for w in fields.values())
        relative = next(w for n, _, w, _ in widget._scalar_fields if n == "relative")
        assert isinstance(relative, QCheckBox)
        relative.setChecked(True)
        assert all(float(w.text()) == 0.0 for w in fields.values())
        assert widget._submit_button.isEnabled()
        relative.setChecked(False)
        widget.close()
        requests[3].set_result(dict.fromkeys(fields, 99.0))
        app.processEvents()
        assert all(not w.text() for w in fields.values())
    finally:
        widget.close()
        shell.user_ns.pop("operator_first")
        shell.user_ns.pop("operator_second")
        app.processEvents()


def test_slits_form_offers_device_optional_centers_gaps_and_relative():
    app = QApplication.instance() or QApplication([])
    InteractiveShell.instance()
    widget = QtPlanWidget(RunEngine({}, call_returns_result=True), move_slits)
    try:
        scalars = {
            name: (kind, required) for name, kind, _, required in widget._scalar_fields
        }
        for name in (
            "horizontal_center",
            "vertical_center",
            "horizontal_gap",
            "vertical_gap",
        ):
            assert scalars[name] == (float, False)
        assert scalars["relative"] == (bool, False)
        devices = {field["name"]: field for field in widget._device_fields}
        assert (
            devices["slits"]["types"] == (Slits,) or Slits in devices["slits"]["types"]
        )
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
