import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

from bluesky import RunEngine  # noqa: E402
from IPython.core.interactiveshell import InteractiveShell  # noqa: E402
from ophyd_async.core import init_devices, set_mock_value  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from hextools.gui.plan_widget import QtPlanWidget  # noqa: E402
from hextools.gui.slit_control import QtFrontEndSlitStatus  # noqa: E402
from hextools.photon_delivery_system.fe_slit import (  # noqa: E402
    FrontEndSlit,
    SlitAperture,
    set_fe_slit1_aperture,
)


def _wait_for(app, condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)
    return condition()


def test_slit_control_form_is_one_aperture_dropdown():
    app = QApplication.instance() or QApplication([])
    InteractiveShell.instance()
    widget = QtPlanWidget(
        RunEngine({}, call_returns_result=True), set_fe_slit1_aperture
    )
    try:
        combos = {name: combo for name, combo, _ in widget._enum_fields}
        assert list(combos) == ["aperture"]
        options = [
            combos["aperture"].itemText(i) for i in range(combos["aperture"].count())
        ]
        assert [o for o in options if o in {a.value for a in SlitAperture}] == [
            a.value for a in SlitAperture
        ]
        assert widget._device_fields == []
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_slit_readout_follows_y(RE: RunEngine):
    app = QApplication.instance() or QApplication([])
    with init_devices(mock=True):
        fe_slit1 = FrontEndSlit("FE:TEST{Slt:1-Ax:Y}", name="fe-slit1")
    widget = QtFrontEndSlitStatus(RE, {"fe_slit1": fe_slit1, "_hidden": fe_slit1})
    try:
        (readout,) = widget.readouts
        RE.loop.call_soon_threadsafe(set_mock_value, fe_slit1.y.user_readback, 30.0)
        assert _wait_for(
            app, lambda: readout.aperture_label.text() == "Pinhole 0.89 mm"
        )
        assert readout.y_label.text() == "30.000"
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
