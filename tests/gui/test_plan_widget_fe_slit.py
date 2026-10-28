import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

from bluesky import RunEngine  # noqa: E402
from IPython.core.interactiveshell import InteractiveShell  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from hextools.gui.plan_widget import QtPlanWidget  # noqa: E402
from hextools.photon_delivery_system.fe_slit import (  # noqa: E402
    SlitAperture,
    set_fe_slit1_aperture,
)


def test_fe_slit1_form_is_one_aperture_dropdown_and_no_device_picker():
    app = QApplication.instance() or QApplication([])
    InteractiveShell.instance()
    widget = QtPlanWidget(
        RunEngine({}, call_returns_result=True), set_fe_slit1_aperture
    )
    try:
        combos = {name: combo for name, combo, _ in widget._enum_fields}
        assert list(combos) == ["aperture"]
        combo = combos["aperture"]
        # A required choice gets no blank entry, so the form starts on the first.
        options = [combo.itemText(i) for i in range(combo.count())]
        assert options == [a.value for a in SlitAperture]
        assert widget._device_fields == []

        combo.setCurrentIndex(combo.findText("Pinhole 0.89 mm"))
        assert combo.currentData() == "Pinhole 0.89 mm"
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
