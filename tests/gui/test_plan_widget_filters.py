import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

from bluesky import RunEngine  # noqa: E402
from IPython.core.interactiveshell import InteractiveShell  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from hextools.gui.plan_widget import QtPlanWidget  # noqa: E402
from hextools.photon_delivery_system.filters import (  # noqa: E402
    Filter1DownstreamPosition,
    Filter1UpstreamPosition,
    Filter2Position,
    set_filters,
)


def test_filters_form_has_one_dropdown_per_filter_and_no_device_pickers():
    app = QApplication.instance() or QApplication([])
    InteractiveShell.instance()
    widget = QtPlanWidget(RunEngine({}, call_returns_result=True), set_filters)
    try:
        combos = {name: combo for name, combo, _ in widget._enum_fields}
        expected = {
            "filter1_upstream_setting": Filter1UpstreamPosition,
            "filter1_downstream_setting": Filter1DownstreamPosition,
            "filter2_setting": Filter2Position,
        }
        assert set(combos) == set(expected)
        for name, combo in combos.items():
            options = [combo.itemText(i) for i in range(1, combo.count())]
            assert options == [p.value for p in expected[name]]
        assert widget._device_fields == []

        combo = combos["filter2_setting"]
        combo.setCurrentIndex(combo.findText("1 mm Cu"))
        assert combo.currentData() == "1 mm Cu"
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
