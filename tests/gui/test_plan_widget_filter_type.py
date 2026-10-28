import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

from qtpy.QtWidgets import QApplication  # noqa: E402

from hextools.gui.plan_widget import QtPlanWidget  # noqa: E402
from hextools.tomography.flyscans import (  # noqa: E402
    tomo_1d_step_scan,
    tomo_2d_step_scan,
    tomo_flyscan,
    tomo_flyscan_loop,
)
from hextools.tomography.radiography import take_radiograph  # noqa: E402


@pytest.mark.parametrize(
    "plan",
    [
        tomo_flyscan,
        tomo_flyscan_loop,
        tomo_1d_step_scan,
        tomo_2d_step_scan,
        take_radiograph,
    ],
)
def test_plan_form_offers_the_proc_filter_type_defaulting_to_average(
    plan, session_client
):
    app = QApplication.instance() or QApplication([])
    widget = QtPlanWidget(session_client, plan)
    try:
        combos = {name: combo for name, combo, _ in widget._enum_fields}
        combo = combos["filter_type"]
        options = [combo.itemText(i) for i in range(combo.count())]
        assert "Sum" in options
        assert combo.currentText() == "Average"

        combo.setCurrentIndex(combo.findText("Sum"))
        assert combo.currentData() == "Sum"
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
