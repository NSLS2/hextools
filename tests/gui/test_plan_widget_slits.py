import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

from qtpy.QtWidgets import QApplication  # noqa: E402

from hextools.gui.plan_widget import QtPlanWidget  # noqa: E402
from hextools.photon_delivery_system import Slits, move_slits  # noqa: E402


def test_slits_form_offers_device_optional_centers_gaps_and_relative(session_client):
    app = QApplication.instance() or QApplication([])
    widget = QtPlanWidget(session_client, move_slits)
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
