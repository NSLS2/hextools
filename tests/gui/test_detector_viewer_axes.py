import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
ntnda = pytest.importorskip("ntnda_qt_viewer")

import numpy as np  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402


# The Kinetix live view read 3200 -> 0 left to right while Phoebus read 0 -> 3200
# (2026-10-22, again 2026-10-27): the viewer labelled a flipped copy of the frame.
def test_detector_viewer_axis_labels_match_source_pixels():
    app = QApplication.instance() or QApplication([])
    rows, cols = 40, 320
    frame = np.tile(np.arange(cols, dtype=np.uint16), (rows, 1))
    viewer = ntnda.NTNDAViewerWidget("NOPV:", show_profile_lines=True)
    try:
        viewer.resize(800, 300)
        viewer.show()
        viewer._on_new_frame(frame)
        viewer._refresh_display()
        view = viewer._image_vb
        view.autoRange(padding=0)
        app.processEvents()

        shown = viewer._image_item.image
        (x_lo, x_hi), _ = view.viewRange()
        left, right = (x_hi, x_lo) if view.xInverted() else (x_lo, x_hi)

        assert left < right, "x axis labels decrease left to right"
        assert shown[0, 0] == 0 and shown[0, -1] == cols - 1
    finally:
        viewer.close()
        viewer.deleteLater()
        app.processEvents()
