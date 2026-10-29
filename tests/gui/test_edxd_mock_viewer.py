import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")
ntnda = pytest.importorskip("ntnda_qt_viewer")

from qtpy.QtWidgets import QApplication  # noqa: E402

from hextools.gui.__main__ import germ_viewer_options  # noqa: E402
from hextools.gui.misc.tabbed_detectors_widget import (  # noqa: E402
    QtTabbedDetectorsWidget,
)


# Copilot on #109: in --mock the GeRM viewer still subscribed to the real GeRM PV,
# so on a beamline workstation real frames mixed with the mock's.
@pytest.mark.parametrize("mock", [True, False])
def test_only_the_mock_germ_viewer_stays_off_the_pv(mock: bool):
    app = QApplication.instance() or QApplication([])
    viewer = ntnda.NTNDAViewerWidget(
        "XF:27ID1-ES{GeRM-Det:1}MCA", **germ_viewer_options(mock=mock)
    )
    try:
        assert viewer._use_pv is not mock
    finally:
        viewer.close()
        viewer.deleteLater()
        app.processEvents()


# Copilot on #109: frames went to the viewer's private _on_new_frame, which
# ignores Stop; push_frame drops frames while the viewer is stopped.
def test_frames_reach_viewers_through_push_frame():
    pushed: list[np.ndarray] = []
    viewer = SimpleNamespace(push_frame=pushed.append)
    frame = np.zeros((2, 3))

    QtTabbedDetectorsWidget._on_frame(
        SimpleNamespace(_viewers_by_name={"germ": [viewer]}),  # type: ignore
        "germ",
        frame,
    )

    assert pushed == [frame]
