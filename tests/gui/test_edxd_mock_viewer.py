import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
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


def test_mock_germ_frames_reach_the_viewer_through_the_client(RE, monkeypatch):
    from bluesky.run_engine import call_in_bluesky_event_loop
    from ophyd_async.core import init_devices, set_mock_value

    from hextools.devices.detectors.germ import GeRMDetector
    from hextools.gui.__main__ import GERM_IMAGE_SHAPE, QtEDXDView
    from hextools.gui.run_engine_clients import InProcessRunEngineClient
    from hextools.plans import edxd
    from hextools.utils.path_providers import mock_path_provider

    app = QApplication.instance() or QApplication([])
    shown: list[tuple[str, np.ndarray]] = []
    monkeypatch.setattr(
        QtTabbedDetectorsWidget,
        "show_frame",
        lambda self, name, frame: shown.append((name, frame)),
    )
    with init_devices(mock=True):
        germ = GeRMDetector("GERM:", mock_path_provider("/tmp/germ"))
    plans = {name: getattr(edxd, name) for name in edxd.__all__}
    view = QtEDXDView(
        InProcessRunEngineClient({"RE": RE, "germ": germ, **plans}, mock=True)
    )
    try:
        frame = np.arange(np.prod(GERM_IMAGE_SHAPE), dtype=np.int32)

        async def publish():
            set_mock_value(germ.driver.mca, frame)

        call_in_bluesky_event_loop(publish())
        # The mock's zeros, sent on subscribing, then the published frame.
        assert [name for name, _ in shown] == ["germ", "germ"]
        assert not shown[0][1].any()
        np.testing.assert_array_equal(shown[1][1], frame.reshape(GERM_IMAGE_SHAPE))

        view.close()
        call_in_bluesky_event_loop(publish())
        assert len(shown) == 2
    finally:
        view.close()
        view.deleteLater()
        app.processEvents()
