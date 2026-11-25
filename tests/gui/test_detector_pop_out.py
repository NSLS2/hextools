import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("ntnda_qt_viewer")

from ophyd_async.sim import SimMotor  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from hextools.gui.misc.tabbed_detectors_widget import (  # noqa: E402
    QtTabbedDetectorsWidget,
)
from hextools.gui.run_engine_clients import InProcessRunEngineClient  # noqa: E402


def _widget(RE) -> QtTabbedDetectorsWidget:
    client = InProcessRunEngineClient(
        {"RE": RE, "cam": SimMotor(name="cam"), "cam2": SimMotor(name="cam2")}
    )
    widget = QtTabbedDetectorsWidget(client)
    # Callback mode: the viewers never touch a PV.
    for name in ("cam", "cam2"):
        widget.add_detector(name, f"{name}:", push_frame=lambda push: None)
    widget.add_combined("Both", ["cam", "cam2"])
    return widget


def test_a_popped_out_view_runs_in_any_tab_instead_of_its_tab_view(RE, monkeypatch):
    app = QApplication.instance() or QApplication([])
    widget = _widget(RE)
    widget.show()
    try:
        tab_viewer = widget._viewers["cam"]
        assert widget._running == {tab_viewer}

        widget.pop_out("cam")
        (window_viewer,) = widget._window_viewers["cam"]
        assert widget._running == {window_viewer}

        # As when another technique tab is shown.
        widget.hide()
        assert widget._running == {window_viewer}
        pushed: list[np.ndarray] = []
        monkeypatch.setattr(window_viewer, "push_frame", pushed.append)
        frame = np.zeros((2, 3))
        widget._on_frame("cam", frame)
        assert pushed == [frame]

        widget.show()
        widget._windows["cam"].close()
        app.processEvents()
        assert widget._running == {tab_viewer}
        assert window_viewer not in widget._viewers_by_name["cam"]
    finally:
        widget.close()
        app.processEvents()


def test_each_detector_streams_to_one_window_at_a_time(RE):
    app = QApplication.instance() or QApplication([])
    widget = _widget(RE)
    widget.show()
    try:
        widget.pop_out("cam")
        widget.pop_out("Both")
        app.processEvents()
        assert list(widget._windows) == ["Both"]
        names = {widget._viewer_names[v] for v in widget._running}
        assert sorted(names) == ["cam", "cam2"]
        assert len(widget._running) == 2
    finally:
        widget.close()
        app.processEvents()
    assert widget._windows == {}
