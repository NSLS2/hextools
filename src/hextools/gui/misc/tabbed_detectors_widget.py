from bluesky_widgets.models.run_engine_client import RunEngineClient
from IPython.core.getipython import get_ipython
from ntnda_qt_viewer import NTNDAViewerWidget
from qtpy.QtCore import Signal, Slot
from qtpy.QtWidgets import QTabWidget

from hextools.gui.device_sources import (
    DeviceSource,
    NamespaceDeviceSource,
    QueueServerDeviceSource,
)


class QtTabbedDetectorsWidget(QTabWidget):
    """Tabbed NTNDArray live viewers, one tab per available detector.

    Parameters
    ----------
    re_client : RunEngineClient or RunEngine
        Queue Server client, or the in-process RunEngine. Determines where
        device availability is checked.
    detectors : dict[str, str]
        Device name -> NTNDArray PV prefix. A detector's tab is only shown
        while a device with that name is available.
    show_rois : bool, optional
        Whether to show the viewers' ROI controls.
    show_profile_lines : bool, optional
        Whether to show the viewers' crosshair profile lines.
    """

    _signal_devices_changed = Signal()

    def __init__(
        self,
        re_client,
        detectors: dict[str, str],
        *args,
        show_rois: bool = False,
        show_profile_lines: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._detectors = dict(detectors)
        self._show_rois = show_rois
        self._show_profile_lines = show_profile_lines
        self._viewers: dict[str, NTNDAViewerWidget] = {}
        self._source = self._make_source(re_client)

        self._signal_devices_changed.connect(self._refresh)
        self._source.subscribe(self._on_devices_changed)
        self._refresh()

    @staticmethod
    def _make_source(re_client) -> DeviceSource:
        if isinstance(re_client, RunEngineClient):
            return QueueServerDeviceSource(re_client)
        ipython = get_ipython()
        return NamespaceDeviceSource(ipython.user_ns if ipython is not None else {})

    def _on_devices_changed(self, event=None):
        # May fire from the Queue Server polling thread; marshal onto the GUI thread.
        self._signal_devices_changed.emit()

    @Slot()
    def _refresh(self):
        for name, prefix in self._detectors.items():
            available = self._source.is_valid_device(name)
            viewer = self._viewers.get(name)
            if viewer is None:
                if not available:
                    continue
                # Created lazily so unavailable detectors open no PV connections.
                viewer = NTNDAViewerWidget(prefix=prefix)
                viewer._roi_controls_widget.setVisible(self._show_rois)  # noqa: SLF001
                # The viewer's flag defaults off but its lines start visible; sync them.
                viewer._on_profile_lines_toggled(self._show_profile_lines)  # noqa: SLF001
                self._viewers[name] = viewer
                self.insertTab(self._tab_position(name), viewer, name)
            self.setTabVisible(self.indexOf(viewer), available)

    def _tab_position(self, name: str) -> int:
        """Index that keeps tabs in the order the detectors were given."""
        order = list(self._detectors)
        return sum(1 for other in order[: order.index(name)] if other in self._viewers)

    def closeEvent(self, event):
        self._source.unsubscribe(self._on_devices_changed)
        super().closeEvent(event)
