from typing import Any

from bluesky_widgets.models.run_engine_client import RunEngineClient
from IPython.core.getipython import get_ipython
from ntnda_qt_viewer import NTNDAViewerWidget
from qtpy.QtCore import Signal, Slot
from qtpy.QtWidgets import QHBoxLayout, QTabWidget, QWidget

from hextools.gui.device_sources import (
    DeviceSource,
    NamespaceDeviceSource,
    QueueServerDeviceSource,
)


class QtTabbedDetectorsWidget(QTabWidget):
    """Tabbed NTNDArray live viewers, one tab per available detector.

    Detectors are registered with :meth:`add_detector`, and side-by-side tabs
    with :meth:`add_combined`.

    Parameters
    ----------
    re_client : RunEngineClient or RunEngine
        Queue Server client, or the in-process RunEngine. Determines where
        device availability is checked.
    """

    _signal_devices_changed = Signal()

    def __init__(self, re_client, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._detectors: dict[str, dict[str, Any]] = {}
        self._combined: dict[str, list[str]] = {}
        self._viewers: dict[str, NTNDAViewerWidget] = {}
        self._combined_tabs: dict[str, QWidget] = {}
        self._source = self._make_source(re_client)

        self._signal_devices_changed.connect(self._refresh)
        self._source.subscribe(self._on_devices_changed)

    def add_detector(
        self,
        name: str,
        prefix: str,
        *,
        show_roi_controls: bool = False,
        **viewer_kwargs: Any,
    ) -> None:
        """Register a detector, shown in its own tab while it is available.

        Parameters
        ----------
        name : str
            Device name; the tab is only shown while a device with this name
            is available. Also used as the tab label.
        prefix : str
            PV prefix passed to :class:`NTNDAViewerWidget` (the full array PV
            when ``raw_waveform=True``).
        show_roi_controls : bool, optional
            Whether to show the viewer's ROI controls.
        **viewer_kwargs
            Any other keyword arguments accepted by :class:`NTNDAViewerWidget`.
        """
        if name in self._detectors:
            raise ValueError(f"Detector {name!r} already added")
        self._detectors[name] = {
            "prefix": prefix,
            "show_roi_controls": show_roi_controls,
            **viewer_kwargs,
        }
        self._refresh()

    def add_combined(self, label: str, names: list[str]) -> None:
        """Add a tab showing already-added detectors side by side.

        Shown after the individual tabs, only while all of its detectors are
        available.
        """
        unknown = [name for name in names if name not in self._detectors]
        if unknown:
            raise ValueError(f"Unknown detectors {unknown}; add them first")
        self._combined[label] = list(names)
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
        available = {name: self._source.is_valid_device(name) for name in self._detectors}
        for name in self._detectors:
            viewer = self._viewers.get(name)
            if viewer is None:
                if not available[name]:
                    continue
                # Created lazily so unavailable detectors open no PV connections.
                viewer = self._make_viewer(name)
                self._viewers[name] = viewer
                self.insertTab(self._tab_position(name), viewer, name)
            self.setTabVisible(self.indexOf(viewer), available[name])

        for label, names in self._combined.items():
            all_available = all(available.get(name, False) for name in names)
            tab = self._combined_tabs.get(label)
            if tab is None:
                if not all_available:
                    continue
                # Separate viewer instances: a widget can only live in one tab.
                tab = QWidget()
                row = QHBoxLayout(tab)
                row.setContentsMargins(0, 0, 0, 0)
                for name in names:
                    row.addWidget(self._make_viewer(name), stretch=1)
                self._combined_tabs[label] = tab
                self.addTab(tab, label)
            self.setTabVisible(self.indexOf(tab), all_available)

    def _make_viewer(self, name: str) -> NTNDAViewerWidget:
        return NTNDAViewerWidget(**self._detectors[name])

    def _tab_position(self, name: str) -> int:
        """Index that keeps tabs in the order the detectors were given."""
        order = list(self._detectors)
        return sum(1 for other in order[: order.index(name)] if other in self._viewers)

    def closeEvent(self, event):
        self._source.unsubscribe(self._on_devices_changed)
        super().closeEvent(event)
