"""Tabbed live viewers for area detector NTNDArray streams."""

from typing import Any

import numpy as np
from ntnda_qt_viewer import NTNDAViewerWidget
from qtpy.QtCore import Signal, Slot
from qtpy.QtWidgets import QHBoxLayout, QTabWidget, QWidget

from hextools.gui._client_signals import QtCatalog
from hextools.gui.run_engine_clients import RunEngineClient


class QtTabbedDetectorsWidget(QTabWidget):
    """Tabbed NTNDArray live viewers, one tab per available detector.

    Detectors are registered with :meth:`add_detector`, and side-by-side tabs
    with :meth:`add_combined`.

    Parameters
    ----------
    re_client : RunEngineClient
        Whose device registry says which detectors are available.
    """

    _signal_frame = Signal(str, object)

    def __init__(self, re_client: RunEngineClient, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._detectors: dict[str, dict[str, Any]] = {}
        self._combined: dict[str, list[str]] = {}
        self._viewers: dict[str, NTNDAViewerWidget] = {}
        self._viewers_by_name: dict[str, list[NTNDAViewerWidget]] = {}
        self._combined_tabs: dict[str, QWidget] = {}
        self._tab_viewers: dict[QWidget, list[NTNDAViewerWidget]] = {}
        self._running: set[NTNDAViewerWidget] = set()
        self._re_client = re_client

        self.currentChanged.connect(self._sync_running)
        self._catalog = QtCatalog(re_client, self)
        self._catalog.changed.connect(lambda *_: self._refresh())
        self._signal_frame.connect(self._on_frame)

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

    def show_frame(self, name: str, frame: np.ndarray) -> None:
        """Display ``frame`` in detector ``name``'s viewers, as if read from its PV.

        Safe to call from any thread. Shown only while the viewers are running.
        """
        self._signal_frame.emit(name, frame)

    @Slot(str, object)
    def _on_frame(self, name: str, frame: np.ndarray) -> None:
        for viewer in self._viewers_by_name.get(name, []):
            viewer.push_frame(frame)

    @Slot()
    def _refresh(self):
        registry = self._re_client.device_registry
        available = [name for name in self._detectors if name in registry]

        for name in self._detectors:
            viewer = self._viewers.get(name)
            if viewer is None:
                if name not in available:
                    continue
                # Created lazily so unavailable detectors open no PV connections.
                viewer = self._make_viewer(name)
                self._viewers[name] = viewer
                self._tab_viewers[viewer] = [viewer]
                self.insertTab(self._tab_position(name), viewer, name)
            self.setTabVisible(self.indexOf(viewer), name in available)

        for label, names in self._combined.items():
            all_available = all(name in available for name in names)
            tab = self._combined_tabs.get(label)
            if tab is None:
                if not all_available:
                    continue
                # Separate viewer instances: a widget can only live in one tab.
                tab = QWidget()
                row = QHBoxLayout(tab)
                row.setContentsMargins(0, 0, 0, 0)
                viewers = []
                for i, name in enumerate(names):
                    # Put the rightmost viewer's vertical profile on its outer edge.
                    kwargs = (
                        {"v_profile_position": "right"}
                        if i == len(names) - 1 and i > 0
                        else {}
                    )
                    viewer = self._make_viewer(name, **kwargs)
                    viewers.append(viewer)
                    row.addWidget(viewer, stretch=1)
                self._combined_tabs[label] = tab
                self._tab_viewers[tab] = viewers
                self.addTab(tab, label)
            self.setTabVisible(self.indexOf(tab), all_available)

        self._sync_running()

    @Slot()
    def _sync_running(self, *_):
        """Run only the viewers in the current tab, and only while shown."""
        current = self.currentWidget() if self.isVisible() else None
        wanted = set(self._tab_viewers.get(current, [])) if current else set()
        # Stop before starting so the same PV is never subscribed twice at once.
        for viewer in self._running - wanted:
            try:
                viewer.stop()
            except RuntimeError:
                # Hidden during teardown, after the viewer's Qt children were deleted.
                pass
        for viewer in wanted - self._running:
            viewer.start()
        self._running = wanted

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_running()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._sync_running()

    def _make_viewer(self, name: str, **overrides: Any) -> NTNDAViewerWidget:
        viewer = NTNDAViewerWidget(**{**self._detectors[name], **overrides})
        self._viewers_by_name.setdefault(name, []).append(viewer)
        return viewer

    def _tab_position(self, name: str) -> int:
        """Index that keeps tabs in the order the detectors were given."""
        order = list(self._detectors)
        return sum(1 for other in order[: order.index(name)] if other in self._viewers)

    def closeEvent(self, event):
        self._catalog.stop()
        for viewer in self._running:
            viewer.stop()
        self._running.clear()
        super().closeEvent(event)
