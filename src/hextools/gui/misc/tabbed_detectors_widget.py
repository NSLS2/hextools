"""Tabbed live viewers for area detector NTNDArray streams."""

from typing import Any

import numpy as np
from ntnda_qt_viewer import NTNDAViewerWidget
from qtpy.QtCore import Qt, Signal, Slot
from qtpy.QtWidgets import QHBoxLayout, QTabWidget, QToolButton, QWidget

from hextools.gui._client_signals import QtCatalog
from hextools.gui.run_engine_clients import RunEngineClient


class _DetectorWindow(QWidget):
    """Top-level window holding popped-out viewers; tells its owner when closed."""

    closed = Signal()

    def __init__(self, content: QWidget, title: str):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(title)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(content)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class QtTabbedDetectorsWidget(QTabWidget):
    """Tabbed NTNDArray live viewers, one tab per available detector.

    Detectors are registered with :meth:`add_detector`, and side-by-side tabs
    with :meth:`add_combined`. Any tab can be popped out into its own window
    with the **Pop Out** button or :meth:`pop_out`; that window keeps showing
    the detectors whichever tab, here or elsewhere in the GUI, is shown.

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
        self._viewer_names: dict[NTNDAViewerWidget, str] = {}
        self._combined_tabs: dict[str, QWidget] = {}
        self._tab_viewers: dict[QWidget, list[NTNDAViewerWidget]] = {}
        self._tab_names: dict[QWidget, list[str]] = {}
        self._windows: dict[str, _DetectorWindow] = {}
        self._window_viewers: dict[str, list[NTNDAViewerWidget]] = {}
        self._running: set[NTNDAViewerWidget] = set()
        self._re_client = re_client

        self._pop_out_button = QToolButton()
        self._pop_out_button.setText("Pop Out")
        self._pop_out_button.setToolTip(
            "Show this view in its own window, which stays visible in any tab"
        )
        self._pop_out_button.clicked.connect(
            lambda *_: self.pop_out(self.tabText(self.currentIndex()))
        )
        self.setCornerWidget(self._pop_out_button, Qt.Corner.TopRightCorner)

        self.currentChanged.connect(self._sync_running)
        self._catalog = QtCatalog(re_client, self)
        self._catalog.changed.connect(lambda *_: self._refresh())
        self._signal_frame.connect(self._on_frame)
        self._update_pop_out_button()

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

    def pop_out(self, label: str) -> None:
        """Show tab ``label`` in its own window, or raise it if already shown.

        While popped out, the window's viewers run instead of the tabs' viewers
        of the same detectors. Any other window showing one of those detectors
        is closed, so each detector streams to one window at a time.

        Raises
        ------
        KeyError
            If no available tab is called ``label``.
        """
        window = self._windows.get(label)
        if window is not None:
            window.raise_()
            window.activateWindow()
            return
        tab = self._tab(label)
        names = self._tab_names[tab]
        for other, viewers in list(self._window_viewers.items()):
            if any(self._viewer_names[v] in names for v in viewers):
                self._windows[other].close()

        content, viewers = self._viewer_row(names)
        # No parent: a parented window is kept above the main window.
        window = _DetectorWindow(content, label)
        window.closed.connect(lambda: self._on_window_closed(label))
        window.resize(tab.size())
        self._windows[label] = window
        self._window_viewers[label] = viewers
        window.show()
        self._sync_running()

    def _tab(self, label: str) -> QWidget:
        for tab in self._tab_names:
            index = self.indexOf(tab)
            if self.tabText(index) == label and self.isTabVisible(index):
                return tab
        raise KeyError(label)

    def _on_window_closed(self, label: str) -> None:
        self._windows.pop(label)
        for viewer in self._window_viewers.pop(label):
            if viewer in self._running:
                viewer.stop()
                self._running.discard(viewer)
            self._viewers_by_name[self._viewer_names.pop(viewer)].remove(viewer)
        self._sync_running()

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
                self._tab_names[viewer] = [name]
                self.insertTab(self._tab_position(name), viewer, name)
            self.setTabVisible(self.indexOf(viewer), name in available)

        for label, names in self._combined.items():
            all_available = all(name in available for name in names)
            tab = self._combined_tabs.get(label)
            if tab is None:
                if not all_available:
                    continue
                # Separate viewer instances: a widget can only live in one tab.
                tab, viewers = self._viewer_row(names)
                self._combined_tabs[label] = tab
                self._tab_viewers[tab] = viewers
                self._tab_names[tab] = list(names)
                self.addTab(tab, label)
            self.setTabVisible(self.indexOf(tab), all_available)

        self._sync_running()

    def _viewer_row(self, names: list[str]) -> tuple[QWidget, list[NTNDAViewerWidget]]:
        """Return new viewers of ``names`` side by side, and the widget holding them."""
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        viewers = []
        for i, name in enumerate(names):
            # Put the rightmost viewer's vertical profile on its outer edge.
            kwargs = (
                {"v_profile_position": "right"} if i == len(names) - 1 and i > 0 else {}
            )
            viewer = self._make_viewer(name, **kwargs)
            viewers.append(viewer)
            row.addWidget(viewer, stretch=1)
        return widget, viewers

    def _update_pop_out_button(self):
        index = self.currentIndex()
        self._pop_out_button.setEnabled(index >= 0 and self.isTabVisible(index))

    @Slot()
    def _sync_running(self, *_):
        """Run the popped-out viewers, and the current tab's while shown.

        A tab's viewer of a popped-out detector doesn't run.
        """
        self._update_pop_out_button()
        popped_out = {v for viewers in self._window_viewers.values() for v in viewers}
        popped_names = {self._viewer_names[v] for v in popped_out}
        current = self.currentWidget() if self.isVisible() else None
        wanted = {
            viewer
            for viewer in (self._tab_viewers.get(current, []) if current else [])
            if self._viewer_names[viewer] not in popped_names
        } | popped_out
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
        self._viewer_names[viewer] = name
        return viewer

    def _tab_position(self, name: str) -> int:
        """Index that keeps tabs in the order the detectors were given."""
        order = list(self._detectors)
        return sum(1 for other in order[: order.index(name)] if other in self._viewers)

    def closeEvent(self, event):
        self._catalog.stop()
        for window in list(self._windows.values()):
            window.close()
        for viewer in self._running:
            viewer.stop()
        self._running.clear()
        super().closeEvent(event)
