"""HEX data acquisition GUI.

Run with::

    python -m hextools.gui

This starts IPython on a profile (``--profile``, default ``collection``) with the
GUI attached, running plans in-process via ``RE(plan(...))``.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import os
import time as ttime
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any

import IPython
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from qtpy import shiboken
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from tiled.structures.core import Spec

from hextools.detectors.germ import GeRMDetectorMock
from hextools.gui._client_signals import QtRunEngineState
from hextools.gui._theme import apply_bnl_theme, saved_theme
from hextools.gui.available_devices import QtAvailableDevices
from hextools.gui.device_progress_viewer import QtReWaitingHookMonitor
from hextools.gui.live_re_md_viewer import QtProposalInfo
from hextools.gui.misc import QtTabbedDetectorsWidget, QtWeatherWidget
from hextools.gui.plan_status import (
    QtPlanExecutionView,
    QtPlanHistory,
    QtPlanLogView,
    QtPlanStatus,
)
from hextools.gui.plan_widget import QtPlanWidget
from hextools.gui.re_execution_controls import QtReExecutionControls
from hextools.gui.run_engine_clients import (
    InProcessRunEngineClient,
    RunEngineClient,
)
from hextools.gui.scan_browser import QtScanBrowser
from hextools.gui.shutter_status import QtShutterStatus
from hextools.gui.signal_status_group import QtBeamlineStatus
from hextools.gui.theme_switch import QtThemeAction
from hextools.photon_delivery_system import Shutter
from hextools.specs import ALL_SPECS
from hextools.utils import is_running_in_ci

from ._event_loop import get_our_app_name
from ._threading import wait_for_workers_to_quit

PLAN_HISTORY_FILE_ENV = "HEXTOOLS_PLAN_HISTORY_FILE"


def default_plan_history_file() -> Path:
    """``$XDG_STATE_HOME/hextools/plan_history[_mock].json``."""
    state_home = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    name = "plan_history_mock.json" if is_running_in_ci() else "plan_history.json"
    return Path(state_home) / "hextools" / name


def _plan_tabs(re_client: RunEngineClient, plans: dict[str, str]) -> QTabWidget:
    """Tabs of :class:`QtPlanWidget`, by tab label to plan name."""
    tabs = QTabWidget()
    for label, plan in plans.items():
        tabs.addTab(QtPlanWidget(re_client, plan), label)
    return tabs


class QtTomographyView(QWidget):
    """Tomography controls arranged around the live detector viewers.

    Detector viewers fill the majority of the tab and the tomography plan
    selector sits in a side column. The shared RE controls and progress bars
    live outside the tab in :class:`QtTabbedTechniqueSelector`.
    """

    def __init__(self, re_client: RunEngineClient, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._re_client = re_client

        vbox = QVBoxLayout()

        top = QHBoxLayout()
        # Detector viewers take the majority of the screen.
        detectors = QtTabbedDetectorsWidget(re_client)
        for i in range(1, 5):
            detectors.add_detector(
                f"kinetix{i}",
                "XF:27ID1-BI{Kinetix-Det:" + str(i) + "}",
                show_profile_lines=True,
                preserve_aspect_ratio=True,
            )
        detectors.add_combined("Dual Cam", ["kinetix1", "kinetix3"])
        detectors.add_detector(
            "phantom",
            "XF:27ID1-ES{Phantom-Det:1}",
            show_profile_lines=True,
            preserve_aspect_ratio=True,
        )
        top.addWidget(detectors, stretch=3)

        # Side column: a tabbed selector offering the tomography plans.
        plan_tabs = _plan_tabs(
            re_client,
            {
                "Dark/Flat": "tomo_dark_flat",
                "Flyscan": "tomo_flyscan",
                "Flyscan Loop": "tomo_flyscan_loop",
                "1D Step": "tomo_1d_step_scan",
                "2D Step": "tomo_2d_step_scan",
                "Radiography": "take_radiograph",
                "Alignment": "tomo_alignment_scan",
            },
        )

        side = QVBoxLayout()
        side.addWidget(plan_tabs, stretch=1)
        top.addLayout(side, stretch=1)

        vbox.addLayout(top, stretch=1)

        self.setLayout(vbox)


def germ_viewer_options(mock: bool) -> dict[str, Any]:
    """Options for the live GeRM viewer.

    A mock viewer is in callback mode, so it never subscribes to the real GeRM
    PV; the mock's frames reach it through ``show_frame``.
    """
    options: dict[str, Any] = {
        "raw_waveform": True,
        "image_shape": (192, 4096),
        "colormap": "JET",
        "roi_suffix_pattern": "1:ROI{}:",
        "auto_scale_method": "minmax",
        "log_scale": True,
        "use_ca": True,
        "show_profile_lines": True,
    }
    if mock:
        options["push_frame"] = lambda push: None
    return options


class QtEDXDView(QWidget):
    """Energy-dispersive X-ray diffraction view: the live GeRM viewer and EDXD plans."""

    def __init__(
        self,
        re_client: RunEngineClient,
        namespace: Mapping[str, Any],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._re_client = re_client
        self._namespace = namespace

        top = QHBoxLayout()
        detectors = QtTabbedDetectorsWidget(re_client)
        mock = is_running_in_ci()
        detectors.add_detector(
            "germ", "XF:27ID1-ES{GeRM-Det:1}MCA", **germ_viewer_options(mock=mock)
        )
        top.addWidget(detectors, stretch=3)

        plan_tabs = _plan_tabs(
            re_client,
            {
                "2Theta Tilt": "edxd_2theta_tilt",
                "Calibration": "edxd_calib_scan",
                "Scan": "edxd_scan",
                "Grid Scan": "edxd_grid_scan",
                "Custom Pos List Grid": "edxd_custom_pos_list_grid",
                "Count": "edxd_count",
                "Configure Test Pulses": "configure_test_pulses",
            },
        )
        side = QVBoxLayout()
        side.addWidget(plan_tabs, stretch=1)
        top.addLayout(side, stretch=1)

        vbox = QVBoxLayout()
        vbox.addLayout(top, stretch=1)
        self.setLayout(vbox)

        self._germ_mock: GeRMDetectorMock | None = None
        if mock:
            self._attach_germ_mock(detectors)

    def _attach_germ_mock(self, detectors: QtTabbedDetectorsWidget) -> None:
        """Show the mock GeRM's frames, gated by the namespace's shutters."""
        user_ns = self._namespace
        mock = getattr(user_ns.get("germ"), "_mock", None)
        if not isinstance(mock, GeRMDetectorMock):
            return
        mock.set_shutters(
            *(
                shutter
                for name in ("fe_shutter", "photon_shutter")
                if isinstance(shutter := user_ns.get(name), Shutter)
            )
        )
        self._show_germ_frame = partial(detectors.show_frame, "germ")
        mock.register_frame_callback(self._show_germ_frame)
        self._germ_mock = mock

    def closeEvent(self, event):
        if self._germ_mock is not None:
            self._germ_mock.unregister_frame_callback(self._show_germ_frame)
            self._germ_mock = None
        super().closeEvent(event)


class QtXRDView(QWidget):
    """X-ray diffraction view: the live Perkin Elmer viewer and XRD plans."""

    def __init__(self, re_client: RunEngineClient, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._re_client = re_client

        top = QHBoxLayout()
        detectors = QtTabbedDetectorsWidget(re_client)
        detectors.add_detector(
            "perkin_elmer", "XF:27ID1-ES{PE-Det:1}", preserve_aspect_ratio=True
        )
        top.addWidget(detectors, stretch=3)

        plan_tabs = _plan_tabs(
            re_client,
            {
                "Dark Reference": "dark_reference",
                "Calibration": "xrd_calibration",
                "Scan": "xrd_scan",
                "2D Scan": "xrd_scan_2d",
                "Radiography": "take_radiograph",
            },
        )
        side = QVBoxLayout()
        side.addWidget(plan_tabs, stretch=1)
        top.addLayout(side, stretch=1)

        vbox = QVBoxLayout()
        vbox.addLayout(top, stretch=1)
        self.setLayout(vbox)


class QtBeamlineView(QWidget):
    """Beamline controls arranged around the visible-light camera viewers.

    Mirrors :class:`QtTomographyView`: the visible-light camera viewers fill the
    majority of the tab and a side column offers the motor and energy plans. The
    shared RE controls and progress bars live outside the tab in
    :class:`QtTabbedTechniqueSelector`.
    """

    def __init__(self, re_client: RunEngineClient, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._re_client = re_client

        vbox = QVBoxLayout()

        top = QHBoxLayout()
        # Visible-light camera viewers take the majority of the screen.
        cameras = QtTabbedDetectorsWidget(re_client)
        for name, prefix in (
            ("fs_window_cam", "XF:27IDA-BI{FS:1-Cam:1}"),
            ("diamond_window_cam", "XF:27IDA-BI{FAM:1-Cam:1}"),
            ("sample_cam", "XF:27ID1-ES{Sample-Cam:1}"),
            ("f_hutch_cam", "XF:27ID1-ES{GigE-Cam:5}"),
        ):
            cameras.add_detector(name, prefix, preserve_aspect_ratio=True)
        top.addWidget(cameras, stretch=3)

        # Side column: a tabbed selector offering the beamline plans.
        plan_tabs = QTabWidget()
        plan_tabs.addTab(QtPlanWidget(re_client, "move_motor"), "Motors")
        plan_tabs.addTab(self._slits_tab(re_client), "Slits")
        plan_tabs.addTab(
            QtPlanWidget(re_client, "set_fe_aperture"), "Frontend Aperture"
        )
        plan_tabs.addTab(self._filters_tab(re_client), "Filters")
        for label, plan in {
            "Change Beam Mode": "change_beam_mode",
            "Change Energy": "change_energy",
            "Reset Scan ID": "reset_scan_id",
            "Settings Snapshot": "store_device_settings_snapshot",
            "Store Settings": "store_device_settings",
            "Apply Settings": "apply_device_settings",
        }.items():
            plan_tabs.addTab(QtPlanWidget(re_client, plan), label)

        side = QVBoxLayout()
        side.addWidget(plan_tabs, stretch=1)
        top.addLayout(side, stretch=1)

        vbox.addLayout(top, stretch=1)

        self.setLayout(vbox)

    @staticmethod
    def _slits_tab(re_client: RunEngineClient) -> QWidget:
        """Live controls for each set of slits."""
        from hextools.gui.hex_config import slits_entries
        from hextools.gui.slits_control import QtSlitsSelector

        return QtSlitsSelector(
            dict(slits_entries(re_client.device_registry)), re_client
        )

    @staticmethod
    def _filters_tab(re_client: RunEngineClient) -> QWidget:
        """Live controls for the filters."""
        from hextools.gui.hex_config import filter_rows
        from hextools.gui.movable_group import QtMovableGroup

        rows, value_types = filter_rows(re_client.device_registry)
        tab = QWidget()
        vbox = QVBoxLayout(tab)
        if rows:
            box = QGroupBox("Filters")
            box_layout = QVBoxLayout(box)
            # Batch by default: each move trips the front-end shutter interlock.
            box_layout.addWidget(
                QtMovableGroup(
                    rows,
                    re_client,
                    value_types=value_types,
                    move_plan="move_filters",
                    tweak_plan=None,
                    batch=True,
                )
            )
            vbox.addWidget(box)
        else:
            vbox.addWidget(QLabel("No filters found"))
        vbox.addStretch(1)
        return tab


class QtTabbedTechniqueSelector(QWidget):
    """Container with shared RE controls, technique tabs, and progress bars.

    The shared RE controls are on top, the technique tabs in the middle, and the
    shared live progress bars along the bottom.
    """

    def __init__(
        self,
        re_client: RunEngineClient,
        namespace: Mapping[str, Any],
        tiled_client: Any = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        from hextools.gui.hex_config import shutter_controls

        self._re_client = re_client
        user_ns = namespace

        vbox = QVBoxLayout()

        # Shared top row: RE controls, metadata, plan status and shutters.
        controls = QHBoxLayout()
        controls.addWidget(QtReExecutionControls(re_client))
        self.proposal_info = QtProposalInfo(re_client)
        controls.addWidget(self.proposal_info)
        controls.addWidget(QtWeatherWidget())
        controls.addWidget(QtPlanStatus(re_client))
        controls.addStretch()
        shutters, shutter_plan = shutter_controls(re_client.device_registry)
        controls.addWidget(QtShutterStatus(re_client, shutters, shutter_plan))
        vbox.addLayout(controls)

        # Technique tabs.
        tabs = QTabWidget()
        tabs.setObjectName("mainViewerTabs")
        tabs.setTabPosition(QTabWidget.TabPosition.West)
        self._beamline = QtBeamlineView(self._re_client)
        tabs.addTab(self._beamline, "Beamline")
        self._tomography = QtTomographyView(self._re_client)
        tabs.addTab(self._tomography, "Tomography")
        self._edxd = QtEDXDView(self._re_client, user_ns)
        tabs.addTab(self._edxd, "EDXD")
        self._xrd = QtXRDView(self._re_client)
        tabs.addTab(self._xrd, "XRD")
        self._available_devices = QtAvailableDevices(re_client)
        tabs.addTab(self._available_devices, "Available Devices")
        execution_view = QtPlanExecutionView(re_client)
        tabs.addTab(execution_view, "Plan Execution")
        self.scan_browser: QtScanBrowser | None = None
        if tiled_client is not None:
            # Every run has the BlueskyRun spec, so selecting it finds all runs.
            self.scan_browser = QtScanBrowser(
                tiled_client,
                [Spec("BlueskyRun", version="3.0"), *ALL_SPECS],
                re_client,
            )
            tabs.addTab(self.scan_browser, "Scan Browser")
        self.plan_history = QtPlanHistory(re_client)
        tabs.addTab(self.plan_history, "History")
        tabs.addTab(QtPlanLogView(re_client), "Log")
        # self.auto_exporter = QtAutoExporterView(autostart=not is_running_in_ci())
        # tabs.addTab(self.auto_exporter, "NXS Export")

        vbox.addWidget(tabs, stretch=1)

        # Shared live per-device progress bars pinned to the bottom.
        vbox.addWidget(QtReWaitingHookMonitor(re_client))

        self.setLayout(vbox)


def _end_ipython_session() -> None:
    """Make the attached IPython shell exit, without waiting for a keypress."""
    ipython = IPython.get_ipython()
    # Only a terminal session; a Jupyter kernel's shell has ask_exit too, and
    # is left running.
    if not isinstance(ipython, TerminalInteractiveShell):
        return
    ipython.ask_exit()
    pt_app = getattr(ipython, "pt_app", None)
    if pt_app is None or not pt_app.app.is_running:
        return
    # Qt runs inside prompt_toolkit's input hook, which only returns when its
    # event loop is woken; call_soon_threadsafe wakes it.
    asyncio.get_running_loop().call_soon_threadsafe(pt_app.app.exit, "")


class _MainWindow(QMainWindow):
    """A ``QMainWindow`` that calls ``on_close`` before Qt deletes its children."""

    def __init__(self, on_close):
        super().__init__()
        self._on_close = on_close

    def closeEvent(self, event):
        self._on_close()
        super().closeEvent(event)


class QtDataAcquisitionWindow:
    """Application window that contains the menu bar and viewer.

    Parameters
    ----------
    re_client : RunEngineClient
        Drives the RunEngine and reads the devices.
    namespace : Mapping[str, Any]
        The IPython session's namespace, for mock detectors.
    tiled_client : tiled client, optional
        Where the Scan Browser finds runs; without one, it isn't shown.

    Attributes
    ----------
    file_menu : qtpy.QtWidgets.QMenu
        File menu.
    help_menu : qtpy.QtWidgets.QMenu
        Help menu.
    main_menu : qtpy.QtWidgets.QMainWindow.menuBar
        Main menubar.
    qt_widget : QtViewer
        Contained viewer widget.
    view_menu : qtpy.QtWidgets.QMenu
        View menu.
    window_menu : qtpy.QtWidgets.QMenu
        Window menu.
    """

    def __init__(
        self,
        re_client: RunEngineClient,
        namespace: Mapping[str, Any],
        tiled_client: Any = None,
        *,
        show: bool = True,
        end_ipython_on_close: bool = False,
    ):
        self.qt_widget = QtTabbedTechniqueSelector(re_client, namespace, tiled_client)

        self._torn_down = False
        self._end_ipython_on_close = end_ipython_on_close
        self._qt_window = _MainWindow(on_close=self._teardown)
        self._qt_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._qt_window.setWindowTitle("HEX Data Acquisition")
        self._qt_window.setUnifiedTitleAndToolBarOnMac(True)
        self._qt_center = QWidget(self._qt_window)

        self._qt_window.setCentralWidget(self._qt_center)
        self._qt_center.setLayout(QHBoxLayout())
        self._status_bar = QStatusBar()
        self._qt_window.setStatusBar(self._status_bar)

        self._re_client = re_client
        # A label, not showMessage(): hovering menu items clears the status bar's
        # message.
        self._re_state_label = QLabel(self._re_state_text())
        self._status_bar.addWidget(self._re_state_label)
        self._help = QLabel("")
        self._status_bar.addPermanentWidget(self._help)
        self._build_menus()
        self._re_state = QtRunEngineState(re_client, self._qt_window)
        self._re_state.changed.connect(lambda *_: self._update_re_state())

        layout = self._qt_center.layout()
        if layout is not None:
            layout.addWidget(self.qt_widget)
        else:
            raise RuntimeError("Failed to get layout for central widget.")

        # self._add_viewer_dock_widget(self.qt_widget.dockConsole)
        # self._add_viewer_dock_widget(self.qt_widget.dockLayerControls)
        # self._add_viewer_dock_widget(self.qt_widget.dockLayerList)

        # self.qt_widget.viewer.events.status.connect(self._status_changed)
        # self.qt_widget.viewer.events.help.connect(self._help_changed)
        # self.qt_widget.viewer.events.title.connect(self._title_changed)
        # self.qt_widget.viewer.events.palette.connect(self._update_palette)

        # In IPython the session can end with the window still open.
        atexit.register(self._close_at_exit)

        if show:
            self.show()

    def _close_at_exit(self):
        # IPython is already exiting; don't ask it again.
        self._end_ipython_on_close = False
        self._qt_window.close()

    def _teardown(self):
        """Undo what the GUI attached outside itself: RunEngine hooks, child process."""
        if self._torn_down:
            return
        self._torn_down = True
        atexit.unregister(self._close_at_exit)
        self._re_state.stop()
        if self._beamline_status is not None:
            self._beamline_status.close()
        # WA_DeleteOnClose deletes the children without sending them closeEvent.
        for widget in self._qt_window.findChildren(QWidget):
            # Closing one widget can delete others (pyqtgraph deletes its viewport).
            if shiboken.isValid(widget):
                widget.close()
        if self._end_ipython_on_close:
            _end_ipython_session()

    def _build_menus(self):
        menu_bar = self._qt_window.menuBar()

        self.file_menu = menu_bar.addMenu("&File")
        change_proposal = self.file_menu.addAction("Change Proposal…")
        change_proposal.triggered.connect(self.qt_widget.proposal_info.change_proposal)
        self.file_menu.addSeparator()
        save_history = self.file_menu.addAction("Save Plan History As…")
        save_history.triggered.connect(self._on_save_history_as)
        screenshot = self.file_menu.addAction("Take Screenshot…")
        screenshot.triggered.connect(self._on_take_screenshot)

        self.view_menu = menu_bar.addMenu("&View")
        beamline_status = self.view_menu.addAction("Beamline Status")
        beamline_status.triggered.connect(self._on_show_beamline_status)
        self._beamline_status: QtBeamlineStatus | None = None

        self.settings_menu = menu_bar.addMenu("&Settings")
        self.settings_menu.addAction(QtThemeAction(self._qt_window))

        self.help_menu = menu_bar.addMenu("&Help")
        self.help_menu.addAction("About", self._on_about)

    def _on_show_beamline_status(self):
        if self._beamline_status is None:
            from hextools.gui.hex_config import status_sections

            re_client = self._re_client
            # No parent: a parented window is kept above the main window.
            window = QtBeamlineStatus(
                status_sections(re_client),
                re_client,
                title=None,
            )
            window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            window.setWindowTitle("HEX Beamline Status")
            window.destroyed.connect(self._on_beamline_status_destroyed)
            screen = self._qt_window.screen().availableGeometry()
            window.resize(
                min(max(window.sizeHint().width(), 1400), screen.width()),
                min(max(window.sizeHint().height(), 700), screen.height()),
            )
            self._beamline_status = window
        self._beamline_status.show()
        self._beamline_status.raise_()
        self._beamline_status.activateWindow()

    def _on_beamline_status_destroyed(self, *_):
        self._beamline_status = None

    def _on_about(self):
        import platform
        from importlib.metadata import PackageNotFoundError, version

        rows = [
            f"<tr><td><b>python</b></td><td>&nbsp;&nbsp;{platform.python_version()}</td></tr>"
        ]
        for package in (
            "hextools",
            "bluesky",
            "ophyd-async",
            "tiled",
            "bluesky-tiled-plugins",
        ):
            try:
                ver = version(package)
            except PackageNotFoundError:
                ver = "not installed"
            rows.append(f"<tr><td><b>{package}</b></td><td>&nbsp;&nbsp;{ver}</td></tr>")
        box = QMessageBox(
            QMessageBox.Icon.Information,
            "About HEX Data Acquisition",
            f"<h3>HEX Data Acquisition</h3><table>{''.join(rows)}</table>",
            QMessageBox.StandardButton.Ok,
            self._qt_window,
        )
        box.exec()

    def _on_save_history_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self._qt_window,
            "Save Plan History As",
            str(Path.home() / "plan_history.json"),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            self._re_client.save_history(path)
        except OSError as ex:
            QMessageBox.critical(
                self._qt_window, "Save failed", f"Couldn't save plan history:\n{ex}"
            )
            return
        self._status_bar.showMessage(f"Plan history saved to {path}", 5000)

    def _on_take_screenshot(self):
        # Grab before the file dialog opens, so it isn't in the picture.
        pixmap = self._qt_window.grab()
        default = (
            Path.home() / f"hex_daq_screenshot_{ttime.strftime('%Y%m%d_%H%M%S')}.png"
        )
        path, _ = QFileDialog.getSaveFileName(
            self._qt_window, "Save Screenshot", str(default), "PNG images (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        if not pixmap.save(path, "PNG"):
            QMessageBox.critical(
                self._qt_window, "Save failed", f"Couldn't save screenshot to:\n{path}"
            )
            return
        self._status_bar.showMessage(f"Screenshot saved to {path}", 5000)

    def _update_re_state(self):
        self._re_state_label.setText(self._re_state_text())

    def _re_state_text(self) -> str:
        return f"RunEngine: {self._re_client.state.capitalize()}"

    def resize(self, width, height):
        """Resize the window.

        Parameters
        ----------
        width : int
            Width in logical pixels.
        height : int
            Height in logical pixels.
        """
        self._qt_window.resize(width, height)

    def show(self):
        """Resize, show, and bring forward the window."""
        self._qt_window.resize(1650, 1280)
        self._qt_window.show()

        # We want to call Window._qt_window.raise_() in every case *except*
        # when instantiating a viewer within a gui_qt() context for the
        # _first_ time within the Qt app's lifecycle.
        #
        # `app_name` will be ours iff the application was instantiated in
        # gui_qt(). isActiveWindow() will be True if it is the second time a
        # _qt_window has been created. See #732
        app = QApplication.instance()
        if app is None:
            raise RuntimeError("Failed to get QApplication instance.")
        app_name = app.applicationName()
        if app_name != get_our_app_name() or self._qt_window.isActiveWindow():
            self._qt_window.raise_()  # for macOS
            self._qt_window.activateWindow()  # for Windows

    def _status_changed(self, event):
        """Update status bar.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self._status_bar.showMessage(event.text)

    def _title_changed(self, event):
        """Update window title.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self._qt_window.setWindowTitle(event.text)

    def _help_changed(self, event):
        """Update help message on status bar.

        Parameters
        ----------
        event : qtpy.QtCore.QEvent
            Event from the Qt context.
        """
        self._help.setText(event.text)

    def close(self):
        """Close the viewer window and cleanup sub-widgets."""
        # on some versions of Darwin, exiting while fullscreen seems to tickle
        # some bug deep in NSWindow.  This forces the fullscreen keybinding
        # test to complete its draw cycle, then pop back out of fullscreen.
        if self._qt_window.isFullScreen():
            self._qt_window.showNormal()
            for _i in range(8):
                ttime.sleep(0.1)
                QApplication.processEvents()
        self.qt_widget.close()
        self._qt_window.close()
        wait_for_workers_to_quit()
        del self._qt_window


def launch_local_viewer(end_ipython_on_close: bool = False):
    """Create and return the GUI in in-process (local) mode.

    Intended to be run from within an IPython session that already has the
    RunEngine (``RE``) and devices loaded, so the plan widget can execute plans
    via ``RE(plan(...))``. With ``end_ipython_on_close``, closing the window also
    ends that IPython session (what ``hex-gui`` does, since it started it).
    """
    from qtpy.QtWidgets import QApplication

    # IPython's Qt event loop hook runs after startup, so create the
    # QApplication now to build widgets safely.
    app = QApplication.instance() or QApplication([])
    apply_bnl_theme(app, saved_theme())
    user_ns = IPython.get_ipython().user_ns
    history_file = os.environ.get(PLAN_HISTORY_FILE_ENV) or default_plan_history_file()
    return QtDataAcquisitionWindow(
        InProcessRunEngineClient(
            user_ns, logger_name="hextools", history_file=history_file
        ),
        user_ns,
        user_ns.get("tiled_reading_client") or user_ns.get("c"),
        end_ipython_on_close=end_ipython_on_close,
    )


def main():
    """Launch the HEX Data Acquisition GUI."""
    parser = argparse.ArgumentParser(description="HEX Data Acquisition")
    parser.add_argument(
        "--profile",
        default="collection",
        help="Profile to start IPython on, with the GUI attached, running plans"
        " via RE(plan(...)).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run devices in mock/simulation mode by setting"
        " HEXTOOLS_RUNNING_IN_CI=YES.",
    )
    parser.add_argument(
        "--history-file",
        default=None,
        help="JSON file to save plan history to as plans finish, and to load it"
        " from on startup (default: ~/.local/state/hextools/plan_history.json,"
        " or plan_history_mock.json with --mock). Can also be set"
        f" with {PLAN_HISTORY_FILE_ENV}.",
    )

    parser.add_argument(
        "--keep-ipython",
        action="store_true",
        help="Keep the IPython session open when the GUI window is closed"
        " (by default, closing the window also ends it).",
    )

    args = parser.parse_args()

    if args.mock:
        os.environ["HEXTOOLS_RUNNING_IN_CI"] = "YES"
    if args.history_file:
        # An env var, so it reaches the GUI built later inside the IPython session.
        os.environ[PLAN_HISTORY_FILE_ENV] = args.history_file

    os.environ["BEAMLINE_ACRONYM"] = "HEX"

    from IPython import start_ipython
    from traitlets.config import Config

    profile_path = (
        Path(__file__).resolve().parent.parent / "profiles" / f"{args.profile}.py"
    )
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    # Mirror the environment tweaks from the pixi `start` task.
    for var in ("SESSION_MANAGER", "PYTHONPATH", "PYTHONUSERBASE"):
        os.environ.pop(var, None)
    os.environ["MPLBACKEND"] = "qtagg"

    config = Config()
    config.InteractiveShellApp.gui = "qt"
    # Run the profile first (defining RE and devices), then attach the GUI.
    # exec_lines run in order, before any command-line files.
    config.InteractiveShellApp.exec_lines = [
        f"get_ipython().run_line_magic('run', {f'-i {profile_path}'!r})",
        "from hextools.gui.__main__ import launch_local_viewer",
        f"launch_local_viewer(end_ipython_on_close={not args.keep_ipython})",
    ]
    config.TerminalIPythonApp.display_banner = False
    start_ipython(argv=[], config=config)


if __name__ == "__main__":
    main()
