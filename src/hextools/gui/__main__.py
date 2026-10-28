"""HEX queue-monitor GUI.

A recreation of the bluesky-widgets ``queue_monitor`` application, consolidated
into a single entrypoint so it can serve as the main HEX GUI. Run with::

    python -m hextools.gui

With ``--queueserver-uri`` (or ``QSERVER_HTTP_SERVER_URI``) the GUI connects to a
QueueServer and submits plans to its queue. Otherwise it starts IPython on a
profile (``--profile``, default ``collection``) with the GUI attached and runs
plans in-process via ``RE(plan(...))``.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import os
import time as ttime
from functools import partial
from pathlib import Path
from typing import Generic, TypeVar

import IPython
from bluesky import RunEngine
from bluesky_widgets.models.run_engine_client import RunEngineClient
from bluesky_widgets.qt.run_engine_client import (
    QLabel,
    QtReConsoleMonitor,
    QtReEnvironmentControls,
    QtReExecutionControls,
    QtReManagerConnection,
    QtRePlanEditor,
    QtRePlanHistory,
    QtRePlanQueue,
    QtReQueueControls,
    QtReRunningPlan,
    QtReStatusMonitor,
)
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.epics.advimba import VimbaDetector
from ophyd_async.fastcs.panda import HDFPanda
from qtpy import shiboken
from qtpy.QtCore import QObject, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hextools.detectors.germ import GeRMDetectorMock
from hextools.detectors.phantom import PhantomDetector
from hextools.edxd import (
    configure_test_pulses,
    edxd_2theta_tilt,
    edxd_calib_scan,
    edxd_count,
    edxd_custom_pos_list_grid,
    edxd_grid_scan,
    edxd_scan,
)
from hextools.gui._hook_chain import install_hook, uninstall_hook
from hextools.gui._theme import apply_bnl_theme, saved_theme
from hextools.gui.auto_exporter_view import QtAutoExporterView
from hextools.gui.available_devices import QtAvailableDevices
from hextools.gui.device_progress_viewer import QtReWaitingHookMonitor
from hextools.gui.live_re_md_viewer import QtProposalInfo
from hextools.gui.misc import QtTabbedDetectorsWidget, QtWeatherWidget
from hextools.gui.plan_status import (
    PlanMonitor,
    QtPlanExecutionView,
    QtPlanHistory,
    QtPlanLogView,
    QtPlanStatus,
)
from hextools.gui.plan_widget import QtPlanWidget
from hextools.gui.re_execution_controls import (
    QtReExecutionControls as QtReExecutionControlsLocal,
)
from hextools.gui.scan_browser import QtScanBrowser
from hextools.gui.shutter_status import QtShutterStatus
from hextools.gui.theme_switch import QtThemeAction
from hextools.machine import NSLS2StorageRing
from hextools.motors import (
    FOV_2_4_mm_Camera,
    FOV_20_40_mm_Camera,
    OpticsTable,
    SampleTower,
    move_motor,
)
from hextools.photon_delivery_system import (
    DCLM,
    FrontEndSlit,
    PPSShutter,
    Shutter,
    Slits,
    change_energy,
    move_slits,
    set_fe_slit1_aperture,
    set_filters,
)
from hextools.photon_delivery_system.dclm import change_beam_mode
from hextools.settings import (
    apply_device_settings,
    store_device_settings,
    store_device_settings_snapshot,
)
from hextools.specs import ALL_SPECS
from hextools.tomography.alignment import tomo_alignment_scan
from hextools.tomography.flyscans import (
    tomo_1d_step_scan,
    tomo_2d_step_scan,
    tomo_dark_flat,
    tomo_flyscan,
    tomo_flyscan_loop,
)
from hextools.tomography.radiography import take_radiograph
from hextools.utils import is_running_in_ci, reset_scan_id

from ._event_loop import get_our_app_name, gui_qt
from ._threading import wait_for_workers_to_quit

RunEngineClientT = TypeVar("RunEngineClientT", bound=RunEngineClient | RunEngine)

PLAN_HISTORY_FILE_ENV = "HEXTOOLS_PLAN_HISTORY_FILE"


def default_plan_history_file() -> Path:
    """``$XDG_STATE_HOME/hextools/plan_history[_mock].json``."""
    state_home = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    name = "plan_history_mock.json" if is_running_in_ci() else "plan_history.json"
    return Path(state_home) / "hextools" / name


# Names must match those defined in the profile / Queue Server namespace.
EXPECTED_DEVICES: list[tuple[str, type]] = [
    ("fe_shutter", PPSShutter),
    ("photon_shutter", Shutter),
    ("a_slits", Slits),
    ("f_slits", Slits),
    ("fe_slit1", FrontEndSlit),
    ("storage_ring", NSLS2StorageRing),
    ("dclm", DCLM),
    ("optics_table", OpticsTable),
    ("sample_tower", SampleTower),
    ("panda", HDFPanda),
    ("kinetix1", KinetixDetector),
    ("kinetix2", KinetixDetector),
    ("kinetix3", KinetixDetector),
    ("kinetix4", KinetixDetector),
    ("double_obj_camera", FOV_2_4_mm_Camera),
    ("wide_fov_camera", FOV_20_40_mm_Camera),
    ("phantom", PhantomDetector),
    ("sample_cam", VimbaDetector),
    ("f_hutch_cam", VimbaDetector),
]


def _patch_setchecked_checkstate():
    """Let ``setChecked`` accept ``Qt.CheckState`` enums.

    Needed for bluesky-widgets on PySide6>=6.9.
    """
    from qtpy.QtWidgets import QAbstractButton

    original = QAbstractButton.setChecked

    def set_checked(self, value):
        if not isinstance(value, (bool, int)):
            value = getattr(value, "value", value)
        return original(self, bool(value))

    QAbstractButton.setChecked = set_checked


_patch_setchecked_checkstate()


class QtOrganizeQueueWidgets(QSplitter):
    """Vertically stacked monitor-mode queue, history, and console widgets."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

        self.setOrientation(Qt.Orientation.Vertical)

        self._frame_1 = QFrame(self)
        self._frame_2 = QFrame(self)
        self._frame_3 = QFrame(self)

        self.addWidget(self._frame_1)
        self.addWidget(self._frame_2)
        self.addWidget(self._frame_3)

        self._running_plan = QtReRunningPlan(model)
        self._running_plan.monitor_mode = True
        self._plan_queue = QtRePlanQueue(model)
        self._plan_queue.monitor_mode = True
        self._plan_history = QtRePlanHistory(model)
        self._plan_history.monitor_mode = True
        self._console_monitor = QtReConsoleMonitor(model)

        vbox = QVBoxLayout()
        vbox.addWidget(self._running_plan)
        self._frame_1.setLayout(vbox)

        vbox = QVBoxLayout()
        vbox.addWidget(self._plan_queue)
        vbox.addWidget(self._plan_history)
        self._frame_2.setLayout(vbox)

        vbox = QVBoxLayout()
        vbox.addWidget(self._console_monitor)
        self._frame_3.setLayout(vbox)


class QtRunEngineManagerMonitor(QWidget):
    """Read-only view of the queue, history, and running plan."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(QtReManagerConnection(model))
        hbox.addWidget(QtReStatusMonitor(model))
        hbox.addWidget(QtProposalInfo(model))
        hbox.addWidget(QtWeatherWidget())
        hbox.addStretch()
        vbox.addLayout(hbox)

        vbox.addWidget(QtOrganizeQueueWidgets(model), stretch=2)

        self.setLayout(vbox)


class QtRunEngineManagerEditor(QWidget):
    """Full control view with environment, queue, and plan editing."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(QtReEnvironmentControls(model))
        hbox.addWidget(QtReQueueControls(model))
        hbox.addWidget(QtReExecutionControls(model))
        hbox.addWidget(QtReStatusMonitor(model))
        hbox.addStretch()
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        vbox1 = QVBoxLayout()

        # Register plan editor so double-clicking a queue item opens it.
        pe = QtRePlanEditor(model)
        pq = QtRePlanQueue(model)
        pq.registered_item_editors.append(pe.edit_queue_item)

        vbox1.addWidget(pe, stretch=1)
        vbox1.addWidget(pq, stretch=1)
        hbox.addLayout(vbox1)

        vbox2 = QVBoxLayout()
        vbox2.addWidget(QtReRunningPlan(model), stretch=1)
        vbox2.addWidget(QtRePlanHistory(model), stretch=2)
        hbox.addLayout(vbox2)
        vbox.addLayout(hbox)
        self.setLayout(vbox)


class QtTomographyView(QWidget):
    """Tomography controls arranged around the live detector viewers.

    Detector viewers fill the majority of the tab and the tomography plan
    selector sits in a side column. The shared RE controls and progress bars
    live outside the tab in :class:`QtTabbedTechniqueSelector`.
    """

    def __init__(self, re_client: RunEngineClient | RunEngine, *args, **kwargs):
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
            )
        detectors.add_combined("Dual Cam", ["kinetix1", "kinetix3"])
        top.addWidget(detectors, stretch=3)

        # Side column: a tabbed selector offering the tomography plans. Each tab
        # validates and runs its plan per the active execution mode (in-process
        # IPython vs. Queue Server).
        plan_tabs = QTabWidget()
        plan_tabs.addTab(QtPlanWidget(re_client, tomo_dark_flat), "Dark/Flat")
        plan_tabs.addTab(QtPlanWidget(re_client, tomo_flyscan), "Flyscan")
        plan_tabs.addTab(QtPlanWidget(re_client, tomo_flyscan_loop), "Flyscan Loop")
        plan_tabs.addTab(QtPlanWidget(re_client, tomo_1d_step_scan), "1D Step")
        plan_tabs.addTab(QtPlanWidget(re_client, tomo_2d_step_scan), "2D Step")
        plan_tabs.addTab(QtPlanWidget(re_client, take_radiograph), "Radiography")
        plan_tabs.addTab(QtPlanWidget(re_client, tomo_alignment_scan), "Alignment")

        side = QVBoxLayout()
        side.addWidget(plan_tabs, stretch=1)
        top.addLayout(side, stretch=1)

        vbox.addLayout(top, stretch=1)

        self.setLayout(vbox)


class QtEDXDView(QWidget):
    """Energy-dispersive X-ray diffraction view: the live GeRM viewer and EDXD plans."""

    def __init__(self, re_client: RunEngineClient | RunEngine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._re_client = re_client

        top = QHBoxLayout()
        detectors = QtTabbedDetectorsWidget(re_client)
        detectors.add_detector(
            "germ",
            "XF:27ID1-ES{GeRM-Det:1}MCA",
            raw_waveform=True,
            image_shape=(192, 4096),
            colormap="JET",
            roi_suffix_pattern="1:ROI{}:",
            auto_scale_method="minmax",
            log_scale=True,
            use_ca=True,
            show_profile_lines=True,
        )
        top.addWidget(detectors, stretch=3)

        plan_tabs = QTabWidget()
        plan_tabs.addTab(QtPlanWidget(re_client, edxd_2theta_tilt), "2Theta Tilt")
        plan_tabs.addTab(QtPlanWidget(re_client, edxd_calib_scan), "Calibration")
        plan_tabs.addTab(QtPlanWidget(re_client, edxd_scan), "Scan")
        plan_tabs.addTab(QtPlanWidget(re_client, edxd_grid_scan), "Grid Scan")
        plan_tabs.addTab(
            QtPlanWidget(re_client, edxd_custom_pos_list_grid), "Custom Pos List Grid"
        )
        plan_tabs.addTab(QtPlanWidget(re_client, edxd_count), "Count")
        plan_tabs.addTab(
            QtPlanWidget(re_client, configure_test_pulses), "Configure Test Pulses"
        )
        side = QVBoxLayout()
        side.addWidget(plan_tabs, stretch=1)
        top.addLayout(side, stretch=1)

        vbox = QVBoxLayout()
        vbox.addLayout(top, stretch=1)
        self.setLayout(vbox)

        self._germ_mock: GeRMDetectorMock | None = None
        if isinstance(re_client, RunEngine) and is_running_in_ci():
            self._attach_germ_mock(detectors)

    def _attach_germ_mock(self, detectors: QtTabbedDetectorsWidget) -> None:
        """Show the mock GeRM's frames, gated by the namespace's shutters."""
        ipython = IPython.get_ipython()
        user_ns = ipython.user_ns if ipython is not None else {}
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
    """X-ray diffraction view: the live Perkin Elmer area detector viewer."""

    def __init__(self, re_client: RunEngineClient | RunEngine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._re_client = re_client

        vbox = QVBoxLayout()
        detectors = QtTabbedDetectorsWidget(re_client)
        detectors.add_detector("perkin_elmer", "XF:27ID1-ES{PE-Det:1}")
        vbox.addWidget(detectors, stretch=1)
        self.setLayout(vbox)


class QtBeamlineView(QWidget):
    """Beamline controls arranged around the visible-light camera viewers.

    Mirrors :class:`QtTomographyView`: the visible-light camera viewers fill the
    majority of the tab and a side column offers the motor and energy plans. The
    shared RE controls and progress bars live outside the tab in
    :class:`QtTabbedTechniqueSelector`.
    """

    def __init__(self, re_client: RunEngineClient | RunEngine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._re_client = re_client

        vbox = QVBoxLayout()

        top = QHBoxLayout()
        # Visible-light camera viewers take the majority of the screen.
        cameras = QtTabbedDetectorsWidget(re_client)
        cameras.add_detector("fs_window_cam", "XF:27IDA-BI{FS:1-Cam:1}")
        cameras.add_detector("diamond_window_cam", "XF:27IDA-BI{FAM:1-Cam:1}")
        cameras.add_detector("sample_cam", "XF:27ID1-ES{Sample-Cam:1}")
        cameras.add_detector("f_hutch_cam", "XF:27IDA-BI{GigE-Cam:5}")
        top.addWidget(cameras, stretch=3)

        # Side column: a tabbed selector offering the beamline plans. Each tab
        # validates and runs its plan per the active execution mode (in-process
        # IPython vs. Queue Server).
        plan_tabs = QTabWidget()
        plan_tabs.addTab(QtPlanWidget(re_client, move_motor), "Motors")
        plan_tabs.addTab(QtPlanWidget(re_client, move_slits), "Slits")
        plan_tabs.addTab(
            QtPlanWidget(re_client, set_fe_slit1_aperture), "Frontend Aperture"
        )
        plan_tabs.addTab(QtPlanWidget(re_client, set_filters), "Filters")
        plan_tabs.addTab(QtPlanWidget(re_client, change_beam_mode), "Change Beam Mode")
        plan_tabs.addTab(QtPlanWidget(re_client, change_energy), "Change Energy")
        plan_tabs.addTab(QtPlanWidget(re_client, reset_scan_id), "Reset Scan ID")
        plan_tabs.addTab(
            QtPlanWidget(re_client, store_device_settings_snapshot), "Settings Snapshot"
        )
        plan_tabs.addTab(
            QtPlanWidget(re_client, store_device_settings), "Store Settings"
        )
        plan_tabs.addTab(
            QtPlanWidget(re_client, apply_device_settings), "Apply Settings"
        )

        side = QVBoxLayout()
        side.addWidget(plan_tabs, stretch=1)
        top.addLayout(side, stretch=1)

        vbox.addLayout(top, stretch=1)

        self.setLayout(vbox)


class QtTabbedTechniqueSelector(QWidget, Generic[RunEngineClientT]):
    """Container with shared RE controls, technique tabs, and progress bars.

    The shared RE controls are on top, the technique tabs in the middle, and the
    shared live progress bars along the bottom.
    """

    def __init__(self, re_client: RunEngineClientT, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._re_client = re_client

        vbox = QVBoxLayout()

        # Shared top row: queue/plan controls (Queue Server only) and RE
        # metadata. In-process mode reads metadata from the local RunEngine.
        controls = QHBoxLayout()
        if isinstance(re_client, RunEngineClient):
            controls.addWidget(QtReQueueControls(re_client))
            controls.addWidget(QtReExecutionControls(re_client))
            controls.addWidget(QtReRunningPlan(re_client))
        else:
            controls.addWidget(QtReExecutionControlsLocal(local=True))
        self.proposal_info = QtProposalInfo(re_client)
        controls.addWidget(self.proposal_info)
        controls.addWidget(QtWeatherWidget())
        ipython = IPython.get_ipython()
        user_ns = ipython.user_ns if ipython is not None else {}
        self._plan_monitor = (
            PlanMonitor(re_client, namespace=user_ns, parent=self)
            if isinstance(re_client, RunEngine)
            else None
        )
        if self._plan_monitor is not None:
            controls.addWidget(QtPlanStatus(self._plan_monitor, "hextools"))
        controls.addStretch()
        if isinstance(re_client, RunEngine) and ipython is not None:
            controls.addWidget(QtShutterStatus(re_client, user_ns))
        vbox.addLayout(controls)

        # Technique tabs.
        tabs = QTabWidget()
        tabs.setObjectName("mainViewerTabs")
        tabs.setTabPosition(QTabWidget.TabPosition.West)
        self._beamline = QtBeamlineView(self._re_client)
        tabs.addTab(self._beamline, "Beamline")
        self._tomography = QtTomographyView(self._re_client)
        tabs.addTab(self._tomography, "Tomography")
        self._edxd = QtEDXDView(self._re_client)
        tabs.addTab(self._edxd, "EDXD")
        self._xrd = QtXRDView(self._re_client)
        tabs.addTab(self._xrd, "XRD")
        self._available_devices = QtAvailableDevices(self._re_client, EXPECTED_DEVICES)
        tabs.addTab(self._available_devices, "Available Devices")
        if self._plan_monitor is not None:
            execution_view = QtPlanExecutionView(self._plan_monitor, "hextools")
            tiled_client = user_ns.get("tiled_reading_client") or user_ns.get("c")
            if tiled_client is not None and isinstance(re_client, RunEngine):
                self.scan_browser = QtScanBrowser(tiled_client, ALL_SPECS, re_client)
                execution_view.add_tab(self.scan_browser, "Scan Browser")
            else:
                self.scan_browser = None
            tabs.addTab(execution_view, "Plan Execution")
            self.plan_history = QtPlanHistory(
                self._plan_monitor,
                history_file=os.environ.get(PLAN_HISTORY_FILE_ENV)
                or default_plan_history_file(),
            )
            tabs.addTab(self.plan_history, "History")
            tabs.addTab(QtPlanLogView(self._plan_monitor, "hextools"), "Log")
        else:
            self.plan_history = None
            self.scan_browser = None
        self.auto_exporter = QtAutoExporterView(autostart=not is_running_in_ci())
        tabs.addTab(self.auto_exporter, "NXS Export")
        vbox.addWidget(tabs, stretch=1)

        # Shared live per-device progress bars pinned to the bottom.
        vbox.addWidget(QtReWaitingHookMonitor(re_client))

        # Queue status pinned to the bottom of the window (Queue Server only).
        if isinstance(re_client, RunEngineClient):
            vbox.addWidget(QtRePlanQueue(re_client))

        self.setLayout(vbox)

    def closeEvent(self, event):
        if self._plan_monitor is not None:
            self._plan_monitor.detach()
        super().closeEvent(event)


class _StateSignal(QObject):
    changed = Signal()


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
    qt_widget : QtViewer
        Contained viewer widget.

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
        re_client: RunEngineClient | RunEngine,
        *,
        show: bool = True,
        end_ipython_on_close: bool = False,
    ):
        self.qt_widget = QtTabbedTechniqueSelector(re_client)

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
        if isinstance(re_client, RunEngine):
            self._install_state_hook(re_client)
        else:
            self._re_state_timer = QTimer(self._qt_window)
            self._re_state_timer.timeout.connect(self._update_re_state)
            self._re_state_timer.start(250)

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
        if isinstance(self._re_client, RunEngine):
            self._uninstall_state_hook(self._re_client)
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
        if self.qt_widget.plan_history is None:
            save_history.setEnabled(False)
            save_history.setToolTip(
                "Plan history is only recorded when running plans locally."
            )
        screenshot = self.file_menu.addAction("Take Screenshot…")
        screenshot.triggered.connect(self._on_take_screenshot)

        self.settings_menu = menu_bar.addMenu("&Settings")
        self.settings_menu.addAction(QtThemeAction(self._qt_window))

        self.help_menu = menu_bar.addMenu("&Help")
        self.help_menu.addAction("About", self._on_about)

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
        history = self.qt_widget.plan_history
        if history is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self._qt_window,
            "Save Plan History As",
            str(Path.home() / "plan_history.json"),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            history.save_as(path)
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

    def _install_state_hook(self, re: RunEngine):
        self._state_signal = _StateSignal(self._qt_window)
        self._state_signal.changed.connect(self._update_re_state)
        # Called from the RunEngine's event-loop thread.
        self._state_hook = install_hook(
            re, "state_hook", lambda *_: self._state_signal.changed.emit()
        )

    def _uninstall_state_hook(self, re: RunEngine):
        uninstall_hook(re, "state_hook", self._state_hook)

    def _update_re_state(self):
        self._re_state_label.setText(self._re_state_text())

    def _re_state_text(self) -> str:
        if isinstance(self._re_client, RunEngine):
            state = self._re_client.state
        else:
            status = self._re_client.re_manager_status or {}
            state = status.get("re_state") or status.get("manager_state") or "unknown"
        return f"RunEngine: {str(state).capitalize()}"

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
        window_layout = self._qt_window.layout()
        if window_layout is not None:
            self._qt_window.resize(window_layout.sizeHint())
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
    re = IPython.get_ipython().user_ns.get("RE", None)
    if not isinstance(re, RunEngine):
        raise RuntimeError(
            "RE not found in IPython user namespace or is not a RunEngine instance."
        )

    return QtDataAcquisitionWindow(
        re_client=re, end_ipython_on_close=end_ipython_on_close
    )


def main():
    """Launch the HEX Data Acquisition GUI."""
    parser = argparse.ArgumentParser(description="HEX Queue Monitor")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--queueserver-uri",
        default=None,
        help="Address of the Bluesky QueueServer http server. If set, connect to"
        " the QueueServer via HTTP and submit plans to its queue.",
    )
    mode.add_argument(
        "--profile",
        default="collection",
        help="Profile to load when running acquisition in-process. If set (or if"
        " no QueueServer is given), start IPython on this profile with the GUI"
        " attached, running plans via RE(plan(...)).",
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

    if args.queueserver_uri:
        with gui_qt("HEX Queue Monitor"):
            apply_bnl_theme(theme=saved_theme())
            re_client = RunEngineClient(http_server_uri=args.queueserver_uri)
            QtDataAcquisitionWindow(re_client=re_client)
    else:
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
