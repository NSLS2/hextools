"""The data-acquisition window, built from a beamline's :class:`GuiSpec`.

Nothing here is specific to one beamline: the techniques, plans, cameras,
devices, proposal system and site all come from the spec (for HEX,
``hextools.gui_config.HEX_GUI``). :func:`run` is the command-line entry point.

With ``--queueserver-uri`` (or ``QSERVER_HTTP_SERVER_URI``) the GUI connects to a
QueueServer and submits plans to its queue. Otherwise it starts IPython on a
profile (``--profile``) and opens the GUI in its own process, attached to that
session (:mod:`hextools.gui.session_server`), so a crash in the GUI cannot stop
a plan or its cleanup.
"""

from __future__ import annotations

import argparse
import logging
import os
import time as ttime
from pathlib import Path
from typing import Any, Generic, TypeVar

from bluesky_widgets.models.run_engine_client import RunEngineClient
from bluesky_widgets.qt.run_engine_client import (
    QLabel,
    QtRePlanQueue,
    QtReQueueControls,
    QtReRunningPlan,
)
from bluesky_widgets.qt.run_engine_client import (
    QtReExecutionControls as QtReExecutionControlsRemote,
)
from qtpy import shiboken
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hextools.gui._theme import apply_bnl_theme, saved_theme, set_settings_scope
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
from hextools.gui.scan_browser import QtScanBrowser
from hextools.gui.session_client import SessionClient
from hextools.gui.spec import GuiSpec, Technique, load_spec
from hextools.gui.theme_switch import QtThemeAction

from ._event_loop import get_our_app_name, gui_qt
from ._threading import wait_for_workers_to_quit

logger = logging.getLogger(__name__)

RunEngineClientT = TypeVar("RunEngineClientT", bound=RunEngineClient | SessionClient)


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


class QtTechniqueView(QWidget):
    """One technique tab: its detector viewers beside a column of plan forms.

    Detector viewers take the majority of the tab; with no plans they fill it,
    with no detectors the plan forms do. The shared RE controls and progress bars
    live outside the tab in :class:`QtTabbedTechniqueSelector`.
    """

    def __init__(
        self,
        re_client: RunEngineClient | SessionClient,
        technique: Technique,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._re_client = re_client
        self.technique = technique

        top = QHBoxLayout()
        if technique.detectors:
            detectors = QtTabbedDetectorsWidget(re_client)
            for view in technique.detectors:
                detectors.add_detector(view.name, view.prefix, **view.options)
            for label, names in technique.combined:
                detectors.add_combined(label, list(names))
            top.addWidget(detectors, stretch=3)

        if technique.plans:
            # Each tab validates and runs its plan per the active execution mode
            # (the IPython session vs. Queue Server).
            plan_tabs = QTabWidget()
            for label, plan in technique.plans:
                plan_tabs.addTab(QtPlanWidget(re_client, plan), label)
            side = QVBoxLayout()
            side.addWidget(plan_tabs, stretch=1)
            top.addLayout(side, stretch=1)

        vbox = QVBoxLayout()
        vbox.addLayout(top, stretch=1)
        self.setLayout(vbox)


class QtTabbedTechniqueSelector(QWidget, Generic[RunEngineClientT]):
    """Container with shared RE controls, technique tabs, and progress bars.

    The shared RE controls are on top, the technique tabs in the middle, and the
    shared live progress bars along the bottom.
    """

    def __init__(self, re_client: RunEngineClientT, spec: GuiSpec, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._re_client = re_client
        self.spec = spec

        vbox = QVBoxLayout()

        # Shared top row: queue/plan controls (Queue Server only) and RE
        # metadata.
        controls = QHBoxLayout()
        session = re_client if isinstance(re_client, SessionClient) else None
        if session is None:
            controls.addWidget(QtReQueueControls(re_client))
            controls.addWidget(QtReExecutionControlsRemote(re_client))
            controls.addWidget(QtReRunningPlan(re_client))
        else:
            controls.addWidget(QtReExecutionControls(session=session))
        self.proposal_info = QtProposalInfo(re_client, proposals=spec.proposals)
        controls.addWidget(self.proposal_info)
        if spec.site is not None:
            controls.addWidget(QtWeatherWidget(spec.site))
        if session is not None:
            controls.addWidget(QtPlanStatus(session, spec.logger_name))
        controls.addStretch()
        if session is not None:
            for factory in spec.status_widgets:
                controls.addWidget(factory(session))
        vbox.addLayout(controls)

        # Technique tabs, then the built-in tabs, then the beamline's extra tabs.
        tabs = QTabWidget()
        tabs.setObjectName("mainViewerTabs")
        tabs.setTabPosition(QTabWidget.TabPosition.West)
        self.technique_views: dict[str, QtTechniqueView] = {}
        for technique in spec.techniques:
            view = QtTechniqueView(self._re_client, technique)
            self.technique_views[technique.title] = view
            tabs.addTab(view, technique.title)
        self._available_devices = QtAvailableDevices(
            self._re_client, list(spec.expected_devices)
        )
        tabs.addTab(self._available_devices, "Available Devices")
        self.scan_browser = None
        self.plan_history = None
        if session is not None:
            execution_view = QtPlanExecutionView(session, spec.logger_name)
            tiled_client = _tiled_client(session.tiled)
            if tiled_client is not None:
                self.scan_browser = QtScanBrowser(
                    tiled_client, list(spec.run_specs), session
                )
                execution_view.add_tab(self.scan_browser, "Scan Browser")
            tabs.addTab(execution_view, "Plan Execution")
            self.plan_history = QtPlanHistory(
                session,
                history_file=os.environ.get(spec.plan_history_env) or None,
            )
            tabs.addTab(self.plan_history, "History")
            tabs.addTab(QtPlanLogView(session, spec.logger_name), "Log")
        self.extra_tabs: dict[str, QWidget] = {}
        for title, factory in spec.extra_tabs:
            widget = factory()
            self.extra_tabs[title] = widget
            tabs.addTab(widget, title)
        vbox.addWidget(tabs, stretch=1)

        # Shared live per-device progress bars pinned to the bottom.
        vbox.addWidget(QtReWaitingHookMonitor(re_client))

        # Queue status pinned to the bottom of the window (Queue Server only).
        if isinstance(re_client, RunEngineClient):
            vbox.addWidget(QtRePlanQueue(re_client))

        self.setLayout(vbox)


def _tiled_client(info: dict[str, Any] | None):
    """Open the session's Tiled catalog; None if it has none or it can't be reached."""
    if not info:
        return None
    from tiled.client import from_uri

    try:
        return from_uri(info["uri"], api_key=info.get("api_key"))
    except Exception:
        logger.exception("Can't open the session's Tiled catalog; no Scan Browser")
        return None


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
        re_client: RunEngineClient | SessionClient,
        spec: GuiSpec,
        *,
        show: bool = True,
    ):
        self.spec = spec
        self.qt_widget = QtTabbedTechniqueSelector(re_client, spec)

        self._torn_down = False
        self._qt_window = _MainWindow(on_close=self._teardown)
        self._qt_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._qt_window.setWindowTitle(spec.title)
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
        if isinstance(re_client, SessionClient):
            re_client.state_changed.connect(self._update_re_state)
            re_client.disconnected.connect(self._on_session_lost)
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

        if show:
            self.show()

    def _on_session_lost(self):
        logger.warning("The IPython session ended; closing the GUI")
        self._qt_window.close()

    def _teardown(self):
        """Close every widget, so each stops its own timers, threads and processes."""
        if self._torn_down:
            return
        self._torn_down = True
        # WA_DeleteOnClose deletes the children without sending them closeEvent.
        for widget in self._qt_window.findChildren(QWidget):
            # Closing one widget can delete others (pyqtgraph deletes its viewport).
            if shiboken.isValid(widget):
                widget.close()
        if isinstance(self._re_client, SessionClient):
            self._re_client.close()

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
                "Plan history is only recorded when attached to an IPython session."
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
        for package in self.spec.about_packages:
            try:
                ver = version(package)
            except PackageNotFoundError:
                ver = "not installed"
            rows.append(f"<tr><td><b>{package}</b></td><td>&nbsp;&nbsp;{ver}</td></tr>")
        title = self.spec.title
        box = QMessageBox(
            QMessageBox.Icon.Information,
            f"About {title}",
            f"<h3>{title}</h3><table>{''.join(rows)}</table>",
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
        stamp = ttime.strftime("%Y%m%d_%H%M%S")
        default = Path.home() / f"{self.spec.name.lower()}_daq_screenshot_{stamp}.png"
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

    def _update_re_state(self, *_):
        self._re_state_label.setText(self._re_state_text())

    def _re_state_text(self) -> str:
        if isinstance(self._re_client, SessionClient):
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


def run_client(argv: list[str] | None = None):
    """Run the GUI process that :func:`.session_server.start_gui` starts."""
    parser = argparse.ArgumentParser(description="Data-acquisition GUI process")
    parser.add_argument("--connect", required=True, help="The session's socket")
    parser.add_argument("spec_ref", help='The GuiSpec, as "module:NAME"')
    args = parser.parse_args(argv)
    spec = load_spec(args.spec_ref)
    with gui_qt(spec.title):
        set_settings_scope(*spec.settings_scope)
        apply_bnl_theme(theme=saved_theme())
        session = SessionClient(args.connect)
        session.connect_to_session()
        QtDataAcquisitionWindow(re_client=session, spec=spec)


def run(spec_ref: str, argv: list[str] | None = None):
    """Launch the data-acquisition GUI for the spec named ``"module:NAME"``."""
    spec = load_spec(spec_ref)
    parser = argparse.ArgumentParser(description=spec.title)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--queueserver-uri",
        default=None,
        help="Address of the Bluesky QueueServer http server. If set, connect to"
        " the QueueServer via HTTP and submit plans to its queue.",
    )
    mode.add_argument(
        "--profile",
        default=spec.default_profile,
        help="Profile to start IPython on when no QueueServer is given. The GUI"
        " opens in its own process attached to that session, running plans there"
        " via RE(plan(...)).",
    )
    mock_vars = " ".join(f"{k}={v}" for k, v in spec.mock_environment.items())
    parser.add_argument(
        "--mock",
        action="store_true",
        help=f"Run devices in mock/simulation mode by setting {mock_vars}.",
    )
    parser.add_argument(
        "--history-file",
        default=None,
        help="JSON file to save plan history to as plans finish, and to load it"
        f" from on startup. Can also be set with {spec.plan_history_env}.",
    )

    parser.add_argument(
        "--keep-ipython",
        action="store_true",
        help="Keep the IPython session open when the GUI window is closed"
        " (by default, closing the window also ends it).",
    )

    args = parser.parse_args(argv)

    if args.mock:
        os.environ.update(spec.mock_environment)
    if args.history_file:
        # An env var, so it reaches the GUI process the session starts.
        os.environ[spec.plan_history_env] = args.history_file

    os.environ.update(spec.environment)

    if args.queueserver_uri:
        with gui_qt(spec.title):
            set_settings_scope(*spec.settings_scope)
            apply_bnl_theme(theme=saved_theme())
            re_client = RunEngineClient(http_server_uri=args.queueserver_uri)
            QtDataAcquisitionWindow(re_client=re_client, spec=spec)
    else:
        from IPython import start_ipython
        from traitlets.config import Config

        if spec.profile_dir is None:
            raise ValueError(f"{spec_ref} names no profile_dir to start from")
        profile_path = spec.profile_dir / f"{args.profile}.py"
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")

        # Mirror the environment tweaks from the pixi `start` task.
        for var in ("SESSION_MANAGER", "PYTHONPATH", "PYTHONUSERBASE"):
            os.environ.pop(var, None)
        os.environ["MPLBACKEND"] = "qtagg"

        config = Config()
        # The session keeps Qt for matplotlib and to run the GUI's commands.
        config.InteractiveShellApp.gui = "qt"
        # Run the profile first (defining RE and devices), then open the GUI.
        # exec_lines run in order, before any command-line files.
        config.InteractiveShellApp.exec_lines = [
            f"get_ipython().run_line_magic('run', {f'-i {profile_path}'!r})",
            "from hextools.gui.session_server import start_gui",
            f"start_gui({spec_ref!r}, end_session_on_close={not args.keep_ipython});",
        ]
        config.TerminalIPythonApp.display_banner = False
        start_ipython(argv=[], config=config)
