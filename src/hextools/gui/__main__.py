"""HEX queue-monitor GUI.

A recreation of the bluesky-widgets ``queue_monitor`` application, consolidated
into a single entrypoint so it can serve as the main HEX GUI. Run with::

    python -m hextools.gui

Connection defaults to a local RE Manager over 0MQ, or an HTTP server if
``--http-server-uri``/``QSERVER_HTTP_SERVER_URI`` is provided.
"""

from __future__ import annotations

import argparse
import os

from bluesky_widgets.models.run_engine_client import RunEngineClient
from bluesky_widgets.qt import Window, gui_qt
from bluesky_widgets.qt.run_engine_client import (
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
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hextools.gui.live_re_md_viewer import QtReMetadataMonitor
from hextools.gui.misc.weather import QtWeatherWidget
from hextools.gui._theme import apply_bnl_theme

try:  # QAction moved from QtWidgets to QtGui in Qt6
    from qtpy.QtWidgets import QAction
except ImportError:
    from qtpy.QtGui import QAction


def _patch_setchecked_checkstate():
    """Let ``setChecked`` accept ``Qt.CheckState`` enums (bluesky-widgets on PySide6>=6.9)."""
    from qtpy.QtWidgets import QAbstractButton

    original = QAbstractButton.setChecked

    def set_checked(self, value):
        if not isinstance(value, (bool, int)):
            value = getattr(value, "value", value)
        return original(self, bool(value))

    QAbstractButton.setChecked = set_checked


_patch_setchecked_checkstate()


class Settings:
    """Connection settings shared with the RE Manager client."""

    http_server_uri: str | None = None
    http_server_api_key: str | None = None
    zmq_re_manager_control_addr: str | None = None
    zmq_re_manager_info_addr: str | None = None


SETTINGS = Settings()

#: RE metadata keys shown in the monitor view's live metadata panel.
RE_METADATA_KEYS = (
    "scan_id",
    "proposal_id",
    "data_session",
    "sample_name",
    "operator",
)


class QtOrganizeQueueWidgets(QSplitter):
    """Vertically stacked monitor-mode queue, history, and console widgets."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

        self.setOrientation(Qt.Vertical)

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
        hbox.addWidget(QtReMetadataMonitor(model, RE_METADATA_KEYS))
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


class QtViewer(QTabWidget):
    """Tabbed container holding the monitor and editor views."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

        self.setObjectName("mainViewerTabs")
        self.setTabPosition(QTabWidget.West)

        self._re_manager_monitor = QtRunEngineManagerMonitor(model.run_engine)
        self.addTab(self._re_manager_monitor, "Monitor Queue")

        self._re_manager_editor = QtRunEngineManagerEditor(model.run_engine)
        self.addTab(self._re_manager_editor, "Edit and Control Queue")


class ViewerModel:
    """Encapsulates the models used by the application."""

    def __init__(self):
        self.run_engine = RunEngineClient(
            zmq_control_addr=SETTINGS.zmq_re_manager_control_addr,
            zmq_info_addr=SETTINGS.zmq_re_manager_info_addr,
            http_server_uri=SETTINGS.http_server_uri,
            http_server_api_key=SETTINGS.http_server_api_key,
        )


class Viewer(ViewerModel):
    """Model extended with a Qt window, exposed to an interactive console."""

    def __init__(self, *, show=True, title="HEX Queue Monitor"):
        super().__init__()

        self._work_dir = os.path.expanduser("~")

        self._widget = QtViewer(self)
        self._window = Window(self._widget, show=show)

        # bluesky-widgets pads the central widget (4px left/right), which shows
        # as a light border around the window; remove it.
        self._window._qt_center.layout().setContentsMargins(0, 0, 0, 0)

        menu_bar = self._window._qt_window.menuBar()
        menu_item_control = menu_bar.addMenu("Control Actions")
        self.action_activate_env_destroy = QAction(
            "Activate 'Destroy Environment'", self._window._qt_window
        )
        self.action_activate_env_destroy.setCheckable(True)
        self._update_action_env_destroy_state()
        self.action_activate_env_destroy.triggered.connect(
            self._activate_env_destroy_triggered
        )
        menu_item_control.addAction(self.action_activate_env_destroy)

        menu_item_save = menu_bar.addMenu("Save and Backup")
        self.action_save_history_as_txt = QAction(
            "Save Plan History (as .txt)", self._window._qt_window
        )
        self.action_save_history_as_txt.triggered.connect(
            self._save_history_as_txt_triggered
        )
        menu_item_save.addAction(self.action_save_history_as_txt)
        self.action_save_history_as_json = QAction(
            "Save Plan History (as .json)", self._window._qt_window
        )
        self.action_save_history_as_json.triggered.connect(
            self._save_history_as_json_triggered
        )
        menu_item_save.addAction(self.action_save_history_as_json)
        self.action_save_history_as_yaml = QAction(
            "Save Plan History (as .yaml)", self._window._qt_window
        )
        self.action_save_history_as_yaml.triggered.connect(
            self._save_history_as_yaml_triggered
        )
        menu_item_save.addAction(self.action_save_history_as_yaml)

        self._widget.model.run_engine.events.status_changed.connect(
            self.on_update_widgets
        )

    def _update_action_env_destroy_state(self):
        env_destroy_activated = self._widget.model.run_engine.env_destroy_activated
        self.action_activate_env_destroy.setChecked(env_destroy_activated)

    def _activate_env_destroy_triggered(self):
        env_destroy_activated = self._widget.model.run_engine.env_destroy_activated
        self._widget.model.run_engine.activate_env_destroy(not env_destroy_activated)

    def _save_history_as_txt_triggered(self):
        self._save_history_to_file("txt")

    def _save_history_as_json_triggered(self):
        self._save_history_to_file("json")

    def _save_history_as_yaml_triggered(self):
        self._save_history_to_file("yaml")

    def _save_history_to_file(self, file_format):
        try:
            fln_pattern = f"{file_format.upper()} (*.{file_format.lower()});; All (*)"
            file_path_init = os.path.join(
                self._work_dir, "plan_history." + file_format.lower()
            )
            file_path_tuple = QFileDialog.getSaveFileName(
                self._widget, "Save Plan History to File", file_path_init, fln_pattern
            )
            file_path = file_path_tuple[0]
            if file_path:
                self._work_dir = os.path.dirname(file_path)
                self._widget.model.run_engine.save_plan_history_to_file(
                    file_path=file_path, file_format=file_format
                )
                print(f"Plan history was successfully saved to file {file_path!r}")
        except Exception as ex:
            print(f"Failed to save data to file: {ex}")

    def on_update_widgets(self, event):
        self._update_action_env_destroy_state()

    @property
    def window(self):
        return self._window

    def show(self):
        """Resize, show, and raise the window."""
        self._window.show()

    def close(self):
        """Close the window."""
        self._window.close()


def main(argv=None):
    """Launch the HEX queue-monitor GUI."""
    parser = argparse.ArgumentParser(description="HEX Queue Monitor")
    parser.add_argument(
        "--zmq-control-addr",
        default=None,
        help="Address of control socket of RE Manager, e.g. tcp://localhost:60615. "
        "Overrides QSERVER_ZMQ_CONTROL_ADDRESS environment variable.",
    )
    parser.add_argument(
        "--zmq-info-addr",
        default=None,
        help="Address of PUB-SUB socket of RE Manager, e.g. tcp://localhost:60625. "
        "Overrides QSERVER_ZMQ_INFO_ADDRESS environment variable.",
    )
    parser.add_argument(
        "--http-server-uri",
        default=None,
        help="Address of HTTP Server, e.g. http://localhost:60610. Activates "
        "communication with Queue Server via HTTP server. Overrides "
        "QSERVER_HTTP_SERVER_URI environment variable. Use "
        "QSERVER_HTTP_SERVER_API_KEY to pass an API key directly.",
    )
    parser.add_argument(
        "--http-server-keyfile",
        default=None,
        help="Path to read to get the single-user API key. Takes priority over "
        "the QSERVER_HTTP_SERVER_API_KEYFILE env variable.",
    )
    args = parser.parse_args(argv)

    zmq_control_addr = args.zmq_control_addr or os.environ.get(
        "QSERVER_ZMQ_CONTROL_ADDRESS", None
    )
    zmq_info_addr = args.zmq_info_addr or os.environ.get(
        "QSERVER_ZMQ_INFO_ADDRESS", None
    )

    http_server_uri = args.http_server_uri or os.environ.get(
        "QSERVER_HTTP_SERVER_URI", None
    )
    http_server_api_key = os.environ.get("QSERVER_HTTP_SERVER_API_KEY", None)
    http_server_api_path = args.http_server_keyfile or os.environ.get(
        "QSERVER_HTTP_SERVER_API_KEYFILE", None
    )
    if http_server_api_key is None and http_server_api_path is not None:
        with open(http_server_api_path) as fin:
            http_server_api_key = fin.read()

    if http_server_uri:
        print("Initializing: communication with Queue Server via HTTP Server ...")
        SETTINGS.http_server_uri = http_server_uri
        SETTINGS.http_server_api_key = http_server_api_key
        SETTINGS.zmq_re_manager_control_addr = None
        SETTINGS.zmq_re_manager_info_addr = None
    else:
        print("Initializing: communication with Queue Server directly via 0MQ ...")
        SETTINGS.http_server_uri = None
        SETTINGS.http_server_api_key = None
        SETTINGS.zmq_re_manager_control_addr = zmq_control_addr
        SETTINGS.zmq_re_manager_info_addr = zmq_info_addr

    with gui_qt("HEX Queue Monitor"):
        apply_bnl_theme()
        Viewer()


if __name__ == "__main__":
    main()
