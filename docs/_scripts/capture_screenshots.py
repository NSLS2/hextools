"""Regenerate the GUI screenshots used in the documentation.

Runs the collection profile in mock mode, opens the GUI offscreen, runs two
small plans so the execution views have content, and saves one PNG per view.
The ``*_annotated`` PNGs have numbered boxes around the components; the
numbers are referenced from ``docs/gui.md``, so keep the two in sync.
"""

import os
import sys
import time
from pathlib import Path

os.environ["HEXTOOLS_RUNNING_IN_CI"] = "YES"
os.environ["BEAMLINE_ACRONYM"] = "HEX"
os.environ["MPLBACKEND"] = "agg"

REPO = Path(__file__).resolve().parents[2]
OUT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else REPO / "docs" / "_static" / "screenshots"
)
OUT.mkdir(parents=True, exist_ok=True)

from IPython.core.interactiveshell import InteractiveShell  # noqa: E402
from qtpy.QtCore import QPoint, QRect, QRectF, Qt  # noqa: E402
from qtpy.QtGui import QColor, QPainter, QPen  # noqa: E402
from qtpy.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTabWidget,
    QToolButton,
    QTreeWidget,
)

app = QApplication.instance() or QApplication([])

# launch_local_viewer() needs an IPython shell with the profile loaded.
shell = InteractiveShell.instance()
shell.run_line_magic(
    "run", f"-i {REPO / 'src' / 'hextools' / 'profiles' / 'collection.py'}"
)

from bluesky import plans as bp  # noqa: E402
from ntnda_qt_viewer import NTNDAViewerWidget  # noqa: E402

from hextools.gui.device_progress_viewer import QtReWaitingHookMonitor  # noqa: E402
from hextools.gui.live_re_md_viewer import QtProposalInfo  # noqa: E402
from hextools.gui.misc import QtTabbedDetectorsWidget, QtWeatherWidget  # noqa: E402
from hextools.gui.plan_status import QtLiveTable, QtPlanStatus  # noqa: E402
from hextools.gui.plan_widget import QtPlanWidget  # noqa: E402
from hextools.gui.re_execution_controls import QtReExecutionControls  # noqa: E402
from hextools.gui.shutter_status import QtShutterStatus  # noqa: E402
from hextools.gui.window import launch_local_viewer  # noqa: E402
from hextools.gui_config import HEX_GUI  # noqa: E402
from hextools.motors import move_motor  # noqa: E402

window = launch_local_viewer(HEX_GUI)
qwin = window._qt_window  # noqa: SLF001
qwin.resize(1700, 1000)
qwin.show()


def pump(seconds: float = 1.0) -> None:
    """Process Qt events for the given number of seconds."""
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


def grab(widget, name: str) -> None:
    """Grab a screenshot of the given widget and save it with the specified name."""
    pump(0.5)
    widget.grab().save(str(OUT / f"{name}.png"))
    print("saved", OUT / f"{name}.png")


CALLOUT = QColor("#ff7a00")
BADGE = 26


def visible(root, cls, test=lambda w: True):
    """Get the first visible widget that passes test."""
    for widget in root.findChildren(cls):
        if widget.isVisible() and test(widget):
            return widget
    raise LookupError(f"No visible {cls.__name__} found in {type(root).__name__}")


def panel(root, cls):
    """Get the titled group box of the ``cls`` panel."""
    widget = visible(root, cls)
    boxes = [box for box in widget.findChildren(QGroupBox) if box.isVisible()]
    return boxes[0] if boxes else widget


def rect_in(widget, root) -> QRect:
    """Get a rectangle the same size of a widget."""
    return QRect(widget.mapTo(root, QPoint(0, 0)), widget.size())


def grab_annotated(root, name: str, targets) -> None:
    """Take a screenshot with some annotations."""
    pump(0.5)
    pixmap = root.grab()
    bounds = root.rect().adjusted(2, 2, -2, -2)
    rects = [
        (target if isinstance(target, QRect) else rect_in(target, root))
        .adjusted(-3, -3, 3, 3)
        .intersected(bounds)
        for target in targets
    ]
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = painter.font()
    font.setBold(True)
    font.setPixelSize(15)
    painter.setFont(font)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(CALLOUT, 3))
    for rect in rects:
        painter.drawRoundedRect(QRectF(rect), 6, 6)
    # Badges after all boxes so no box is drawn over a number.
    for number, rect in enumerate(rects, start=1):
        x = rect.left() - BADGE // 2
        if x < -4:
            # Flush with the window's left edge (menu bar, status bar)
            # A badge there would cover text.
            x = rect.right() - BADGE
        x = min(max(x, 0), root.width() - BADGE)
        y = min(max(rect.top() - BADGE // 2, 0), root.height() - BADGE)
        badge = QRectF(x, y, BADGE, BADGE)
        painter.setPen(QPen(QColor("white"), 2))
        painter.setBrush(CALLOUT)
        painter.drawEllipse(badge)
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, str(number))
    painter.end()
    pixmap.save(str(OUT / f"{name}.png"))
    print("saved", OUT / f"{name}.png")


pump(3)
tabs = window.qt_widget.findChild(QTabWidget, "mainViewerTabs")
index = {tabs.tabText(i): i for i in range(tabs.count())}

for label, name in [
    ("Beamline", "gui_beamline"),
    ("EDXD", "gui_edxd"),
    ("Available Devices", "gui_available_devices"),
]:
    tabs.setCurrentIndex(index[label])
    grab(qwin, name)

tabs.setCurrentIndex(index["Tomography"])
pump(0.5)
tomography = tabs.widget(index["Tomography"])
detectors = visible(tomography, QtTabbedDetectorsWidget)
plan_tabs = visible(
    tomography, QTabWidget, lambda w: isinstance(w.currentWidget(), QtPlanWidget)
)
plan_tabs.setCurrentIndex(
    next(i for i in range(plan_tabs.count()) if plan_tabs.tabText(i) == "Flyscan")
)

grab_annotated(
    qwin,
    "gui_overview_annotated",
    [
        qwin.menuBar(),
        panel(qwin, QtReExecutionControls),
        panel(qwin, QtProposalInfo),
        panel(qwin, QtWeatherWidget),
        panel(qwin, QtPlanStatus),
        panel(qwin, QtShutterStatus),
        tabs.tabBar(),
        detectors,
        plan_tabs,
        visible(qwin, QtReWaitingHookMonitor),
        qwin.statusBar(),
    ],
)

viewer = detectors.currentWidget()
if not isinstance(viewer, NTNDAViewerWidget):
    raise RuntimeError(f"No detector viewer shown; got {viewer!r}")
grab_annotated(
    detectors,
    "gui_detector_viewer_annotated",
    [
        detectors.tabBar(),
        viewer._indicator,  # noqa: SLF001
        visible(viewer, QLineEdit),
        visible(viewer, QPushButton, lambda w: w.text() == "Settings"),
        visible(viewer, QPushButton, lambda w: w.text() == "Start"),
        visible(viewer, QPushButton, lambda w: w.text() == "Stop"),
        viewer._glw,  # noqa: SLF001
        viewer._status_label,  # noqa: SLF001
    ],
)

form = plan_tabs.currentWidget()
grab_annotated(
    plan_tabs,
    "gui_plan_form_annotated",
    [
        plan_tabs.tabBar(),
        visible(form, QComboBox),
        visible(form, QToolButton, lambda w: w.text() == "+"),
        visible(form, QLineEdit, lambda w: w.placeholderText() == "required"),
        visible(form, QLineEdit, lambda w: w.placeholderText() == "optional"),
        visible(form, QCheckBox),
        visible(form, QLabel, lambda w: bool(w.toolTip())),
        visible(form, QPushButton, lambda w: w.text() == "Run Plan"),
    ],
)

ns = shell.user_ns
ns["RE"](move_motor(ns["sim_x"], 1.5))
pump(1)
ns["RE"](bp.scan([], ns["sim_x"], 0, 2, 5))
pump(2)

tabs.setCurrentIndex(index["Plan Execution"])
execution = tabs.widget(index["Plan Execution"])
execution_tabs = visible(execution, QTabWidget)
execution_tabs.setCurrentIndex(0)
pump(0.5)
grab_annotated(
    execution,
    "gui_plan_execution_annotated",
    [
        visible(execution, QGroupBox, lambda w: w.title() == "Current Plan"),
        execution_tabs.tabBar(),
        visible(execution, QtLiveTable),
    ],
)

scan_browser = window.qt_widget.scan_browser
if scan_browser is not None:
    execution_tabs.setCurrentWidget(scan_browser)
    pump(0.5)
    search = visible(scan_browser, QPushButton, lambda w: w.text() == "Search")
    search.click()
    pump(5)
    runs = visible(scan_browser, QTableWidget)
    if runs.rowCount():
        runs.selectRow(0)
        pump(1)
    grab_annotated(
        scan_browser,
        "gui_scan_browser_annotated",
        [
            rect_in(visible(scan_browser, QComboBox), scan_browser).united(
                rect_in(search, scan_browser)
            ),
            runs,
            visible(scan_browser, QTreeWidget),
        ],
    )

tabs.setCurrentIndex(index["Log"])
grab(qwin, "gui_log")

tabs.setCurrentIndex(index["History"])
history = window.qt_widget.plan_history
history_table = visible(history, QTableWidget)
history_table.selectRow(0)
pump(0.5)
row = history_table.visualItemRect(history_table.item(0, 0)).united(
    history_table.visualItemRect(history_table.item(0, history_table.columnCount() - 1))
)
grab_annotated(
    history,
    "gui_history_annotated",
    [
        history_table,
        QRect(history_table.viewport().mapTo(history, row.topLeft()), row.size()),
        visible(history, QPushButton, lambda w: w.text() == "Re-run Selected Plan"),
    ],
)

# Skip interpreter teardown, which can hang on the RunEngine and Tiled threads.
os._exit(0)
