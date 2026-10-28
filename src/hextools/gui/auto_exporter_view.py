"""Run the Tiled auto exporter in a subprocess and show its log."""

from __future__ import annotations

import html
import logging
import re
import sys
import time

from qtpy.QtCore import QProcess, QProcessEnvironment, QTimer, Slot
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hextools.gui.plan_status import _monospace_font, _record_html

# Matches the format hextools.log.configure_logger prints.
_RECORD_RE = re.compile(
    r"^\[\d{2}:\d{2}:\d{2}\] (?P<level>[A-Z]+) (?P<name>[\w.]+): (?P<message>.*)$"
)


class QtAutoExporterView(QWidget):
    """Start/stop the auto exporter and show everything it logs.

    The exporter (``python -m <module>``) runs in its own process, so a crash or
    traceback in an export job can't take down the GUI or the RunEngine. If it
    exits unexpectedly it is restarted after ``restart_delay`` seconds.
    """

    _MAX_LINES = 20000

    def __init__(
        self,
        module: str,
        autostart: bool = True,
        restart_delay: float = 30.0,
        parent=None,
    ):
        super().__init__(parent)
        self._module = module
        self._stopping = False
        self._buffer = ""

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self._process.setProcessEnvironment(env)
        self._process.readyReadStandardOutput.connect(self._on_output)
        self._process.started.connect(self._on_started)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)

        self._restart_timer = QTimer(self)
        self._restart_timer.setSingleShot(True)
        self._restart_timer.setInterval(int(restart_delay * 1000))
        self._restart_timer.timeout.connect(self.start)

        self._status = QLabel("Stopped")
        self._start_button = QPushButton("Start")
        self._start_button.clicked.connect(self.start)
        self._stop_button = QPushButton("Stop")
        self._stop_button.clicked.connect(self.stop)
        self._stop_button.setEnabled(False)
        self._debug = QCheckBox("Debug logging")
        self._debug.setToolTip("Takes effect the next time the exporter starts.")

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Auto exporter:"))
        controls.addWidget(self._status)
        controls.addStretch()
        controls.addWidget(self._debug)
        controls.addWidget(self._start_button)
        controls.addWidget(self._stop_button)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(_monospace_font())
        self._log.setMaximumBlockCount(self._MAX_LINES)

        vbox = QVBoxLayout()
        vbox.addLayout(controls)
        vbox.addWidget(self._log, stretch=1)
        self.setLayout(vbox)

        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.stop)

        if autostart:
            QTimer.singleShot(0, self.start)

    @Slot()
    def start(self) -> None:
        """Start the exporter process if it isn't already running."""
        self._restart_timer.stop()
        if self._process.state() != QProcess.ProcessState.NotRunning:
            return
        self._stopping = False
        args = ["-m", self._module]
        if self._debug.isChecked():
            args.append("--debug")
        self._append_note(f"Starting auto exporter: {sys.executable} {' '.join(args)}")
        self._status.setText("Starting…")
        self._start_button.setEnabled(False)
        self._process.start(sys.executable, args)

    @Slot()
    def stop(self) -> None:
        """Stop the exporter process and don't restart it."""
        self._stopping = True
        self._restart_timer.stop()
        if self._process.state() == QProcess.ProcessState.NotRunning:
            self._set_stopped("Stopped")
            return
        self._append_note("Stopping auto exporter")
        self._process.terminate()
        if not self._process.waitForFinished(5000):
            self._process.kill()
            self._process.waitForFinished(1000)

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)

    @Slot()
    def _on_started(self) -> None:
        self._status.setText(f"Running (pid {self._process.processId()})")
        self._stop_button.setEnabled(True)

    @Slot(int, QProcess.ExitStatus)
    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self._flush_buffer()
        if self._stopping:
            self._append_note("Auto exporter stopped")
            self._set_stopped("Stopped")
            return
        how = (
            "crashed"
            if exit_status == QProcess.ExitStatus.CrashExit
            else f"exited with code {exit_code}"
        )
        delay = self._restart_timer.interval() // 1000
        self._append_note(
            f"Auto exporter {how}; restarting in {delay} s", logging.ERROR
        )
        self._set_stopped(f"Exited ({how}); restarting in {delay} s")
        self._start_button.setEnabled(True)
        self._restart_timer.start()

    @Slot(QProcess.ProcessError)
    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._append_note(
                f"Auto exporter failed to start: {self._process.errorString()}",
                logging.ERROR,
            )
            self._set_stopped("Failed to start")

    @Slot()
    def _on_output(self) -> None:
        data = bytes(self._process.readAllStandardOutput().data())
        self._buffer += data.decode(errors="replace")
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            self._append_line(line.rstrip("\r"))

    def _flush_buffer(self) -> None:
        if self._buffer:
            self._append_line(self._buffer)
            self._buffer = ""

    def _set_stopped(self, status: str) -> None:
        self._status.setText(status)
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)

    def _append_line(self, line: str) -> None:
        match = _RECORD_RE.match(line)
        if match is None:
            # Tracebacks and output not from a hextools logger
            text = html.escape(line).replace(" ", "&nbsp;")
        else:
            levelname = match["level"]
            levelno = logging.getLevelName(levelname)
            if not isinstance(levelno, int):
                levelno = logging.INFO
            text = _record_html(
                time.time(), levelno, levelname, match["message"], "%Y-%m-%d %H:%M:%S"
            )
        self._append_html(text)

    def _append_note(self, message: str, levelno: int = logging.INFO) -> None:
        self._append_html(
            _record_html(
                time.time(),
                levelno,
                logging.getLevelName(levelno),
                message,
                "%Y-%m-%d %H:%M:%S",
            )
        )

    def _append_html(self, text: str) -> None:
        scrollbar = self._log.verticalScrollBar()
        at_bottom = scrollbar.value() == scrollbar.maximum()
        self._log.appendHtml(text)
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())
