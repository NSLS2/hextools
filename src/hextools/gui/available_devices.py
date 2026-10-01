"""Qt widget listing expected beamline devices and their connection status.

In Queue Server mode a device counts as connected if the server reports it among
its allowed devices or plans. In local (in-process) mode the device must exist in
the IPython namespace, be of the expected type, and have actually connected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import IPython
from bluesky import RunEngine
from bluesky_widgets.models.run_engine_client import RunEngineClient
from qtpy.QtCore import QTimer, Signal, Slot
from qtpy.QtWidgets import (
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

try:
    from ophyd_async.core import Device as _AsyncDevice
except ImportError:  # pragma: no cover
    _AsyncDevice = None  # type: ignore[assignment,misc]

_CONNECTED_STYLE = "color: #2e7d32; font-weight: bold;"
_DISCONNECTED_STYLE = "color: #c62828; font-weight: bold;"
_LOCAL_POLL_MS = 2000


def _is_connected(obj: Any) -> bool:
    """Return True only if ``obj`` has completed a successful connection."""
    if _AsyncDevice is not None and isinstance(obj, _AsyncDevice):
        if getattr(obj, "_mock", None) is not None:
            return True
        task = getattr(obj, "_connect_task", None)
        return bool(task is not None and task.done() and not task.cancelled() and task.exception() is None)
    # Classic ophyd objects expose a ``connected`` property.
    return bool(getattr(obj, "connected", False))


class QtAvailableDevices(QWidget):
    """Table of device name, type, and connection status.

    Parameters
    ----------
    re_client : RunEngineClient or RunEngine
        Queue Server client model, or the in-process RunEngine.
    devices : Sequence[tuple[str, type]]
        ``(name, type)`` pairs of the devices expected to be available.
    namespace : Mapping[str, object], optional
        Namespace searched in local mode. Defaults to the IPython ``user_ns``.
    """

    _signal_refresh = Signal()

    def __init__(
        self,
        re_client: RunEngineClient | RunEngine,
        devices: Sequence[tuple[str, type]],
        *,
        namespace: Mapping[str, Any] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._re_client = re_client
        self._devices = list(devices)
        self._is_qserver = isinstance(re_client, RunEngineClient)
        if namespace is None and not self._is_qserver:
            ip = IPython.get_ipython()
            namespace = ip.user_ns if ip is not None else {}
        self._namespace = namespace or {}
        self._status_labels: dict[str, QLabel] = {}

        grid = QGridLayout()
        for col, header in enumerate(("Name", "Type", "Status")):
            label = QLabel(f"<b>{header}</b>")
            grid.addWidget(label, 0, col)
        for row, (name, dtype) in enumerate(self._devices, start=1):
            grid.addWidget(QLabel(name), row, 0)
            grid.addWidget(QLabel(dtype.__name__), row, 1)
            status = QLabel()
            self._status_labels[name] = status
            grid.addWidget(status, row, 2)
        grid.setColumnStretch(3, 1)
        grid.setRowStretch(len(self._devices) + 1, 1)

        content = QWidget()
        content.setLayout(grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        vbox = QVBoxLayout()
        vbox.addWidget(scroll)
        self.setLayout(vbox)

        self._signal_refresh.connect(self._refresh)
        if self._is_qserver:
            events = self._re_client.events  # type: ignore[union-attr]
            events.allowed_devices_changed.connect(self._on_model_changed)
            events.allowed_plans_changed.connect(self._on_model_changed)
        else:
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._refresh)
            self._timer.start(_LOCAL_POLL_MS)

        self._refresh()

    def _on_model_changed(self, event=None):
        # Model events fire from a polling thread; marshal onto the GUI thread.
        self._signal_refresh.emit()

    def _local_device(self, name: str, dtype: type) -> Any | None:
        obj = self._namespace.get(name)
        return obj if isinstance(obj, dtype) else None

    def _check(self, name: str, dtype: type) -> bool:
        if self._is_qserver:
            model = self._re_client
            allowed_devices = getattr(model, "_allowed_devices", {}) or {}  # noqa: SLF001
            allowed_plans = getattr(model, "_allowed_plans", {}) or {}  # noqa: SLF001
            info = allowed_devices.get(name)
            if isinstance(info, dict):
                return info.get("classname") in (None, dtype.__name__)
            return name in allowed_plans
        obj = self._local_device(name, dtype)
        return obj is not None and _is_connected(obj)

    @Slot()
    def _refresh(self):
        for name, dtype in self._devices:
            connected = self._check(name, dtype)
            label = self._status_labels[name]
            label.setText("Connected" if connected else "Disconnected")
            label.setStyleSheet(_CONNECTED_STYLE if connected else _DISCONNECTED_STYLE)

    def closeEvent(self, event):
        if self._is_qserver:
            events = self._re_client.events  # type: ignore[union-attr]
            for emitter in (events.allowed_devices_changed, events.allowed_plans_changed):
                try:
                    emitter.disconnect(self._on_model_changed)
                except (ValueError, TypeError, RuntimeError):
                    pass
        super().closeEvent(event)
