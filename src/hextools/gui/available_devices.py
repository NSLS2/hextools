"""Qt widget listing beamline devices and their connection status.

In IPython-session mode every ``ophyd.Device`` or ``ophyd_async`` ``Device``
in the session's namespace is listed, and counts as connected only once it has
actually connected; the session reports both. In Queue Server mode a fixed list
of expected devices is shown, each connected if the server reports it among its
allowed devices or plans.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from bluesky_widgets.models.run_engine_client import RunEngineClient
from qtpy.QtCore import QTimer, Signal, Slot
from qtpy.QtWidgets import (
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

_DEVICE_BASES: list[type] = []
try:
    from ophyd_async.core import Device as _AsyncDevice

    _DEVICE_BASES.append(_AsyncDevice)
except ImportError:  # pragma: no cover
    _AsyncDevice = None  # type: ignore
try:
    from ophyd import Device as _OphydDevice

    _DEVICE_BASES.append(_OphydDevice)
except ImportError:  # pragma: no cover
    pass

_CONNECTED_STYLE = "color: #2e7d32; font-weight: bold;"
_DISCONNECTED_STYLE = "color: #c62828; font-weight: bold;"
_SESSION_POLL_MS = 2000


def _is_connected(obj: Any) -> bool:
    """Return True only if ``obj`` has completed a successful connection."""
    if _AsyncDevice is not None and isinstance(obj, _AsyncDevice):
        if getattr(obj, "_mock", None) is not None:
            return True
        task = getattr(obj, "_connect_task", None)
        return bool(
            task is not None
            and task.done()
            and not task.cancelled()
            and task.exception() is None
        )
    # Classic ophyd objects expose a ``connected`` property.
    return bool(getattr(obj, "connected", False))


def _discover_devices(namespace: Mapping[str, Any]) -> list[tuple[str, str, Any]]:
    """Return ``(label, type name, device)`` for each distinct device in ``namespace``.

    Aliases of one device share a row, labelled by the alias matching the device's
    own ``name`` (else the first alphabetically) with the others in parentheses.
    """
    aliases: dict[int, list[str]] = {}
    devices: dict[int, Any] = {}
    for name, obj in sorted(namespace.items()):
        if name.startswith("_") or not isinstance(obj, tuple(_DEVICE_BASES)):
            continue
        aliases.setdefault(id(obj), []).append(name)
        devices[id(obj)] = obj
    entries = []
    for key, names in aliases.items():
        obj = devices[key]
        primary = next((n for n in names if n == getattr(obj, "name", None)), names[0])
        others = [n for n in names if n != primary]
        label = f"{primary} ({', '.join(others)})" if others else primary
        entries.append((label, type(obj).__name__, obj))
    return sorted(entries, key=lambda entry: entry[0])


class QtAvailableDevices(QWidget):
    """Table of device name, type, and connection status.

    Parameters
    ----------
    re_client : RunEngineClient or SessionClient
        Queue Server client model, or the IPython session.
    devices : Sequence[tuple[str, type]], optional
        ``(name, type)`` pairs of the devices expected to be available. Only used
        in Queue Server mode; the session reports the devices it has.
    """

    _signal_refresh = Signal()

    def __init__(
        self,
        re_client: Any,
        devices: Sequence[tuple[str, type]] = (),
        *,
        parent=None,
    ):
        super().__init__(parent)
        self._re_client = re_client
        self._devices = list(devices)
        self._is_qserver = isinstance(re_client, RunEngineClient)
        self._rows: list[tuple[str, str]] = []
        self._status_labels: dict[str, QLabel] = {}

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)

        vbox = QVBoxLayout()
        vbox.addWidget(self._scroll)
        self.setLayout(vbox)

        self._signal_refresh.connect(self._refresh)
        if self._is_qserver:
            events = self._re_client.events  # type: ignore
            events.allowed_devices_changed.connect(self._on_model_changed)  # type: ignore
            events.allowed_plans_changed.connect(self._on_model_changed)  # type: ignore
        else:
            # Polling also picks up devices connecting, and ones created or
            # deleted at the IPython prompt.
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._refresh)
            self._timer.start(_SESSION_POLL_MS)

        self._refresh()

    def _on_model_changed(self, event=None):
        # Model events fire from a polling thread; marshal onto the GUI thread.
        self._signal_refresh.emit()

    def _qserver_connected(self, name: str, dtype: type) -> bool:
        model = self._re_client
        allowed_devices = getattr(model, "_allowed_devices", {}) or {}  # noqa: SLF001
        allowed_plans = getattr(model, "_allowed_plans", {}) or {}  # noqa: SLF001
        info = allowed_devices.get(name)
        if isinstance(info, dict):
            return info.get("classname") in (None, dtype.__name__)
        return name in allowed_plans

    def _rebuild(self, rows: list[tuple[str, str]]):
        grid = QGridLayout()
        for col, header in enumerate(("Name", "Type", "Status")):
            grid.addWidget(QLabel(f"<b>{header}</b>"), 0, col)
        self._status_labels = {}
        for row, (label, type_name) in enumerate(rows, start=1):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(QLabel(type_name), row, 1)
            status = QLabel()
            self._status_labels[label] = status
            grid.addWidget(status, row, 2)
        grid.setColumnStretch(3, 1)
        grid.setRowStretch(len(rows) + 1, 1)
        content = QWidget()
        content.setLayout(grid)
        self._scroll.setWidget(content)  # deletes the previous content widget
        self._rows = rows

    @Slot()
    def _refresh(self):
        if self._is_qserver:
            self._show(
                [
                    (name, dtype.__name__, self._qserver_connected(name, dtype))
                    for name, dtype in self._devices
                ]
            )
        else:
            self._re_client.device_status(self._show)

    def _show(self, entries: Sequence[Sequence[Any]]):
        """Show ``(label, type name, connected)`` rows."""
        rows = [(str(label), str(type_name)) for label, type_name, _ in entries]
        if rows != self._rows:
            self._rebuild(rows)
        for label, _, connected in entries:
            status = self._status_labels[str(label)]
            status.setText("Connected" if connected else "Disconnected")
            status.setStyleSheet(_CONNECTED_STYLE if connected else _DISCONNECTED_STYLE)

    def closeEvent(self, event):
        if self._is_qserver:
            events = self._re_client.events  # type: ignore
            for emitter in (
                events.allowed_devices_changed,  # type: ignore
                events.allowed_plans_changed,  # type: ignore
            ):
                try:
                    emitter.disconnect(self._on_model_changed)
                except (ValueError, TypeError, RuntimeError):
                    pass
        super().closeEvent(event)
