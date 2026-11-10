"""Compact live status sections for devices, packed into one group box."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable, Mapping
from typing import Any

from bluesky.protocols import Subscribable
from ophyd_async.core import Device, SignalR, walk_devices
from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from hextools.gui.signal_status import QtSignalStatus
from hextools.utils import get_obj_from_ipython_ns

# Sections with more fields than this split them over two columns.
_MAX_ROWS_PER_FIELD_COLUMN = 4


def hinted_signals(device: Device) -> dict[str, SignalR[Any]]:
    """Map each hinted field of ``device`` to its signal, in definition order."""
    if isinstance(device, SignalR):
        return {device.name: device}
    fields = set(getattr(device, "hints", {}).get("fields", []))
    # Hints are not reliably ordered, so follow the device tree instead.
    return {
        child.name: child
        for child in walk_devices(device).values()
        if isinstance(child, SignalR) and child.name in fields
    }


def field_label(device_name: str, field: str) -> str:
    """Strip the top-level device name from a field name."""
    if field.startswith(device_name):
        stripped = field[len(device_name) :].lstrip("-_.")
        if stripped:
            return stripped
    return field


class QtStatusGroup(QWidget):
    """Titled section of live :class:`QtSignalStatus` rows.

    Long sections split their fields over two columns, filled top to bottom.

    Parameters
    ----------
    title : str
        Section header.
    fields : Mapping[str, Subscribable]
        Row label to the signal or device shown on that row, in order.
    loop : asyncio.AbstractEventLoop
        The bluesky event loop (e.g. ``RE.loop``) that the signals live on.
    precision : int, optional
        Decimal places shown for float values.
    """

    def __init__(
        self,
        title: str,
        fields: Mapping[str, Subscribable[Any]],
        loop: asyncio.AbstractEventLoop,
        precision: int = 3,
        parent=None,
    ):
        super().__init__(parent)
        self._title = title
        header = QLabel(title)
        header.setObjectName("statusSectionHeader")

        field_columns = 2 if len(fields) > _MAX_ROWS_PER_FIELD_COLUMN else 1
        self.rows = max(1, math.ceil(len(fields) / field_columns))
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(0)
        self._statuses: list[QtSignalStatus] = []
        for i, (label, signal) in enumerate(fields.items()):
            status = QtSignalStatus(signal, loop, label=label, precision=precision)
            status.label_widget.setObjectName("statusFieldLabel")
            status.label_widget.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            grid.addWidget(status, i % self.rows, i // self.rows)
            self._statuses.append(status)
        if not self._statuses:
            grid.addWidget(QLabel("No fields"), 0, 0)
        grid.setColumnStretch(field_columns, 1)
        # Align each field column's values.
        for col in range(field_columns):
            column = self._statuses[col * self.rows : (col + 1) * self.rows]
            width = max((s.label_widget.sizeHint().width() for s in column), default=0)
            for status in column:
                status.label_widget.setFixedWidth(width)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(2)
        vbox.addWidget(header)
        vbox.addLayout(grid)
        self.setLayout(vbox)

    @property
    def title(self) -> str:
        """The section header text."""
        return self._title

    def closeEvent(self, event):
        for status in self._statuses:
            status.close()
        super().closeEvent(event)


class QtDeviceStatus(QtStatusGroup):
    """Section of live :class:`QtSignalStatus` rows for a device's hinted fields.

    Parameters
    ----------
    device : Device or str
        The device, or its variable name in the IPython namespace.
    loop : asyncio.AbstractEventLoop
        The bluesky event loop (e.g. ``RE.loop``) that the device lives on.
    display_name : str, optional
        Section header. Defaults to the device's ``name``.
    precision : int, optional
        Decimal places shown for float values.
    """

    def __init__(
        self,
        device: Device | str,
        loop: asyncio.AbstractEventLoop,
        display_name: str | None = None,
        precision: int = 3,
        parent=None,
    ):
        if isinstance(device, str):
            resolved = get_obj_from_ipython_ns(device, Device)
            if resolved is None:
                raise ValueError(f"No device named {device!r} in the IPython namespace")
            device = resolved
        self._device = device
        fields = {
            field_label(device.name, field): signal
            for field, signal in hinted_signals(device).items()
        }
        super().__init__(
            display_name if display_name is not None else device.name,
            fields,
            loop,
            precision=precision,
            parent=parent,
        )


class QtBeamlineStatus(QWidget):
    """Device status sections, packed into balanced columns.

    Each section goes into the currently shortest column, so short and tall
    sections interleave instead of leaving gaps.

    Parameters
    ----------
    devices : Iterable[Device | str | QtStatusGroup] or Mapping[str, Device | str]
        The devices, their IPython namespace variable names, or ready-made
        status sections. A mapping gives each device section a display name.
    loop : asyncio.AbstractEventLoop
        The bluesky event loop (e.g. ``RE.loop``) that the devices live on.
    columns : int, optional
        Number of section columns.
    precision : int, optional
        Decimal places shown for float values.
    title : str or None, optional
        Title of a group box around the sections; None for no group box.
    """

    def __init__(
        self,
        devices: Iterable[Device | str | QtStatusGroup] | Mapping[str, Device | str],
        loop: asyncio.AbstractEventLoop,
        columns: int = 4,
        precision: int = 3,
        title: str | None = "Beamline Status",
        parent=None,
    ):
        super().__init__(parent)
        if isinstance(devices, Mapping):
            entries = [(device, name) for name, device in devices.items()]
        else:
            entries = [(device, None) for device in devices]

        column_layouts = [QVBoxLayout() for _ in range(columns)]
        heights = [0] * columns
        self._device_statuses: list[QtStatusGroup] = []
        for device, display_name in entries:
            status = (
                device
                if isinstance(device, QtStatusGroup)
                else QtDeviceStatus(
                    device, loop, display_name=display_name, precision=precision
                )
            )
            col = heights.index(min(heights))
            column_layouts[col].addWidget(status)
            # Header and inter-section gap count as about two rows.
            heights[col] += status.rows + 2
            self._device_statuses.append(status)

        row = QHBoxLayout()
        row.setSpacing(20)
        for column in column_layouts:
            column.setSpacing(8)
            column.addStretch(1)
            row.addLayout(column, stretch=1)

        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(8, 6, 8, 6)
        if title is None:
            content_layout.addLayout(row)
        else:
            group_box = QGroupBox(title)
            group_box.setLayout(row)
            content_layout.addWidget(group_box)
        content_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(self._content)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(scroll)
        self.setLayout(vbox)

    def sizeHint(self) -> QSize:
        # A scroll area's own hint ignores its content; fit the content instead.
        return self._content.sizeHint() + QSize(4, 4)

    def closeEvent(self, event):
        for status in self._device_statuses:
            status.close()
        super().closeEvent(event)
