"""Titled groups of live signal status rows, packed into one group box."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from typing import cast

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

from hextools.catalog.model import DeviceNode, DeviceRegistry
from hextools.gui.run_engine_clients import RunEngineClient
from hextools.gui.signal_status import QtSignalStatus

# Sections with more fields than this split them over two columns.
_MAX_ROWS_PER_FIELD_COLUMN = 4


def field_label(device_name: str, field: str) -> str:
    """Strip the top-level device name from a field name."""
    if field.startswith(device_name):
        stripped = field[len(device_name) :].lstrip("-_.")
        if stripped:
            return stripped
    return field


def _descendants(path: str, node: DeviceNode) -> Iterator[tuple[str, DeviceNode]]:
    for key, child in node.children.items():
        child_path = node.child_path(path, key)
        yield child_path, child
        yield from _descendants(child_path, child)


def hinted_fields(registry: DeviceRegistry, path: str) -> tuple[str, dict[str, str]]:
    """Return the device's own name, and its hinted fields mapped to their paths.

    Fields are in definition order; a signal maps its own name to ``path``.

    Raises
    ------
    KeyError
        If there is no device or signal at ``path``.
    """
    node = registry.node(path)
    name = node.name or path
    if node.signal:
        return name, {name: path}
    hinted = set(node.hinted_fields)
    fields = {
        child.name: child_path
        for child_path, child in _descendants(path, node)
        if child.signal and child.name is not None and child.name in hinted
    }
    return name, fields


class QtStatusGroup(QWidget):
    """Titled section of live :class:`QtSignalStatus` rows.

    Long sections split their fields over two columns, filled top to bottom.

    Parameters
    ----------
    title : str
        Section header.
    fields : Mapping[str, str]
        Row label to the name, in the client's device registry, of the signal or
        device shown on that row, in order.
    re_client : RunEngineClient
        Where the readings come from.
    precision : int, optional
        Decimal places shown for float values.
    """

    def __init__(
        self,
        title: str,
        fields: Mapping[str, str],
        re_client: RunEngineClient,
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
        for i, (label, name) in enumerate(fields.items()):
            status = QtSignalStatus(re_client, name, label=label, precision=precision)
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
    name : str
        The device, as named in the client's device registry.
    re_client : RunEngineClient
        Where the device's hinted fields are described and read from.
    display_name : str, optional
        Section header. Defaults to the device's ``name``.
    precision : int, optional
        Decimal places shown for float values.
    """

    def __init__(
        self,
        name: str,
        re_client: RunEngineClient,
        display_name: str | None = None,
        precision: int = 3,
        parent=None,
    ):
        device_name, hinted = hinted_fields(re_client.device_registry, name)
        fields = {
            field_label(device_name, field): path for field, path in hinted.items()
        }
        super().__init__(
            display_name if display_name is not None else device_name,
            fields,
            re_client,
            precision=precision,
            parent=parent,
        )


class QtBeamlineStatus(QWidget):
    """Device status sections, packed into balanced columns.

    Each section goes into the currently shortest column, so short and tall
    sections interleave instead of leaving gaps.

    Parameters
    ----------
    devices : Iterable[str | QtStatusGroup] or Mapping[str, str]
        Device names in the client's device registry, or ready-made status
        sections. A mapping gives each device section a display name.
    re_client : RunEngineClient
        Where the devices are described and read from.
    columns : int, optional
        Number of section columns.
    precision : int, optional
        Decimal places shown for float values.
    title : str or None, optional
        Title of a group box around the sections; None for no group box.
    """

    def __init__(
        self,
        devices: Iterable[str | QtStatusGroup] | Mapping[str, str],
        re_client: RunEngineClient,
        columns: int = 4,
        precision: int = 3,
        title: str | None = "Beamline Status",
        parent=None,
    ):
        super().__init__(parent)
        entries: list[tuple[str | QtStatusGroup, str | None]]
        if isinstance(devices, Mapping):
            mapping = cast(Mapping[str, str], devices)
            entries = [(device, name) for name, device in mapping.items()]
        else:
            entries = [(device, None) for device in devices]

        column_layouts = [QVBoxLayout() for _ in range(columns)]
        heights = [0] * columns
        self._groups: list[QtStatusGroup] = []
        for device, display_name in entries:
            status = (
                device
                if isinstance(device, QtStatusGroup)
                else QtDeviceStatus(
                    device, re_client, display_name=display_name, precision=precision
                )
            )
            col = heights.index(min(heights))
            column_layouts[col].addWidget(status)
            # Header and inter-section gap count as about two rows.
            heights[col] += status.rows + 2
            self._groups.append(status)

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
        for status in self._groups:
            status.close()
        super().closeEvent(event)
