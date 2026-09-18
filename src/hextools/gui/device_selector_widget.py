"""Qt widgets for selecting Queue Server devices by type.

The widgets discover the devices currently allowed by the Queue Server (via the
``RunEngineClient`` model) and present the subset matching a requested device
type. :class:`QtDeviceSelector` offers a single-selection dropdown;
:class:`QtMultiDeviceSelector` offers a checkable list for selecting one or any
number of devices.

Device "type" is expressed with :class:`DeviceType`, which maps onto the
protocol flags reported by the Queue Server (``is_readable``, ``is_movable``,
``is_flyable``). Selection can optionally be narrowed further to specific ophyd
class names.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import Enum

from qtpy.QtCore import Qt, Signal, Slot
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

_NO_SELECTION = "\u2014"  # em dash


class DeviceType(str, Enum):
    """Device categories that can be filtered on, matching Queue Server flags."""

    ANY = "any"
    READABLE = "readable"
    MOVABLE = "movable"
    FLYABLE = "flyable"


_TYPE_FLAG = {
    DeviceType.READABLE: "is_readable",
    DeviceType.MOVABLE: "is_movable",
    DeviceType.FLYABLE: "is_flyable",
}


def _matches(info: dict, device_type: DeviceType, classnames: frozenset[str] | None) -> bool:
    if classnames is not None and info.get("classname") not in classnames:
        return False
    if device_type is DeviceType.ANY:
        return True
    return bool(info.get(_TYPE_FLAG[device_type], False))


def _iter_matching_devices(
    devices: dict,
    device_type: DeviceType,
    classnames: frozenset[str] | None,
    *,
    include_components: bool,
    _prefix: str = "",
) -> Iterable[str]:
    """Yield the dotted names of devices (and optionally components) that match."""
    for name, info in devices.items():
        if not isinstance(info, dict):
            continue
        full_name = f"{_prefix}{name}"
        if _matches(info, device_type, classnames):
            yield full_name
        if include_components and info.get("components"):
            yield from _iter_matching_devices(
                info["components"],
                device_type,
                classnames,
                include_components=include_components,
                _prefix=f"{full_name}.",
            )


class _DeviceSelectorBase(QWidget):
    """Shared discovery/refresh logic for the device selector widgets."""

    signal_devices_changed = Signal()

    def __init__(
        self,
        model,
        device_type: DeviceType | str = DeviceType.ANY,
        parent=None,
        *,
        classnames: Sequence[str] | None = None,
        include_components: bool = False,
    ):
        super().__init__(parent)
        self.model = model
        self._device_type = DeviceType(device_type)
        self._classnames = frozenset(classnames) if classnames is not None else None
        self._include_components = include_components
        self._device_names: list[str] = []

        self.signal_devices_changed.connect(self._refresh)
        self.model.events.allowed_devices_changed.connect(self._on_allowed_devices_changed)

        self._refresh()

    def _on_allowed_devices_changed(self, event=None):
        # Fired from the model's polling thread; marshal onto the GUI thread.
        self.signal_devices_changed.emit()

    def _available_devices(self) -> list[str]:
        allowed = getattr(self.model, "_allowed_devices", {}) or {}  # noqa: SLF001
        names = _iter_matching_devices(
            allowed,
            self._device_type,
            self._classnames,
            include_components=self._include_components,
        )
        return sorted(names)

    @Slot()
    def _refresh(self):
        names = self._available_devices()
        if names == self._device_names:
            return
        self._device_names = names
        self._populate(names)

    def _populate(self, names: list[str]):
        raise NotImplementedError

    @property
    def device_names(self) -> list[str]:
        """Names of the devices currently offered for selection."""
        return list(self._device_names)

    def closeEvent(self, event):
        try:
            self.model.events.allowed_devices_changed.disconnect(self._on_allowed_devices_changed)
        except (ValueError, TypeError, RuntimeError):
            pass
        super().closeEvent(event)


class QtDeviceSelector(_DeviceSelectorBase):
    """Dropdown for selecting a single device of a given type.

    Parameters
    ----------
    model : bluesky_widgets.models.run_engine_client.RunEngineClient
        The Queue Server client model (e.g. ``viewer.run_engine``).
    device_type : DeviceType or str, optional
        Which category of devices to offer. Default: :attr:`DeviceType.ANY`.
    parent : QWidget, optional
        Parent widget.
    classnames : Sequence[str], optional
        If given, only devices whose ophyd class name is in this set are offered.
    include_components : bool, optional
        If True, matching sub-components are offered using dotted names.
    title : str, optional
        Group box title. Default: derived from ``device_type``.
    allow_empty : bool, optional
        If True (default), include a placeholder entry representing no selection.
    """

    signal_selection_changed = Signal(object)

    def __init__(
        self,
        model,
        device_type: DeviceType | str = DeviceType.ANY,
        parent=None,
        *,
        classnames: Sequence[str] | None = None,
        include_components: bool = False,
        title: str | None = None,
        allow_empty: bool = True,
    ):
        self._allow_empty = allow_empty
        self._combo = QComboBox()
        super().__init__(
            model,
            device_type,
            parent,
            classnames=classnames,
            include_components=include_components,
        )

        if title is None:
            title = f"{self._device_type.value.capitalize()} device"

        self._combo.currentIndexChanged.connect(self._on_index_changed)

        group_box = QGroupBox(title)
        inner = QVBoxLayout()
        inner.setContentsMargins(8, 6, 8, 6)
        inner.addWidget(self._combo)
        group_box.setLayout(inner)

        vbox = QVBoxLayout()
        vbox.addWidget(group_box)
        self.setLayout(vbox)

    def _populate(self, names: list[str]):
        previous = self.selected_device
        self._combo.blockSignals(True)
        self._combo.clear()
        if self._allow_empty:
            self._combo.addItem(_NO_SELECTION, userData=None)
        for name in names:
            self._combo.addItem(name, userData=name)
        if previous is not None:
            index = self._combo.findData(previous)
            if index >= 0:
                self._combo.setCurrentIndex(index)
        self._combo.blockSignals(False)
        self._on_index_changed()

    def _on_index_changed(self, *_):
        self.signal_selection_changed.emit(self.selected_device)

    @property
    def selected_device(self) -> str | None:
        """The currently selected device name, or None."""
        return self._combo.currentData()

    def set_selected_device(self, name: str | None):
        """Select ``name`` if it is available; otherwise clear the selection."""
        index = self._combo.findData(name) if name is not None else -1
        if index < 0 and self._allow_empty:
            index = 0
        if index >= 0:
            self._combo.setCurrentIndex(index)


class QtMultiDeviceSelector(_DeviceSelectorBase):
    """Checkable list for selecting one or any number of devices of a type.

    Parameters
    ----------
    model : bluesky_widgets.models.run_engine_client.RunEngineClient
        The Queue Server client model (e.g. ``viewer.run_engine``).
    device_type : DeviceType or str, optional
        Which category of devices to offer. Default: :attr:`DeviceType.ANY`.
    parent : QWidget, optional
        Parent widget.
    classnames : Sequence[str], optional
        If given, only devices whose ophyd class name is in this set are offered.
    include_components : bool, optional
        If True, matching sub-components are offered using dotted names.
    title : str, optional
        Group box title. Default: derived from ``device_type``.
    """

    signal_selection_changed = Signal(object)

    def __init__(
        self,
        model,
        device_type: DeviceType | str = DeviceType.ANY,
        parent=None,
        *,
        classnames: Sequence[str] | None = None,
        include_components: bool = False,
        title: str | None = None,
    ):
        self._list = QListWidget()
        super().__init__(
            model,
            device_type,
            parent,
            classnames=classnames,
            include_components=include_components,
        )

        if title is None:
            title = f"{self._device_type.value.capitalize()} devices"

        self._list.itemChanged.connect(self._on_item_changed)

        group_box = QGroupBox(title)
        inner = QVBoxLayout()
        inner.setContentsMargins(8, 6, 8, 6)
        inner.addWidget(self._list)
        group_box.setLayout(inner)

        vbox = QVBoxLayout()
        vbox.addWidget(group_box)
        self.setLayout(vbox)

    def _populate(self, names: list[str]):
        previous = set(self.selected_devices)
        self._list.blockSignals(True)
        self._list.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in previous else Qt.Unchecked)
            self._list.addItem(item)
        self._list.blockSignals(False)
        self._on_item_changed()

    def _on_item_changed(self, *_):
        self.signal_selection_changed.emit(self.selected_devices)

    @property
    def selected_devices(self) -> list[str]:
        """Names of the currently checked devices, in list order."""
        selected = []
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected

    def set_selected_devices(self, names: Iterable[str]):
        """Check exactly the devices in ``names`` that are available."""
        wanted = set(names)
        self._list.blockSignals(True)
        for row in range(self._list.count()):
            item = self._list.item(row)
            item.setCheckState(Qt.Checked if item.text() in wanted else Qt.Unchecked)
        self._list.blockSignals(False)
        self._on_item_changed()
