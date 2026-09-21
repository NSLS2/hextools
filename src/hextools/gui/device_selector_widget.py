"""Qt widgets for selecting Queue Server devices by type.

The widgets discover the devices currently allowed by the Queue Server (via the
``RunEngineClient`` model) and present the subset matching a requested device
type. :class:`QtDeviceSelector` offers a single-selection dropdown;
:class:`QtMultiDeviceSelector` offers a growable set of dropdown rows (add with
``+``, remove with ``-``) for selecting any number of distinct devices.

Device "type" is expressed with :class:`DeviceType`, which maps onto the
protocol flags reported by the Queue Server (``is_readable``, ``is_movable``,
``is_flyable``). Selection can optionally be narrowed further to specific ophyd
class names.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import Enum

from qtpy.QtCore import Signal, Slot
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QSizePolicy,
    QToolButton,
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


class DynamicDeviceSelector(QWidget):
    """Select any number of distinct devices via growable dropdown rows.

    Each row is a dropdown offering only the devices not already chosen in
    another row. The first row cannot be removed; later rows show a ``-`` button
    that removes the whole row. A ``+`` button on the last row appends another
    row and is hidden once every available device has been selected.
    """

    selection_changed = Signal(object)

    def __init__(self, options: Sequence[str] | None = None, parent=None):
        super().__init__(parent)
        self._options: list[str] = list(options or [])
        self._rows: list[dict] = []
        self._updating = False

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(4)
        self.setLayout(self._rows_layout)

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self._add_row()
        self._refresh()

    def _make_button(self, text: str) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setFixedWidth(24)
        return button

    def _add_row(self, select: str | None = None) -> dict:
        combo = QComboBox()
        plus = self._make_button("+")
        minus = self._make_button("\u2212")

        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(4)
        hbox.addWidget(combo, 1)
        hbox.addWidget(plus)
        hbox.addWidget(minus)

        row = {"widget": container, "combo": combo, "plus": plus, "minus": minus}
        self._rows.append(row)
        self._rows_layout.addWidget(container)

        self._populate_combo(row, select)
        combo.currentIndexChanged.connect(self._on_combo_changed)
        plus.clicked.connect(self._on_add_clicked)
        minus.clicked.connect(lambda *_, r=row: self._remove_row(r))
        return row

    def _remove_row(self, row: dict):
        if len(self._rows) <= 1:
            return
        self._rows.remove(row)
        row["widget"].setParent(None)
        row["widget"].deleteLater()
        self._refresh()

    def _selected_except(self, exclude: dict | None) -> set[str]:
        chosen = set()
        for row in self._rows:
            if row is exclude:
                continue
            value = row["combo"].currentData()
            if value is not None:
                chosen.add(value)
        return chosen

    def _remaining(self) -> list[str]:
        chosen = self._selected_except(None)
        return [name for name in self._options if name not in chosen]

    def _populate_combo(self, row: dict, select: str | None = None):
        combo = row["combo"]
        current = select if select is not None else combo.currentData()
        others = self._selected_except(row)

        combo.blockSignals(True)
        combo.clear()
        for name in self._options:
            if name not in others:
                combo.addItem(name, userData=name)
        index = combo.findData(current)
        if index < 0 and combo.count():
            index = 0
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _refresh(self):
        if self._updating:
            return
        self._updating = True
        try:
            for row in self._rows:
                self._populate_combo(row)
            has_more = bool(self._remaining())
            last = len(self._rows) - 1
            for i, row in enumerate(self._rows):
                row["plus"].setVisible(i == last and has_more)
                row["minus"].setVisible(i > 0)
        finally:
            self._updating = False
        self.selection_changed.emit(self.selected_devices())

    def _on_combo_changed(self, *_):
        self._refresh()

    def _on_add_clicked(self, *_):
        remaining = self._remaining()
        if not remaining:
            return
        self._add_row(select=remaining[0])
        self._refresh()

    def selected_devices(self) -> list[str]:
        """Devices chosen across the rows, in row order (no duplicates)."""
        result: list[str] = []
        for row in self._rows:
            value = row["combo"].currentData()
            if value is not None and value not in result:
                result.append(value)
        return result

    def set_available(self, options: Sequence[str]):
        """Update the available devices, dropping rows that no longer apply."""
        self._options = list(options)
        kept = []
        for row in self._rows:
            value = row["combo"].currentData()
            if value is None or value in self._options:
                kept.append(row)
            else:
                row["widget"].setParent(None)
                row["widget"].deleteLater()
        self._rows = kept
        if not self._rows:
            self._add_row()
        self._refresh()

    def set_selected_devices(self, names: Iterable[str]):
        """Replace the rows so that exactly ``names`` (if available) are chosen."""
        wanted = [name for name in names if name in self._options]
        for row in self._rows:
            row["widget"].setParent(None)
            row["widget"].deleteLater()
        self._rows = []
        for name in wanted:
            self._add_row(select=name)
        if not self._rows:
            self._add_row()
        self._refresh()


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
    """Growable dropdown rows for selecting any number of devices of a type.

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
        self._selector = DynamicDeviceSelector()
        super().__init__(
            model,
            device_type,
            parent,
            classnames=classnames,
            include_components=include_components,
        )

        if title is None:
            title = f"{self._device_type.value.capitalize()} devices"

        self._selector.selection_changed.connect(self._on_selection_changed)

        group_box = QGroupBox(title)
        inner = QVBoxLayout()
        inner.setContentsMargins(8, 6, 8, 6)
        inner.addWidget(self._selector)
        group_box.setLayout(inner)

        vbox = QVBoxLayout()
        vbox.addWidget(group_box)
        self.setLayout(vbox)

    def _populate(self, names: list[str]):
        self._selector.set_available(names)

    def _on_selection_changed(self, *_):
        self.signal_selection_changed.emit(self.selected_devices)

    @property
    def selected_devices(self) -> list[str]:
        """Names of the currently selected devices, in row order."""
        return self._selector.selected_devices()

    def set_selected_devices(self, names: Iterable[str]):
        """Show one row per device in ``names`` that is available."""
        self._selector.set_selected_devices(names)
