"""Controls for the four blades and the gap/center motions of a set of slits."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

from bluesky import RunEngine
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from hextools.gui.device_sources import NamespaceDeviceSource
from hextools.gui.movable_group import QtMovableGroup
from hextools.photon_delivery_system.slits import Slits

# Gaps and centers drive the same blades, so each is its own batch group, as in
# move_slits, which moves gaps before centers.
# (title, ((attribute, label), ...))
_GROUPS = (
    (
        "Blades",
        (
            ("inboard", "Inboard"),
            ("outboard", "Outboard"),
            ("bottom", "Bottom"),
            ("top", "Top"),
        ),
    ),
    ("Gap", (("horizontal_gap", "H Gap"), ("vertical_gap", "V Gap"))),
    (
        "Center",
        (("horizontal_center", "H Center"), ("vertical_center", "V Center")),
    ),
)

# (row, column, row span) of the blades, gap, and center groups.
_SIDE_BY_SIDE = ((0, 0, 2), (0, 1, 1), (1, 1, 1))
_STACKED = ((0, 0, 1), (1, 0, 1), (2, 0, 1))


class QtSlitsControl(QWidget):
    """Group box of batchable movable groups for one set of slits.

    The four real blade motors are beside (or above, if ``stacked``) the derived
    gap and center motions, which are in separate groups so a batch never mixes
    them.

    Parameters
    ----------
    path : str
        Namespace path of the slits, e.g. ``"a_slits"``.
    re : RunEngine
        The in-process RunEngine.
    namespace : MutableMapping[str, object], optional
        Namespace holding the slits. Defaults to the IPython ``user_ns``.
    title : str, optional
        Group box title. Defaults to ``path``.
    precision : int, optional
        Decimal places shown for readbacks.
    tweak : float, optional
        Initial tweak step for every motion.
    stacked : bool, optional
        Put the blades above the gap and center motions instead of beside them.
    """

    def __init__(
        self,
        path: str,
        re: RunEngine,
        namespace: MutableMapping[str, Any] | None = None,
        title: str | None = None,
        precision: int = 3,
        tweak: float = 0.1,
        stacked: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        if namespace is None:
            from IPython.core.getipython import get_ipython

            ipython = get_ipython()
            namespace = ipython.user_ns if ipython is not None else {}
        if not NamespaceDeviceSource(namespace).is_valid_device(path, (Slits,)):
            raise ValueError(f"No slits at {path!r} in the namespace")

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(12)
        self.groups: list[QtMovableGroup] = []
        positions = _STACKED if stacked else _SIDE_BY_SIDE
        for (group_title, motions), (row, col, span) in zip(
            _GROUPS, positions, strict=True
        ):
            group = QtMovableGroup(
                {label: f"{path}.{attr}" for attr, label in motions},
                re,
                namespace=namespace,
                title=group_title,
                precision=precision,
                tweak=tweak,
            )
            grid.addWidget(group, row, col, span, 1, Qt.AlignmentFlag.AlignTop)
            self.groups.append(group)
        grid.setRowStretch(grid.rowCount(), 1)
        labels = [
            c.label_widget for g in self.groups for c in g.controls if c.label_widget
        ]
        width = max((lbl.sizeHint().width() for lbl in labels), default=0)
        for lbl in labels:
            lbl.setFixedWidth(width)

        group_box = QGroupBox(title if title is not None else path)
        group_box.setLayout(grid)
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(group_box)
        self.setLayout(vbox)

    def closeEvent(self, event):
        for group in self.groups:
            group.close()
        super().closeEvent(event)


class QtSlitsSelector(QWidget):
    """Dropdown of slits, with the :class:`QtSlitsControl` of the selected one below.

    Each control is built the first time its slits are selected, then kept.

    Parameters
    ----------
    slits : Mapping[str, str]
        Namespace path of each set of slits to its display title, in order.
    re : RunEngine
        The in-process RunEngine.
    namespace : MutableMapping[str, object], optional
        Namespace holding the slits. Defaults to the IPython ``user_ns``.
    stacked : bool, optional
        Passed to each :class:`QtSlitsControl`.
    """

    def __init__(
        self,
        slits: Mapping[str, str],
        re: RunEngine,
        namespace: MutableMapping[str, Any] | None = None,
        stacked: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._re = re
        self._namespace = namespace
        self._stacked = stacked
        self.controls: dict[str, QtSlitsControl] = {}

        self._combo = QComboBox()
        for path, title in slits.items():
            self._combo.addItem(title, path)
        self._pages = QStackedWidget()
        vbox = QVBoxLayout()
        vbox.addWidget(self._combo)
        vbox.addWidget(self._pages)
        vbox.addStretch(1)
        self.setLayout(vbox)

        if not slits:
            self._combo.setEnabled(False)
            self._pages.addWidget(QLabel("No slits found"))
            return
        self._combo.currentIndexChanged.connect(self._show)
        self._show(0)

    def select(self, path: str):
        """Show the control for the slits at ``path``."""
        self._combo.setCurrentIndex(self._combo.findData(path))

    def _show(self, index: int):
        path = self._combo.itemData(index)
        control = self.controls.get(path)
        if control is None:
            control = QtSlitsControl(
                path,
                self._re,
                namespace=self._namespace,
                title=self._combo.itemText(index),
                stacked=self._stacked,
            )
            self.controls[path] = control
            self._pages.addWidget(control)
        self._pages.setCurrentWidget(control)

    def closeEvent(self, event):
        for control in self.controls.values():
            control.close()
        super().closeEvent(event)
