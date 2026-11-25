"""A group of movable controls with an optional batch move."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hextools.catalog.model import TypeExpr
from hextools.gui._client_signals import QtRunEngineState
from hextools.gui.movable_control import (
    QtMovableControl,
    set_enabled_keep_focus,
)
from hextools.gui.plan_errors import show_plan_error
from hextools.gui.run_engine_clients import (
    DeviceName,
    RunEngineClient,
    check_plan_name,
)


class QtMovableGroup(QWidget):
    """One or more :class:`QtMovableControl` rows that can move together.

    With *Batch* checked, editing a setpoint stages it, once focus leaves the field
    or Enter is pressed, instead of moving.
    *Move* then runs one ``move_plan(path1, value1, path2, value2, ...)``,
    moving every staged movable together. Unchecking *Batch* or *Clear* discards
    the staged setpoints.

    Parameters
    ----------
    movables : Sequence[str] or Mapping[str, str]
        Device names of the movables, or row label to name.
    re_client : RunEngineClient
        Runs the moves, and describes and reads the movables.
    title : str, optional
        Header shown above the rows.
    precision : int, optional
        Decimal places shown for float readbacks.
    tweak : float, optional
        Initial tweak step for every row.
    value_types : Mapping[str, TypeExpr], optional
        Value type of the rows that need one, by path; see
        :class:`QtMovableControl`.
    move_plan : str, optional
        Namespace name of a plan called as ``move_plan(device, value, device,
        value, ...)``, for both single-row and batch moves.
    tweak_plan : str or None, optional
        Namespace name of the relative-move plan for every row's tweaks. None
        disables tweaking.
    batch : bool, optional
        Start in batch mode.
    """

    def __init__(
        self,
        movables: Sequence[str] | Mapping[str, str],
        re_client: RunEngineClient,
        title: str | None = None,
        precision: int = 3,
        tweak: float = 1.0,
        value_types: Mapping[str, TypeExpr] | None = None,
        move_plan: str = "mv",
        tweak_plan: str | None = "mvr",
        batch: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._re_client = re_client
        self._move_plan = check_plan_name(move_plan)
        self._pending = False
        self._staged: dict[QtMovableControl, Any] = {}

        rows = (
            list(movables.items())
            if isinstance(movables, Mapping)
            else [(None, path) for path in movables]
        )
        self.controls = [
            QtMovableControl(
                path,
                re_client,
                label=label,
                precision=precision,
                tweak=tweak,
                value_type=(value_types or {}).get(path),
                move_plan=move_plan,
                tweak_plan=tweak_plan,
            )
            for label, path in rows
        ]

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(6)
        if title is not None:
            header = QLabel(title)
            header.setObjectName("statusSectionHeader")
            vbox.addWidget(header)
        for control in self.controls:
            vbox.addWidget(control)
        labels = [c.label_widget for c in self.controls if c.label_widget]
        # Padding keeps the readback clear of the longest label.
        width = max((lbl.sizeHint().width() for lbl in labels), default=0) + 12
        for lbl in labels:
            lbl.setFixedWidth(width)

        self._batch = QCheckBox("Batch")
        self._batch.setToolTip("Stage setpoints, then move them together")
        self._batch.toggled.connect(self._on_batch_toggled)
        self._summary = QLabel()
        self._move_button = QPushButton("Move")
        self._move_button.clicked.connect(self._on_move_clicked)
        self._clear_button = QPushButton("Clear")
        self._clear_button.clicked.connect(self._clear_staged)
        bar = QHBoxLayout()
        bar.addWidget(self._batch, 0, Qt.AlignmentFlag.AlignVCenter)
        bar.addWidget(self._summary, 0, Qt.AlignmentFlag.AlignVCenter)
        bar.addStretch(1)
        bar.addWidget(self._clear_button)
        bar.addWidget(self._move_button)
        # Reserve the hidden buttons' space so toggling Batch doesn't shift the layout.
        for widget in (self._summary, self._clear_button, self._move_button):
            policy = widget.sizePolicy()
            policy.setRetainSizeWhenHidden(True)
            widget.setSizePolicy(policy)
        vbox.addSpacing(4)
        vbox.addLayout(bar)
        self.setLayout(vbox)

        self._state = QtRunEngineState(re_client, self)
        self._state.changed.connect(lambda *_: self._update_bar())
        self.set_batch(batch)
        self._update_bar()

    @property
    def staged(self) -> list[tuple[str, Any]]:
        """``(path, value)`` of each staged setpoint, in row order."""
        return [(c.path, self._staged[c]) for c in self.controls if c in self._staged]

    def set_batch(self, enabled: bool):
        """Turn batch mode on or off."""
        self._batch.setChecked(enabled)

    def _on_batch_toggled(self, enabled: bool):
        self._clear_staged()
        for control in self.controls:
            control.set_batch(self._stage if enabled else None)
        self._update_bar()

    def _stage(self, control: QtMovableControl, value: Any):
        if value is None:
            self._staged.pop(control, None)
        else:
            self._staged[control] = value
        control.mark_staged(value is not None)
        self._update_bar()

    def _clear_staged(self):
        for control in self._staged:
            control.mark_staged(False)
        self._staged.clear()
        self._update_bar()

    def _update_bar(self):
        batching = self._batch.isChecked()
        ready = not self._pending and self._re_client.state == "idle"
        for widget in (self._summary, self._move_button, self._clear_button):
            widget.setVisible(batching)
        set_enabled_keep_focus(self._batch, not self._pending)
        self._summary.setText(f"({len(self._staged)} staged)" if self._staged else "")
        set_enabled_keep_focus(self._move_button, ready and bool(self._staged))
        set_enabled_keep_focus(
            self._clear_button, not self._pending and bool(self._staged)
        )

    def _on_move_clicked(self):
        moves = self.staged
        if not moves or self._pending or self._re_client.state != "idle":
            return
        self._pending = True
        for control in self.controls:
            control.set_busy(True)
        self._update_bar()
        # Defer so the click handler returns before the plan runs.
        QTimer.singleShot(0, lambda: self._execute(moves))

    def _execute(self, moves: list[tuple[str, Any]]):
        args = [arg for path, value in moves for arg in (DeviceName(path), value)]
        try:
            # Another plan may have started since the click.
            if self._re_client.state != "idle":
                raise RuntimeError("RunEngine is busy!")
            self._re_client.execute_plan(self._move_plan, *args)
        except Exception as error:
            # Keep the staged targets so the move can be retried.
            show_plan_error(self, "Batch move failed", error)
            return
        finally:
            self._pending = False
            for control in self.controls:
                control.set_busy(False)
            self._update_bar()
        self._clear_staged()
        # The moved-to setpoints are now the ones that need no move.
        for control in self.controls:
            control.capture_original()

    def closeEvent(self, event):
        self._state.stop()
        for control in self.controls:
            control.close()
        super().closeEvent(event)
