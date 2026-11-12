"""One-line control for a movable, after the Phoebus motor widget."""

from __future__ import annotations

import enum
import inspect
import keyword
import typing
from collections.abc import Callable, MutableMapping, Sequence
from typing import Any

import numpy as np
from bluesky import RunEngine
from bluesky.protocols import Movable, Subscribable
from ophyd_async.core import Signal
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from hextools.gui._ipython import run_in_ipython, show_plan_error
from hextools.gui.device_sources import NamespaceDeviceSource
from hextools.gui.reading_sources import NamespaceReadingSource, ReadingSource
from hextools.gui.signal_status import QtSignalStatus

_INVALID_STYLE = "border: 1px solid #c62828;"
_STAGED_STYLE = "border: 1px solid #e0a800;"

#: Called with ``(control, value)`` to stage a setpoint, or ``value=None`` to unstage.
StageHandler = Callable[["QtMovableControl", Any], None]


def set_type(obj: Movable[Any]) -> type | None:
    """Return the value type ``obj.set`` takes, or None if it can't be determined."""
    if isinstance(obj, Signal) and obj.datatype is not None:
        return obj.datatype
    set_method = inspect.unwrap(type(obj).set)
    try:
        hints = typing.get_type_hints(set_method)
        params = [p for p in inspect.signature(set_method).parameters if p != "self"]
    except Exception:
        hints, params = {}, []
    hint = hints.get(params[0]) if params else None
    if isinstance(hint, type):
        return hint
    # Generic movables, e.g. StandardMovable[float], declare it on a base class.
    for cls in type(obj).__mro__:
        for base in getattr(cls, "__orig_bases__", ()):
            origin, args = typing.get_origin(base), typing.get_args(base)
            if hasattr(origin, "set") and args and isinstance(args[0], type):
                return args[0]
    return None


def _serialize(value: Any) -> str:
    """Python source for ``value`` to put in the ``mv`` command."""
    if isinstance(value, enum.Enum):
        value = value.value
    if isinstance(value, bool | int | float | str):
        return repr(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} for a move command")


def set_enabled_keep_focus(widget: QWidget, enabled: bool):
    """Enable or disable ``widget`` without Qt passing its focus to the next one."""
    if not enabled and widget.hasFocus():
        widget.clearFocus()
    widget.setEnabled(enabled)


def move_command(plan_name: str, moves: Sequence[tuple[str, Any]]) -> str:
    """``RE(plan_name(path1, value1, path2, value2, ...))`` for validated paths."""
    args = ", ".join(f"{path}, {_serialize(value)}" for path, value in moves)
    return f"RE({plan_name}({args}))"


def check_plan_name(name: str) -> str:
    """Return ``name`` if it is a (dotted) Python name, so it is safe in a command.

    Raises
    ------
    ValueError
        If ``name`` is anything else.
    """
    parts = name.split(".")
    if not all(p.isidentifier() and not keyword.iskeyword(p) for p in parts):
        raise ValueError(f"{name!r} is not a plan name")
    return name


class QtMovableControl(QWidget):
    """``label  readback  [setpoint]  [<] [tweak] [>]`` for a movable.

    Entering a setpoint runs ``RE(move_plan(path, value))`` in the IPython shell,
    and the tweak buttons run ``RE(tweak_plan(path, -tweak))`` /
    ``RE(tweak_plan(path, tweak))``. Tweaks are offered only for int and float
    movables with a ``tweak_plan``; enum and bool movables get a dropdown
    setpoint. The setpoint starts at the first readback value.

    Parameters
    ----------
    path : str
        Namespace path of the movable, e.g. ``"sample_tower.y"``.
    re : RunEngine
        The in-process RunEngine.
    namespace : MutableMapping[str, object], optional
        Namespace holding the movable and the plans. Defaults to the IPython
        ``user_ns``.
    label : str, optional
        Row label. Defaults to ``path``.
    readback : str, optional
        Name, in ``source``, of what the readback shows. Defaults to ``path``
        if the movable is subscribable.
    value_type : type, optional
        The type ``set`` takes, e.g. an enum for a dropdown. Inferred from the
        movable if not given, which fails for most ophyd v1 devices.
    precision : int, optional
        Decimal places shown for a float readback.
    tweak : float, optional
        Initial tweak step.
    move_plan : str, optional
        Namespace name of a plan called as ``move_plan(device, value, device,
        value, ...)`` that returns once the move is complete.
    tweak_plan : str or None, optional
        Namespace name of the relative-move plan, with the same signature.
        None disables tweaking.
    """

    def __init__(
        self,
        path: str,
        re: RunEngine,
        namespace: MutableMapping[str, Any] | None = None,
        label: str | None = None,
        readback: str | None = None,
        precision: int = 3,
        tweak: float = 1.0,
        value_type: type | None = None,
        move_plan: str = "mv",
        tweak_plan: str | None = "mvr",
        parent=None,
    ):
        super().__init__(parent)
        if namespace is None:
            from IPython.core.getipython import get_ipython

            ipython = get_ipython()
            namespace = ipython.user_ns if ipython is not None else {}
        source = NamespaceDeviceSource(namespace)
        # Validating the path keeps arbitrary code out of the generated command.
        if not source.is_valid_device(path, (Movable,)):
            raise ValueError(f"No movable at {path!r} in the namespace")
        obj = source.resolve_device(path)
        self._path = path
        self._re = re
        self._namespace = namespace
        self._move_plan = check_plan_name(move_plan)
        self._tweak_plan = check_plan_name(tweak_plan) if tweak_plan else None
        self._pending = False
        self._busy = False
        self._stage: StageHandler | None = None
        self._original: Any = None
        self._precision = precision
        self._type = value_type if value_type is not None else set_type(obj)
        if not _is_supported(self._type):
            raise TypeError(
                f"{path!r} takes {getattr(self._type, '__name__', 'an unknown type')}; "
                "only bool, int, float, str, and enum setpoints are supported"
            )

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if readback is None and isinstance(obj, Subscribable):
            readback = path
        reading_source: ReadingSource = NamespaceReadingSource(namespace, re.loop)
        self._readback = (
            QtSignalStatus(
                readback, reading_source, label=label or path, precision=precision
            )
            if readback is not None
            else None
        )
        if self._readback is not None:
            row.addWidget(self._readback, stretch=1)

        self._controls: list[QWidget] = []
        self._setpoint = self._make_setpoint()
        row.addWidget(self._setpoint)
        if self._readback is not None:
            self._readback.value_changed.connect(self._init_setpoint)
        self._tweak: QLineEdit | None = None
        self._tweak_widgets: list[QWidget] = []
        if self._is_numeric and self._tweak_plan is not None:
            left = QPushButton("\u25c0")
            right = QPushButton("\u25b6")
            self._tweak = QLineEdit(_format_number(tweak, self._type))
            self._tweak.setFixedWidth(70)
            self._tweak.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._tweak.setToolTip("Tweak step")
            for button, sign in ((left, -1), (right, 1)):
                button.setFixedWidth(28)
                button.setToolTip(f"Tweak by {'-' if sign < 0 else '+'}step")
                button.clicked.connect(lambda *_, s=sign: self._on_tweak(s))
            for widget in (left, self._tweak, right):
                row.addWidget(widget)
                self._controls.append(widget)
                self._tweak_widgets.append(widget)
        self.setLayout(row)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._update_enabled)
        self._timer.start()
        self._update_enabled()

    @property
    def _is_numeric(self) -> bool:
        return self._type in (int, float)

    @property
    def label_widget(self) -> QLabel | None:
        """The row label, e.g. for aligning several rows; None without a readback."""
        return self._readback.label_widget if self._readback is not None else None

    @property
    def path(self) -> str:
        """Validated namespace path of the movable."""
        return self._path

    def set_batch(self, handler: StageHandler | None):
        """Stage setpoints with ``handler`` instead of moving; None moves again.

        Tweaks are disabled while staging. A setpoint changed back to the value it
        had when staging began (see :meth:`capture_original`) is unstaged.
        """
        self._stage = handler
        if handler is not None:
            self.capture_original()
        self.mark_staged(False)
        self._update_enabled()

    def capture_original(self):
        """Remember the current setpoint as the value that needs no move."""
        self._original = self._current_value()

    def _current_value(self) -> Any:
        """Return the setpoint's value, or None if it is empty or invalid."""
        setpoint = self._setpoint
        if isinstance(setpoint, QComboBox):
            index = setpoint.currentIndex()
            return setpoint.itemData(index) if index >= 0 else None
        try:
            return self._parse(setpoint.text().strip())  # type: ignore[attr-defined]
        except ValueError:
            return None

    def _is_original(self, value: Any) -> bool:
        if self._original is None:
            return False
        if self._type is float:
            # Compare as displayed, so re-typing the shown value counts as unchanged.
            return self._format(value) == self._format(self._original)
        return value == self._original

    def mark_staged(self, staged: bool):
        """Highlight the setpoint while it holds a staged value."""
        self._setpoint.setStyleSheet(_STAGED_STYLE if staged else "")
        self._setpoint.setToolTip("Staged for the batch move" if staged else "")

    def set_busy(self, busy: bool):
        """Disable the inputs while an outside move, e.g. a batch, is running."""
        self._busy = busy
        self._update_enabled()

    def _init_setpoint(self, value: Any):
        """Start the setpoint at the first readback, unless the user got there first."""
        if self._readback is not None:
            self._readback.value_changed.disconnect(self._init_setpoint)
        setpoint = self._setpoint
        if isinstance(setpoint, QComboBox):
            if setpoint.currentIndex() == -1:
                for i in range(setpoint.count()):
                    if setpoint.itemData(i) == value:
                        setpoint.setCurrentIndex(i)
                        break
        elif isinstance(setpoint, QLineEdit) and not setpoint.text():
            text = self._format(value)
            if text is not None:
                setpoint.setText(text)
        if self._stage is not None and self._original is None:
            self.capture_original()

    def _format(self, value: Any) -> str | None:
        """Setpoint text for a readback ``value``, or None if it doesn't fit."""
        if isinstance(value, bool):
            return None
        if self._type is float and isinstance(value, int | float | np.number):
            return f"{float(value):.{self._precision}f}"
        if self._type is int and isinstance(value, int | np.integer):
            return str(int(value))
        if self._type is str and isinstance(value, str):
            return value
        return None

    def _make_setpoint(self) -> QWidget:
        choices = self._choices()
        if choices is not None:
            combo = QComboBox()
            combo.setPlaceholderText("Select\u2026")
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            combo.setMinimumWidth(100)
            for text, value in choices:
                combo.addItem(text, value)
            combo.setCurrentIndex(-1)
            combo.activated.connect(lambda i: self._move(combo.itemData(i)))
            self._controls.append(combo)
            return combo
        edit = QLineEdit()
        edit.setFixedWidth(100)
        edit.setPlaceholderText("Setpoint")
        edit.returnPressed.connect(lambda: self._on_setpoint_entered(edit))
        edit.editingFinished.connect(lambda: self._on_setpoint_finished(edit))
        edit.textEdited.connect(lambda *_: self._on_setpoint_edited(edit))
        self._controls.append(edit)
        return edit

    def _on_setpoint_edited(self, edit: QLineEdit):
        edit.setStyleSheet("")
        # A staged value no longer matches the text, so drop it until re-staged.
        if self._stage is not None:
            self._stage(self, None)

    def _on_setpoint_finished(self, edit: QLineEdit):
        """In batch mode, stage an edited setpoint once focus leaves it."""
        if self._stage is None or not edit.isModified():
            return
        edit.setModified(False)
        self._on_setpoint_entered(edit)

    def _choices(self) -> list[tuple[str, Any]] | None:
        """``(text, value)`` options for a dropdown setpoint, or None for free text."""
        if self._type is bool:
            return [("True", True), ("False", False)]
        if isinstance(self._type, type) and issubclass(self._type, enum.Enum):
            return [
                (m.value if isinstance(m.value, str) else m.name, m) for m in self._type
            ]
        return None

    def _parse(self, text: str) -> Any:
        if self._type in (int, float):
            return self._type(text)
        return text

    def _on_setpoint_entered(self, edit: QLineEdit):
        try:
            value = self._parse(edit.text().strip())
        except ValueError:
            edit.setStyleSheet(_INVALID_STYLE)
            edit.setToolTip(f"Not a valid {self._type.__name__}")  # type: ignore[union-attr]
            return
        edit.setToolTip("")
        self._move(value)

    def _on_tweak(self, sign: int):
        if self._tweak is None or self._tweak_plan is None:
            return
        try:
            step = self._parse(self._tweak.text().strip())
        except ValueError:
            self._tweak.setStyleSheet(_INVALID_STYLE)
            return
        self._tweak.setStyleSheet("")
        self._run(self._tweak_plan, sign * step)

    def _move(self, value: Any):
        if self._stage is not None:
            self._stage(self, None if self._is_original(value) else value)
            return
        self._run(self._move_plan, value)

    def _run(self, plan_name: str, value: Any):
        if self._pending or self._busy or self._re.state != "idle":
            return
        code = move_command(plan_name, [(self._path, value)])
        self._pending = True
        self._update_enabled()
        # Defer so the click handler returns before IPython executes the cell.
        QTimer.singleShot(0, lambda: self._execute(code))

    def _execute(self, code: str):
        try:
            error = run_in_ipython(code)
        finally:
            self._pending = False
            self._update_enabled()
        if error is not None:
            show_plan_error(self, f"Moving {self._path} failed", error)

    def _update_enabled(self):
        enabled = not self._pending and not self._busy and self._re.state == "idle"
        for widget in self._controls:
            batching = self._stage is not None and widget in self._tweak_widgets
            set_enabled_keep_focus(widget, enabled and not batching)

    def closeEvent(self, event):
        self._timer.stop()
        if self._readback is not None:
            self._readback.close()
        super().closeEvent(event)


def _is_supported(datatype: type | None) -> bool:
    return datatype in (bool, int, float, str) or (
        isinstance(datatype, type) and issubclass(datatype, enum.Enum)
    )


def _format_number(value: float, datatype: type | None) -> str:
    return str(int(value)) if datatype is int else f"{value:g}"
