"""A Qt widget that builds a plan-submission form from a plan's type hints.

:class:`QtPlanWidget` takes any bluesky plan function and inspects its signature
to generate input fields:

* scalar parameters (``int``, ``float``, ``bool``, ``str`` and their optional
  variants) get a matching editor;
* a parameter typed as a *list* of devices (e.g. ``list[KinetixDetector |
  PhantomDetector]``) gets a growable device selector;
* a *single* device parameter that is required gets a device dropdown, while a
  single device parameter that defaults to ``None`` is treated as auto-resolved
  by the plan at run time and is omitted from the form.

Device fields are populated and validated through a pluggable
:class:`~hextools.gui.device_sources.DeviceSource`, so the same widget can check
inputs against an in-process IPython namespace or the Queue Server's allowed
devices. Depending on :class:`ExecutionMode`, the submit button either adds a
:class:`~bluesky_queueserver_api.BPlan` to the queue or runs the plan in this
process by sending ``RE(plan(...))`` to the IPython shell.
"""

from __future__ import annotations

import collections.abc
import inspect
import types
import typing
from enum import Enum

import IPython
from bluesky import RunEngine
from bluesky_queueserver_api import BPlan
from bluesky_widgets.qt.threading import FunctionWorker
from bluesky_widgets.models.run_engine_client import RunEngineClient
from qtpy.QtCore import Qt, QTimer, Signal, Slot
from qtpy.QtGui import QDoubleValidator, QIntValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from hextools.gui._ipython import run_in_ipython
from hextools.gui.device_selector_widget import DynamicDeviceSelector
from hextools.gui.device_sources import (
    DeviceSource,
    NamespaceDeviceSource,
    QueueServerDeviceSource,
)


_NO_SELECTION = "\u2014"  # em dash placeholder for an unset single device


def _unwrap_optional(annotation):
    """Return ``(base_type, is_optional)`` for ``T`` or ``Optional[T]``."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union or isinstance(annotation, types.UnionType):
        args = list(typing.get_args(annotation))
        non_none = [a for a in args if a is not type(None)]
        is_optional = len(non_none) != len(args)
        if len(non_none) == 1:
            return non_none[0], is_optional
        return annotation, is_optional
    return annotation, False


def _scalar_type(annotation):
    """Return int/float/bool/str for a scalar annotation, else None."""
    base, _ = _unwrap_optional(annotation)
    if base is bool:
        return bool
    if base in (int, float, str):
        return base
    return None


def _enum_type(annotation):
    """Return the Enum subclass for an enum annotation, else None."""
    base, _ = _unwrap_optional(annotation)
    if isinstance(base, type) and issubclass(base, Enum):
        return base
    return None


def _is_list_annotation(annotation) -> bool:
    base, _ = _unwrap_optional(annotation)
    origin = typing.get_origin(base)
    return origin in (list, tuple, set, frozenset) or origin in (
        collections.abc.Sequence,
        collections.abc.Iterable,
    )


def _device_types(annotation) -> tuple[type, ...] | None:
    """Return the concrete device classes named by ``annotation``, or None."""
    base, _ = _unwrap_optional(annotation)
    origin = typing.get_origin(base)
    if origin is typing.Union or isinstance(base, types.UnionType):
        members = [
            a
            for a in typing.get_args(base)
            if a is not type(None) and isinstance(a, type)
        ]
        return tuple(members) or None
    if isinstance(base, type):
        return (base,)
    return None


def _element_device_types(annotation) -> tuple[type, ...] | None:
    """Return the device classes of a list annotation's element type."""
    base, _ = _unwrap_optional(annotation)
    args = typing.get_args(base)
    if not args:
        return None
    return _device_types(args[0])


class QtPlanWidget(QWidget):
    """Build and run an arbitrary plan, either in-process or via the Queue Server.

    Parameters
    ----------
    model : bluesky_widgets.models.run_engine_client.RunEngineClient or None
        The Queue Server client model (e.g. ``viewer.run_engine``). Required for
        :attr:`ExecutionMode.QUEUESERVER`; may be ``None`` when running
        in-process.
    plan : Callable
        The plan function whose signature drives the generated form. Its
        ``__name__`` is used both as the Queue Server plan name and, for
        in-process execution, as the callable looked up in the IPython
        namespace.
    parent : QWidget, optional
        Parent widget.
    execution_mode : ExecutionMode or str, optional
        Whether the submit button adds the plan to the Queue Server
        (:attr:`ExecutionMode.QUEUESERVER`, the default) or runs it in this
        process via ``RE(plan(...))`` (:attr:`ExecutionMode.IN_PROCESS`).
    device_source : DeviceSource, optional
        Source used to populate and validate device fields. Defaults to a
        :class:`~hextools.gui.device_sources.QueueServerDeviceSource` wrapping
        ``model`` in queueserver mode, or a
        :class:`~hextools.gui.device_sources.NamespaceDeviceSource` in
        in-process mode.
    namespace : Mapping[str, object], optional
        Namespace used for in-process validation/execution. Defaults to the
        active IPython ``user_ns``. Ignored when ``device_source`` is given.
    re_name : str, optional
        Name of the ``RunEngine`` object in the namespace for in-process
        execution. Default ``"RE"``.
    title : str, optional
        Group box title. Default: derived from the plan name.
    """

    signal_devices_changed = Signal()

    def __init__(
        self,
        re_client,
        plan,
        parent=None,
        *,
        title: str | None = None,
    ):
        super().__init__(parent)
        self._re_client = re_client
        self._plan = plan
        if isinstance(re_client, RunEngineClient):
            self._device_source = QueueServerDeviceSource(re_client)
        else:
            ipython = IPython.get_ipython()
            if ipython is None:
                raise RuntimeError("Failed to detect ipython instance, required for plan argument validation")
            self._device_source = NamespaceDeviceSource(ipython.user_ns)
        self._worker = None
        self._scalar_fields: list[tuple[str, type, QWidget, bool]] = []
        self._enum_fields: list[tuple[str, QComboBox, bool]] = []
        self._device_fields: list[dict] = []

        if title is None:
            title = plan.__name__.replace("_", " ").title()

        outer = QVBoxLayout()
        group_box = QGroupBox(title)
        vbox = QVBoxLayout()

        self._build_fields(vbox)

        button_label = (
            "Run Plan"
            if not isinstance(re_client, RunEngineClient)
            else "Submit to Queue"
        )
        self._submit_button = QPushButton(button_label)
        self._submit_button.clicked.connect(self._on_submit_clicked)
        vbox.addWidget(self._submit_button)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        vbox.addWidget(self._status_label)

        group_box.setLayout(vbox)
        outer.addWidget(group_box)
        self.setLayout(outer)

        self.signal_devices_changed.connect(self._refresh_devices)
        self._device_source.subscribe(self._on_devices_changed)
        self._refresh_devices()

    # -- Form construction -----------------------------------------------------

    def _build_fields(self, vbox: QVBoxLayout):
        form = QFormLayout()
        form.setContentsMargins(8, 6, 8, 6)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)

        parameters = inspect.signature(self._plan).parameters
        for name, param in parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            annotation = param.annotation
            required = param.default is inspect.Parameter.empty

            kind = _scalar_type(annotation)
            if kind is not None:
                self._add_scalar_field(form, name, kind, param, required)
                continue

            enum_cls = _enum_type(annotation)
            if enum_cls is not None:
                self._add_enum_field(form, name, enum_cls, param, required)
                continue

            if _is_list_annotation(annotation):
                types_ = _element_device_types(annotation)
                self._add_device_list_field(vbox, name, types_, required)
                continue

            # Single device parameter.
            if required:
                types_ = _device_types(annotation)
                self._add_single_device_field(form, name, types_)
            # Optional single device (default None): resolved by the plan.

        if form.rowCount():
            params_box = QGroupBox("Parameters")
            params_box.setLayout(form)
            vbox.addWidget(params_box)

    def _add_scalar_field(self, form, name, kind, param, required):
        label = f"{name.replace('_', ' ').title()}:"
        if kind is bool:
            widget = QCheckBox()
            widget.setChecked(bool(param.default) if not required else False)
        else:
            widget = QLineEdit()
            if kind is int:
                widget.setValidator(QIntValidator(widget))
            elif kind is float:
                validator = QDoubleValidator(widget)
                validator.setNotation(QDoubleValidator.StandardNotation)
                widget.setValidator(validator)
            if not required and param.default is not None:
                widget.setText(str(param.default))
            widget.setPlaceholderText("required" if required else "optional")
        self._scalar_fields.append((name, kind, widget, required))
        form.addRow(label, widget)

    def _add_enum_field(self, form, name, enum_cls, param, required):
        combo = QComboBox()
        for member in enum_cls:
            combo.addItem(str(member.value), member.value)
        if not required and param.default is not None:
            index = combo.findData(getattr(param.default, "value", param.default))
            if index >= 0:
                combo.setCurrentIndex(index)
        self._enum_fields.append((name, combo, required))
        form.addRow(f"{name.replace('_', ' ').title()}:", combo)

    def _add_device_list_field(self, vbox, name, types_, required):
        selector = DynamicDeviceSelector()
        box = QGroupBox(name.replace("_", " ").title())
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(selector)
        box.setLayout(layout)
        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        vbox.addWidget(box)
        self._device_fields.append(
            {
                "name": name,
                "types": types_,
                "kind": "list",
                "widget": selector,
                "required": required,
            }
        )

    def _add_single_device_field(self, form, name, types_):
        combo = QComboBox()
        form.addRow(f"{name.replace('_', ' ').title()}:", combo)
        self._device_fields.append(
            {
                "name": name,
                "types": types_,
                "kind": "single",
                "widget": combo,
                "required": True,
            }
        )

    # -- Device availability ---------------------------------------------------

    def _on_devices_changed(self, event=None):
        # May be fired from a background thread; marshal onto the GUI thread.
        self.signal_devices_changed.emit()

    @Slot()
    def _refresh_devices(self):
        for field in self._device_fields:
            options = self._device_source.list_devices(field["types"])
            if field["kind"] == "list":
                field["widget"].set_available(options)
            else:
                combo = field["widget"]
                current = combo.currentData()
                combo.blockSignals(True)
                combo.clear()
                combo.addItem(_NO_SELECTION, None)
                for name in options:
                    combo.addItem(name, name)
                index = combo.findData(current)
                combo.setCurrentIndex(index if index >= 0 else 0)
                combo.blockSignals(False)

    # -- Submission ------------------------------------------------------------

    def _collect_inputs(self) -> tuple[dict, set[str]]:
        """Gather plan kwargs and the set of device-valued keys.

        Device kwargs hold device *names* (strings, or a list of strings);
        scalar kwargs hold Python values. Raises ValueError on bad input.
        """
        kwargs: dict = {}
        device_keys: set[str] = set()

        for field in self._device_fields:
            name, types_, kind = field["name"], field["types"], field["kind"]
            if kind == "list":
                selected = field["widget"].selected_devices()
                if field["required"] and not selected:
                    raise ValueError(f"Select at least one device for '{name}'")
                for dev in selected:
                    self._validate_device(name, dev, types_)
                if selected:
                    kwargs[name] = selected
                    device_keys.add(name)
            else:
                dev = field["widget"].currentData()
                if dev is None:
                    if field["required"]:
                        raise ValueError(f"Select a device for '{name}'")
                    continue
                self._validate_device(name, dev, types_)
                kwargs[name] = dev
                device_keys.add(name)

        for name, kind, widget, required in self._scalar_fields:
            if kind is bool:
                kwargs[name] = widget.isChecked()
                continue
            text = widget.text().strip()
            if not text:
                if required:
                    raise ValueError(f"'{name}' is required")
                continue
            try:
                kwargs[name] = (
                    int(text)
                    if kind is int
                    else float(text)
                    if kind is float
                    else text
                )
            except ValueError as ex:
                raise ValueError(f"'{name}' is not a valid {kind.__name__}") from ex

        for name, combo, required in self._enum_fields:
            value = combo.currentData()
            if value is None:
                if required:
                    raise ValueError(f"'{name}' is required")
                continue
            kwargs[name] = value

        return kwargs, device_keys

    def _validate_device(self, param_name, device_name, types_):
        if not self._device_source.is_valid_device(device_name, types_):
            raise ValueError(
                f"'{device_name}' is not a valid device for '{param_name}'"
            )

    def _format_plan_call(self, kwargs: dict, device_keys: set[str]) -> str:
        """Render ``RE(plan(...))`` for in-process execution in the namespace."""
        parts = []
        for name, value in kwargs.items():
            if name in device_keys:
                rendered = (
                    "[" + ", ".join(value) + "]"
                    if isinstance(value, list)
                    else value
                )
            else:
                rendered = repr(value)
            parts.append(f"{name}={rendered}")
        return f"RE({self._plan.__name__}({', '.join(parts)}))"

    def _on_submit_clicked(self):
        if self._worker is not None:
            return
        try:
            kwargs, device_keys = self._collect_inputs()
        except ValueError as ex:
            self._set_status(str(ex), error=True)
            return

        if isinstance(self._re_client, RunEngine):
            self._run_in_process(kwargs, device_keys)
        else:
            self._submit_to_queue(kwargs)

    def _submit_to_queue(self, kwargs: dict):
        item = BPlan(self._plan.__name__, **kwargs)
        self._submit_button.setEnabled(False)
        self._set_status("Submitting\u2026")
        self._worker = FunctionWorker(self._submit_plan, item)
        if self._worker.returned is None or self._worker.errored is None:
            raise RuntimeError("Failed to create worker for submitting plan.")
        self._worker.returned.connect(self._on_submit_returned)
        self._worker.errored.connect(self._on_submit_errored)
        self._worker.start()

    def _submit_plan(self, item):
        self._re_client.queue_item_add(item=item)

    def _run_in_process(self, kwargs: dict, device_keys: set[str]):
        code = self._format_plan_call(kwargs, device_keys)
        self._set_status("Running\u2026")
        # Defer so the click handler returns before IPython executes the cell.
        QTimer.singleShot(0, lambda: self._execute_cell(code))

    def _execute_cell(self, code: str):
        error = run_in_ipython(code)
        if error is not None:
            self._set_status(f"Plan failed: {error}", error=True)
            QMessageBox.critical(
                self,
                "Plan execution failed",
                f"{type(error).__name__}: {error}",
            )
            return
        self._set_status("Plan complete.")

    def _on_submit_returned(self, _result=None):
        self._worker = None
        self._submit_button.setEnabled(True)
        self._set_status("Plan added to the queue.")

    def _on_submit_errored(self, exc):
        self._worker = None
        self._submit_button.setEnabled(True)
        self._set_status(f"Failed to add plan: {exc}", error=True)

    def _set_status(self, message: str, *, error: bool = False):
        self._status_label.setText(message)
        self._status_label.setStyleSheet("color: #DB3526;" if error else "")

    def closeEvent(self, event):
        self._device_source.unsubscribe(self._on_devices_changed)
        super().closeEvent(event)
