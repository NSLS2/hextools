"""A Qt widget that builds a plan-submission form from a plan's catalog entry.

:class:`QtPlanWidget` looks a plan up by name in a RunEngine client's
:class:`~hextools.catalog.model.PlanCatalog` and builds input fields from its
parameters' types:

* scalar parameters (``boolean``, ``integer``, ``number``, ``string`` and their
  optional variants) get a matching editor, and ``enum`` ones a dropdown;
* a *sequence* of devices gets a growable device selector, and a sequence of
  scalars a comma-separated text field;
* a *single* device parameter that is required gets a device dropdown, while a
  single device parameter that defaults to ``None`` is treated as auto-resolved
  by the plan at run time and is omitted from the form.

Device fields are offered and validated from the client's
:class:`~hextools.catalog.model.DeviceRegistry`, and offered again whenever it
changes. The run button runs the plan through the client.
"""

from __future__ import annotations

import html
from typing import Any

from qtpy.QtCore import QEvent, QSize, Qt, QTimer, Slot
from qtpy.QtGui import QDoubleValidator, QIntValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from hextools.catalog.model import TypeExpr
from hextools.gui._client_signals import QtCatalog
from hextools.gui.device_selector_widget import DynamicDeviceSelector
from hextools.gui.plan_errors import show_plan_error
from hextools.gui.run_engine_clients import (
    DeviceName,
    PlanError,
    PlanPausedError,
    RunEngineClient,
)

_NO_SELECTION = "\u2014"  # em dash placeholder for an unset single device
_SCALAR_KINDS: dict[str, type] = {
    "boolean": bool,
    "integer": int,
    "number": float,
    "string": str,
}


def _tooltip(description: str | None) -> str | None:
    """Rich-text tooltip for a parameter description, wrapping long lines."""
    if not description:
        return None
    return f"<p style='white-space:normal'>{html.escape(description)}</p>"


def _unwrap_optional(expr: TypeExpr) -> TypeExpr:
    """Return the type ``expr`` makes optional, or ``expr`` itself."""
    return expr["of"] if expr.get("kind") == "optional" else expr


def _scalar_type(expr: TypeExpr) -> type | None:
    """Return bool/int/float/str for a scalar type, else None."""
    return _SCALAR_KINDS.get(_unwrap_optional(expr).get("kind", ""))


def _enum_choices(expr: TypeExpr) -> list[tuple[str, Any]] | None:
    """``(text, value)`` of each choice of an enum type, else None."""
    inner = _unwrap_optional(expr)
    if inner.get("kind") != "enum":
        return None
    return [(str(c.get("label", c["value"])), c["value"]) for c in inner["choices"]]


def _sequence_items(expr: TypeExpr) -> TypeExpr | None:
    """Return the item type of a sequence or tuple type, else None."""
    inner = _unwrap_optional(expr)
    items = inner.get("items") if inner.get("kind") in ("sequence", "tuple") else None
    if isinstance(items, list):
        items = items[0] if items else None
    return _unwrap_optional(items) if items else None


def _device_type(expr: TypeExpr | None) -> TypeExpr | None:
    """Return ``expr`` (made non-optional) if it is a device type, else None."""
    if expr is None:
        return None
    inner = _unwrap_optional(expr)
    return inner if inner.get("kind") == "device" else None


class _VerticalScrollArea(QScrollArea):
    """Scroll vertically only, always wide enough for content plus the scrollbar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def setWidget(self, widget: QWidget):
        super().setWidget(widget)
        widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        # Content width changes (e.g. devices populated, rows added) must re-size
        # the area.
        if obj is self.widget() and event.type() == QEvent.Type.LayoutRequest:
            self.updateGeometry()
        return super().eventFilter(obj, event)

    def _content_width(self) -> int:
        widget = self.widget()
        if widget is None:
            return 0
        return (
            widget.minimumSizeHint().width()
            + self.verticalScrollBar().sizeHint().width()
            + 2 * self.frameWidth()
        )

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        return QSize(max(hint.width(), self._content_width()), hint.height())

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(max(hint.width(), self._content_width()), hint.height())


class QtPlanWidget(QWidget):
    """Build and run a plan from its entry in a RunEngine client's plan catalog.

    Parameters
    ----------
    re_client : RunEngineClient
        Describes and runs the plan, and offers and validates its devices.
    plan_name : str
        Name of the plan in the client's plan catalog.
    parent : QWidget, optional
        Parent widget.
    title : str, optional
        Group box title. Default: derived from the plan name.

    Raises
    ------
    KeyError
        If the client has no plan called ``plan_name``.
    """

    def __init__(
        self,
        re_client: RunEngineClient,
        plan_name: str,
        parent=None,
        *,
        title: str | None = None,
    ):
        super().__init__(parent)
        self._re_client = re_client
        self._plan_name = plan_name
        try:
            self._plan = re_client.plan_catalog[plan_name]
        except KeyError:
            raise KeyError(
                f"No plan called {plan_name!r} in the RunEngine client's plan "
                "catalog; is it defined or imported in the profile namespace?"
            ) from None
        self._scalar_fields: list[tuple[str, type, QWidget, bool]] = []
        self._scalar_list_fields: list[tuple[str, type | None, QWidget, bool]] = []
        self._enum_fields: list[tuple[str, QComboBox, bool]] = []
        self._device_fields: list[dict] = []

        if title is None:
            title = plan_name.replace("_", " ").title()

        outer = QVBoxLayout()
        group_box = QGroupBox(title)
        vbox = QVBoxLayout()

        # Inputs scroll instead of growing the window; the button and status stay
        # visible.
        fields = QWidget()
        fields_layout = QVBoxLayout(fields)
        fields_layout.setContentsMargins(0, 0, 0, 0)
        self._build_fields(fields_layout)
        fields_layout.addStretch()
        scroll = _VerticalScrollArea()
        scroll.setWidget(fields)
        vbox.addWidget(scroll, stretch=1)

        self._submit_button = QPushButton("Run Plan")
        self._submit_button.clicked.connect(self._on_submit_clicked)
        vbox.addWidget(self._submit_button)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        vbox.addWidget(self._status_label)

        group_box.setLayout(vbox)
        outer.addWidget(group_box)
        self.setLayout(outer)

        self._catalog = QtCatalog(re_client, self)
        self._catalog.changed.connect(lambda *_: self._refresh_devices())
        self._refresh_devices()

    # -- Form construction -----------------------------------------------------

    def _tooltip(self, name: str) -> str | None:
        return _tooltip(self._plan.parameter(name).description)

    def _add_row(self, form: QFormLayout, name: str, widget: QWidget):
        label = QLabel(f"{name.replace('_', ' ').title()}:")
        tooltip = self._tooltip(name)
        if tooltip:
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
        form.addRow(label, widget)

    def _build_fields(self, vbox: QVBoxLayout):
        form = QFormLayout()
        form.setContentsMargins(8, 6, 8, 6)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)

        for param in self._plan.parameters:
            if param.kind in ("var_positional", "var_keyword"):
                continue
            name, expr, required = param.name, param.type, param.required
            default = param.default if param.has_default else None

            kind = _scalar_type(expr)
            if kind is not None:
                self._add_scalar_field(form, name, kind, default, required)
                continue

            choices = _enum_choices(expr)
            if choices is not None:
                self._add_enum_field(form, name, choices, default, required)
                continue

            items = _sequence_items(expr)
            if items is not None:
                device = _device_type(items)
                if device is not None:
                    self._add_device_list_field(vbox, name, device, required)
                else:
                    # Other sequences are entered as comma-separated text.
                    item_kind = _SCALAR_KINDS.get(items.get("kind", ""))
                    self._add_scalar_list_field(
                        form, name, item_kind, default, required
                    )
                continue

            # Single device parameter; any other required type offers any device.
            if required:
                self._add_single_device_field(form, name, _device_type(expr))
            # Optional single device (default None): resolved by the plan.

        if form.rowCount():
            params_box = QGroupBox("Parameters")
            params_box.setLayout(form)
            vbox.addWidget(params_box)

    def _add_scalar_list_field(self, form, name, kind, default, required):
        widget = QLineEdit()
        if not required and default is not None:
            widget.setText(", ".join(str(v) for v in default))
        widget.setPlaceholderText("required" if required else "optional")
        self._scalar_list_fields.append((name, kind, widget, required))
        self._add_row(form, name, widget)

    def _add_scalar_field(self, form, name, kind, default, required):
        if kind is bool:
            widget = QCheckBox()
            widget.setChecked(bool(default) if not required else False)
        else:
            widget = QLineEdit()
            if kind is int:
                widget.setValidator(QIntValidator(widget))
            elif kind is float:
                validator = QDoubleValidator(widget)
                validator.setNotation(QDoubleValidator.Notation.StandardNotation)
                widget.setValidator(validator)
            if not required and default is not None:
                widget.setText(str(default))
            widget.setPlaceholderText("required" if required else "optional")
        self._scalar_fields.append((name, kind, widget, required))
        self._add_row(form, name, widget)

    def _add_enum_field(self, form, name, choices, default, required):
        combo = QComboBox()
        if not required and default is None:
            combo.addItem(_NO_SELECTION, None)
        for text, value in choices:
            combo.addItem(text, value)
        if not required and default is not None:
            index = combo.findData(default)
            if index >= 0:
                combo.setCurrentIndex(index)
        self._enum_fields.append((name, combo, required))
        self._add_row(form, name, combo)

    def _add_device_list_field(self, vbox, name, device_type, required):
        selector = DynamicDeviceSelector()
        box = QGroupBox(name.replace("_", " ").title())
        if tooltip := self._tooltip(name):
            box.setToolTip(tooltip)
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(selector)
        box.setLayout(layout)
        box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        vbox.addWidget(box)
        self._device_fields.append(
            {
                "name": name,
                "type": device_type,
                "kind": "list",
                "widget": selector,
                "required": required,
            }
        )

    def _add_single_device_field(self, form, name, device_type):
        combo = QComboBox()
        self._add_row(form, name, combo)
        self._device_fields.append(
            {
                "name": name,
                "type": device_type,
                "kind": "single",
                "widget": combo,
                "required": True,
            }
        )

    # -- Device availability ---------------------------------------------------

    def _device_options(self, device_type: TypeExpr | None) -> list[str]:
        """Sorted paths of the devices of ``device_type``; None means any device."""
        registry = self._re_client.device_registry
        if device_type is None:
            return sorted(path for path, _ in registry.walk(signals=False))
        return sorted(registry.find(device_type))

    @Slot()
    def _refresh_devices(self):
        for field in self._device_fields:
            options = self._device_options(field["type"])
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
            name, device_type, kind = field["name"], field["type"], field["kind"]
            if kind == "list":
                selected = field["widget"].selected_devices()
                if field["required"] and not selected:
                    raise ValueError(f"Select at least one device for '{name}'")
                for dev in selected:
                    self._validate_device(name, dev, device_type)
                if selected:
                    kwargs[name] = selected
                    device_keys.add(name)
            else:
                dev = field["widget"].currentData()
                if dev is None:
                    if field["required"]:
                        raise ValueError(f"Select a device for '{name}'")
                    continue
                self._validate_device(name, dev, device_type)
                kwargs[name] = dev
                device_keys.add(name)

        for name, kind, widget, required in self._scalar_fields:
            if kind is bool:
                kwargs[name] = widget.isChecked()  # type: ignore
                continue
            text = widget.text().strip()  # type: ignore
            if not text:
                if required:
                    raise ValueError(f"'{name}' is required")
                continue
            try:
                kwargs[name] = (
                    int(text) if kind is int else float(text) if kind is float else text
                )
            except ValueError as ex:
                raise ValueError(f"'{name}' is not a valid {kind.__name__}") from ex

        for name, kind, widget, required in self._scalar_list_fields:
            text = widget.text().strip()  # type: ignore
            elems = [e.strip() for e in text.split(",") if e.strip()]
            if not elems:
                if required:
                    raise ValueError(f"'{name}' is required")
                continue
            try:
                kwargs[name] = (
                    [int(e) for e in elems]
                    if kind is int
                    else [float(e) for e in elems]
                    if kind is float
                    else elems
                )
            except ValueError as ex:
                raise ValueError(
                    f"'{name}' contains invalid {getattr(kind, '__name__', 'str')} "
                    "values"
                ) from ex

        for name, combo, required in self._enum_fields:
            value = combo.currentData()
            if value is None:
                if required:
                    raise ValueError(f"'{name}' is required")
                continue
            kwargs[name] = value

        return kwargs, device_keys

    def _validate_device(self, param_name, device_name, device_type):
        try:
            node = self._re_client.device_registry.node(device_name)
        except KeyError:
            node = None
        valid = node is not None and (
            not node.signal if device_type is None else node.satisfies(device_type)
        )
        if not valid:
            raise ValueError(
                f"'{device_name}' is not a valid device for '{param_name}'"
            )

    def _on_submit_clicked(self):
        try:
            kwargs, device_keys = self._collect_inputs()
        except ValueError as ex:
            self._set_status(str(ex), error=True)
            return

        for name in device_keys & kwargs.keys():
            value = kwargs[name]
            kwargs[name] = (
                [DeviceName(v) for v in value]
                if isinstance(value, list)
                else DeviceName(value)
            )
        self._set_status("Running\u2026")
        # Defer so the click handler returns before the plan runs.
        QTimer.singleShot(0, lambda: self._execute(kwargs))

    def _execute(self, kwargs: dict):
        try:
            self._re_client.execute_plan(self._plan_name, **kwargs)
        except PlanPausedError as error:
            self._set_status("Plan paused.")
            show_plan_error(self, "Plan execution failed", error)
            return
        except PlanError as error:
            self._set_status(f"Plan failed: {error}", error=True)
            show_plan_error(self, "Plan execution failed", error)
            return
        self._set_status("Plan complete.")

    def _set_status(self, message: str, *, error: bool = False):
        self._status_label.setText(message)
        self._status_label.setStyleSheet("color: #DB3526;" if error else "")

    def closeEvent(self, event):
        self._catalog.stop()
        super().closeEvent(event)
