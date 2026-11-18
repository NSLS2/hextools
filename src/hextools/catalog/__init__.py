"""Describe plans and devices for clients, in the plan and device catalog format.

See ``docs/plan_device_schema.md``. :mod:`~hextools.catalog.generate` builds the
documents from a namespace, :mod:`~hextools.catalog.model` gives typed views of
them with lookups, and :mod:`~hextools.catalog.namespace` finds the devices in a
namespace.
"""

from hextools.catalog.generate import (
    Units,
    describe_plan,
    device_registry,
    is_plan,
    plan_catalog,
    set_type,
    to_value,
    type_expr,
    write_catalog,
)
from hextools.catalog.model import (
    NO_DEFAULT,
    SCHEMA_VERSION,
    DeviceNode,
    DeviceRegistry,
    Parameter,
    PlanCatalog,
    PlanDescription,
    device_types,
    load_catalog,
)

__all__ = [
    "NO_DEFAULT",
    "SCHEMA_VERSION",
    "DeviceNode",
    "DeviceRegistry",
    "Parameter",
    "PlanCatalog",
    "PlanDescription",
    "Units",
    "describe_plan",
    "device_registry",
    "device_types",
    "is_plan",
    "load_catalog",
    "plan_catalog",
    "set_type",
    "to_value",
    "type_expr",
    "write_catalog",
]
