# Plan and device catalog schema

:::{note}
This is a **proposed** format. The GUI currently uses the Queue Server's plan
description format (`Plan`, `PlanParameter` and `ParameterAnnotation` in
`hextools.gui.run_engine_clients`); this schema is meant to replace it.
:::

A GUI client needs to know which plans it can run, what each parameter
accepts, and which devices exist and what they can do. In process, the GUI can
inspect Python objects directly: `isinstance` checks, the device tree,
`describe()`, signal datatypes. This schema carries the same information as
JSON, so a client can behave the same way whether the plans run in process or
on a remote server.

The JSON Schema is
[`plan_device_catalog.schema.json`](_static/schemas/plan_device_catalog.schema.json)
(draft 2020-12).

## Design

- **Plans say what they need; devices say what they are.** A device parameter
  lists the protocols and classes it requires. The client works out which
  devices qualify by checking the device registry, so the choices are never
  stale and are always checked the same way.
- **Types are trees, not strings.** `list[Movable | None]` becomes nested
  objects the client can walk, rather than a Python string it would have to
  evaluate.
- **Values are JSON.** Defaults and arguments are JSON values. Devices, plans
  and records are tagged objects (`{"$device": ...}`), so they can't be
  mistaken for strings.
- **Signals reuse event-model.** A signal's dtype, shape, units, precision,
  limits and enum choices are its `DataKey`, exactly as `describe()` returns
  it.
- **Changes are incremental.** Both documents have a `revision`. Clients cache
  the registry and apply deltas instead of re-fetching it.

There are four documents, distinguished by their top-level keys:

| Document         | Top-level keys                               | Purpose                                  |
| ---------------- | -------------------------------------------- | ---------------------------------------- |
| `PlanCatalog`    | `schema_version`, `revision`, `plans`        | The plans that may be run.               |
| `DeviceRegistry` | `schema_version`, `revision`, `devices`      | The devices that exist and what they do. |
| `RegistryDelta`  | `schema_version`, `base_revision`, `revision` | Changes to the registry.                 |
| `PlanCall`       | `plan`, `args`, `kwargs`                     | A request to run a plan.                 |

## Plan catalog

`plans` maps each plan name to a plan:

| Key             | Type             | Meaning                                                                           |
| --------------- | ---------------- | --------------------------------------------------------------------------------- |
| `name`          | string           | Plan name, as passed in a `PlanCall`.                                             |
| `module`        | qualified name   | Where the plan is defined.                                                        |
| `description`   | string           | Docstring summary.                                                                |
| `parameters`    | list             | In signature order; see below.                                                    |
| `gui_supported` | bool             | False if a required parameter cannot be entered in a form (e.g. type `any`).      |
| `ui`            | [UI hints](#ui-hints) | Optional presentation hints.                                                 |

Each parameter has:

| Key           | Type                       | Meaning                                                                                       |
| ------------- | -------------------------- | --------------------------------------------------------------------------------------------- |
| `name`        | identifier                 | Parameter name.                                                                               |
| `kind`        | string                     | `positional_only`, `positional_or_keyword`, `var_positional`, `keyword_only` or `var_keyword`. |
| `type`        | [type expression](#type-expressions) | What it accepts.                                                                    |
| `required`    | bool                       | True if it has no default.                                                                    |
| `default`     | [value](#values)           | The default, if not required.                                                                 |
| `description` | string                     | From the plan's numpydoc `Parameters` section.                                                |
| `ui`          | UI hints                   | Optional.                                                                                     |

```{literalinclude} _static/schemas/examples/plan_catalog.json
:language: json
```

## Type expressions

Every type is an object with a `kind`:

| `kind`     | Other keys                                                                                       | Python equivalent                    |
| ---------- | ------------------------------------------------------------------------------------------------ | ------------------------------------ |
| `number`   | `units`, `minimum`, `maximum`, `exclusive_minimum`, `exclusive_maximum`, `step`, `precision`     | `float`                              |
| `integer`  | `units`, `minimum`, `maximum`, `step`                                                            | `int`                                |
| `boolean`  |                                                                                                  | `bool`                               |
| `string`   | `min_length`, `max_length`, `pattern`                                                            | `str`                                |
| `literal`  | `values`                                                                                         | `Literal["a", "b"]`                  |
| `enum`     | `name`, `choices` (each a `value`, plus optional `label` and `description`)                      | an `Enum` subclass                   |
| `device`   | `protocols` (all required), `classes` (any one of), `signals`                                    | a device class or protocol           |
| `plan`     | `names`                                                                                          | a plan passed as an argument         |
| `sequence` | `items`, `min_items`, `max_items`, `unique`                                                      | `list[T]`, `Sequence[T]`             |
| `tuple`    | `items` (one type per position)                                                                  | `tuple[A, B]`                        |
| `mapping`  | `values`                                                                                         | `dict[str, T]`                       |
| `union`    | `options`                                                                                        | `A \| B`                             |
| `optional` | `of`                                                                                             | `T \| None`                          |
| `record`   | `name`, `description`, `fields`                                                                  | a dataclass; see [Records](#records) |
| `any`      | `python` (the annotation, for display)                                                           | anything else                        |

### Which devices satisfy a `device` type

A device node satisfies `{"kind": "device", ...}` if all of these hold:

1. every name in `protocols` is in the node's `protocols`;
2. if `classes` is given, at least one of them is in the node's `classes`.
   The node lists its whole class hierarchy, so this is an `isinstance` check;
3. the node is not a signal, unless `signals` is true.

The candidates are every top-level device and every descendant, addressed by
[path](#device-paths). The same check validates user input, so the dropdown
choices and the validation always agree.

### Generating types from Python

The in-process client builds type expressions from type hints
(`typing.get_type_hints`):

| Annotation                                          | Type expression                                                |
| --------------------------------------------------- | -------------------------------------------------------------- |
| `float`, `int`, `bool`, `str`                       | `number`, `integer`, `boolean`, `string`                       |
| `Literal[...]`                                      | `literal`                                                      |
| `Enum` subclass                                     | `enum`, with a choice per member, valued `member.value`         |
| runtime-checkable protocol from `bluesky.protocols` | `device` with `protocols: [name]`                              |
| ophyd or ophyd-async `Device` subclass              | `device` with `classes: [qualified name]`                      |
| `A \| B`, `Union[A, B]`                             | `union`; `optional` if one option is `None`                    |
| `list[T]`, `Sequence[T]`, `Iterable[T]`, `set[T]`   | `sequence` (`unique` for sets)                                 |
| `tuple[A, B]`, `tuple[T, ...]`                      | `tuple`, or `sequence` for the variadic form                   |
| `dict[str, T]`, `Mapping[str, T]`                   | `mapping`                                                      |
| dataclass                                           | `record`                                                       |
| `Annotated[T, ...]`                                 | `T`, with constraints taken from the metadata (see below)      |
| anything else                                       | `any`                                                          |

Constraints and units come from `typing.Annotated` metadata, so they are part
of the plan's signature, e.g.
`Annotated[float, Units("mm"), Ge(0)]` gives
`{"kind": "number", "units": "mm", "minimum": 0}`. A union of device classes
with no protocols in common becomes one `device` type with all the classes
listed, rather than a `union`.

`hextools.catalog` implements this. `plan_catalog(namespace)` and
`device_registry(namespace)` build the two documents from the plans (generator
functions) and devices in a namespace, and `write_catalog` writes both to one
YAML file, e.g. from the IPython session:

```python
from hextools.catalog import write_catalog

write_catalog(get_ipython().user_ns, "catalog.yml")
```

`Units` is `hextools.catalog.Units`; the numeric constraints are
[annotated-types](https://github.com/annotated-types/annotated-types)
(`Ge`, `Gt`, `Le`, `Lt`, `MultipleOf`, `MinLen`, `MaxLen`).

## Values

Defaults and arguments are JSON:

| Python value             | JSON value                                         |
| ------------------------ | -------------------------------------------------- |
| `None`, bool, int, float, str | the same                                      |
| enum member              | its `value`                                        |
| list, tuple, set         | array                                              |
| `dict[str, T]`           | object, with no key starting with `$`              |
| device                   | `{"$device": "<path>"}`                            |
| plan                     | `{"$plan": "<name>"}`                              |
| dataclass instance       | `{"$record": "<qualified name>", <field>: <value>, ...}` |

A default that can't be represented (e.g. one computed at import time) is left
out, and the parameter is still marked not required: the plan supplies its own
default if the argument is omitted.

A `PlanCall` names the plan and gives its `args` and `kwargs` as values:

```{literalinclude} _static/schemas/examples/plan_call.json
:language: json
```

### Device paths

A device path is a top-level name followed by any number of steps:

- `.name` for a child attribute;
- `[1]` for a `DeviceVector` child;
- `['x']` for a `DeviceMap` child.

For example: `a_slits.horizontal_gap`, `panda.pcomp[1]`. Paths are checked
against this pattern before they are used, and the in-process client only
follows steps that the device tree actually has, so no user text is ever
evaluated. A path always starts from the registry key, never from an alias.

## Records

A `record` describes a dataclass parameter, so a plan can take related inputs
(e.g. a stage and its detectors) as one argument:

```python
@dataclass(frozen=True)
class FlyscanSetup:
    """Detectors and stage for a flyscan."""

    rotation: Annotated[Movable, Locatable]
    detectors: Sequence[KinetixDetector | PhantomDetector]
    exposure: Annotated[float, Units("s"), Gt(0)] = 0.01


def tomo_flyscan(setup: FlyscanSetup, num_projections: int = 1800): ...
```

Each field has a `name`, `type`, `required`, `default`, `description` and
`ui`, like a parameter. Fields are always passed by keyword.

- Fields made with `field(init=False)` are left out.
- A field is required if it has neither a `default` nor a `default_factory`. A
  `default_factory`'s result is used as the default only if it can be
  represented as a value.
- Field descriptions come from `field(metadata={"description": ...})`, or
  the dataclass docstring's numpydoc `Attributes` section.
- Records may nest, and may appear inside `sequence`, `optional` and `union`
  types. Devices are checked at every depth.
- The class must use the `__init__` that `@dataclass` generates, since the
  record is built from its fields as keyword arguments.

Pydantic models and `TypedDict`s can be described the same way.

### Running a plan with a record

- **In process**, a record value becomes a constructor call in the generated
  cell, e.g. `FlyscanSetup(rotation=sample_tower.rotation, detectors=[kinetix1],
  exposure=0.02)`. The class is added to the IPython namespace under its
  qualified name if it isn't there already. Field names are checked as
  identifiers, and field values are serialized like any other value.
- **On a server**, the worker decodes `$record` and `$device` values before
  calling the plan. It only imports record classes that appear in the plan
  catalog, so a request cannot make it import anything else.

## Device registry

`devices` maps each top-level name to a device node. Nodes nest through
`children`:

| Key             | Type                  | Meaning                                                                                             |
| --------------- | --------------------- | --------------------------------------------------------------------------------------------------- |
| `classes`       | list of qualified names | The class hierarchy, most derived first (the MRO without `object`).                               |
| `protocols`     | list                  | The bluesky protocols it satisfies, as checked with `isinstance` on the server.                     |
| `signal`        | bool                  | True for a signal rather than a device.                                                             |
| `name`          | string                | Its own `name`, the prefix of its data keys.                                                        |
| `description`   | string                | Docstring summary.                                                                                  |
| `connected`     | bool                  | Whether it is connected.                                                                            |
| `access`        | `r`, `rw`, `w`, `x`   | Signals only.                                                                                       |
| `read_format`   | `hinted`, `read`, `config`, `none` | How its parent reads it.                                                               |
| `hinted_fields` | list                  | Data keys of its hinted fields, in definition order.                                                |
| `set_type`      | type expression       | What `set()` accepts (`Movable` nodes). Lets a client build the right setpoint editor.              |
| `datakey`       | `DataKey`             | Its own data key from `describe()`: dtype, shape, units, precision, limits, choices.                |
| `index`         | `int` or `str`        | On `DeviceVector` and `DeviceMap` nodes: children are addressed as `[1]` or `['x']`.                |
| `children`      | object                | Child name (or index, as a string) to node.                                                         |
| `aliases`       | list                  | Top-level only: other names bound to the same object.                                               |
| `ui`            | UI hints              | Optional.                                                                                           |

The registry describes structure and metadata only. Live values come through
a `ReadableDeviceSource`.

```{literalinclude} _static/schemas/examples/device_registry.json
:language: json
```

### Deltas

After the first full registry, a server sends `RegistryDelta`s. A client
applies a delta only if its `base_revision` matches the revision it holds,
and re-fetches the whole registry otherwise.

- `added` and `changed` hold whole top-level entries; `changed` replaces them.
- `removed` lists top-level names to drop.

```{literalinclude} _static/schemas/examples/registry_delta.json
:language: json
```

## UI hints

Any plan, parameter, field or device node may carry a `ui` object. Clients may
ignore any of it.

| Key        | Meaning                                                  |
| ---------- | -------------------------------------------------------- |
| `label`    | Display name, instead of one derived from `name`.        |
| `group`    | Tab or section to put it in.                             |
| `order`    | Sort order within its group.                             |
| `widget`   | Preferred editor, e.g. `slider`, `spinbox`, `file`.      |
| `hidden`   | Don't show it; use the default.                          |
| `advanced` | Show it only in an "advanced" section.                   |

## Converting from the Queue Server format

Until a server provides this format, a client can convert the Queue Server's
`allowed_plans` and `allowed_devices`. Some information is lost:

| Queue Server                                          | This schema                                                              |
| ----------------------------------------------------- | ------------------------------------------------------------------------ |
| `annotation.type` string                              | Parsed into a type expression; names it doesn't know become `any`.       |
| `annotation.devices` (type name → device names)       | `device` with `classes: [type name]`. Only the class name is known, so only devices listed for it qualify. |
| `annotation.enums` (enum name → strings)              | `enum` with those choices.                                               |
| `annotation.plans`                                    | `plan` with those `names`.                                               |
| `default` (Python source)                             | Parsed with `ast.literal_eval`; left out if that fails.                  |
| `min`, `max`, `step` (strings)                        | Numbers on the `number` or `integer` type.                               |
| `kind` (`{"name", "value"}`)                          | `kind`, lower-cased.                                                     |
| `allowed_devices` `classname`, `components`           | A node with `classes: [classname]`, `protocols` from the `is_readable`, `is_movable` and `is_flyable` flags, and `children` from `components`. |

Records, `set_type`, data keys, the class hierarchy and the other protocols
have no Queue Server equivalent.
