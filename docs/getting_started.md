# Getting started

## Prerequisites

`hextools` is managed with [pixi](https://pixi.sh). Pixi creates isolated
environments from `pixi.toml` and pins every dependency in `pixi.lock`, so the
same commands work on a beamline workstation and on a laptop.

```bash
git clone https://github.com/NSLS2/hextools.git
cd hextools
pixi install          # solves and installs the default environment
```

## Environments

`pixi.toml` defines several environments. The ones you will use day to day are:

| Environment    | Use it for                                                                 |
| -------------- | -------------------------------------------------------------------------- |
| `terminal`     | The IPython command-line session (`pixi run start`) and the HEX Data Acquisition GUI (`pixi run hex-gui`) on the beamline. |
| `qs`           | The Bluesky Queue Server backend and HTTP server.                          |
| `dev`          | Development tools, plus the terminal and GUI with simulated devices (`pixi run start-mock`, `pixi run hex-gui-mock`). |
| `exporter`     | The standalone Tiled auto-export service (`pixi run -e exporter auto-exporter`). |

Pixi picks a default environment for each task, so in most cases
`pixi run <task>` is all you need. Pass `-e <environment>` to choose one
explicitly.

### Installed launchers and environment changes

Use the named Pixi task instead of hard-coding an interpreter under `.pixi/envs`.
PR [#109](https://github.com/NSLS2/hextools/pull/109) consolidated the GUI into
`terminal`/`dev` and removed the separate `gui` environment. A site-installed
wrapper still pointing at `.pixi/envs/gui/bin/python` is not compatible with this
manifest. For the standalone exporter, the supported command is:

```bash
pixi run --manifest-path /opt/bluesky/hextools/pixi.toml --locked -e exporter auto-exporter
```

`/usr/local/bin/exporter` is not installed or defined by this repository.
Its deployment owner must inspect/update that wrapper; do not recreate an
obsolete environment as a workaround. The GUI's **NXS Export** panel instead
starts `python -m hextools.exporters.auto_exporter` with the GUI's own interpreter.
Do not delete or replace an environment while processes still use it.

A reported `p4p.cleanup` traceback (`No module named p4p.server`) at interpreter
exit is a **separate observation**, not proof of the launcher problem's cause.
The locked development environment includes `p4p.server` and exits cleanly.
An incomplete/changed environment, changed import paths, or shutdown-time import
ordering remain possible on the reporting workstation until measured.

An operator can run these **local, read-only** diagnostics on the affected
workstation. Redact credentials or tokens from wrapper contents before sharing
output. These commands do not execute the wrapper, create environments, or
connect to beamline systems:

```bash
readlink -f /usr/local/bin/exporter
sed -n '1,12p' /usr/local/bin/exporter
git -C /opt/bluesky/hextools rev-parse HEAD
git -C /opt/bluesky/hextools status --short -- pixi.toml pixi.lock
grep -nE '^\[environments|^\[feature.exporter|^auto-exporter|^terminal =|^dev =|^gui =' /opt/bluesky/hextools/pixi.toml
ls -ld /opt/bluesky/hextools/.pixi/envs/{gui,terminal,dev,exporter}/bin/python
```

For **the exact interpreter that emitted the p4p traceback**, if it still
exists, substitute its absolute path below (not a different environment):

```bash
env EPICS_CA_AUTO_ADDR_LIST=NO EPICS_CA_ADDR_LIST=127.0.0.1 \
    EPICS_PVA_AUTO_ADDR_LIST=NO EPICS_PVA_ADDR_LIST=127.0.0.1 \
    /path/to/that/python -c 'import importlib.util, os, sys; print(sys.executable); print(sys.prefix); print("PYTHONPATH", os.environ.get("PYTHONPATH")); print(sys.path); print(importlib.util.find_spec("p4p")); print(importlib.util.find_spec("p4p.server"))'
```

Also retain the full original traceback, launch command, and whether an
environment update/removal was running at that time. A successful import in a
new interpreter cannot establish what an already-exiting process could see.

## Mock mode

Every entry point can run without beamline hardware. Mock mode is switched on
by the `HEXTOOLS_RUNNING_IN_CI` environment variable (set for you by the
`start-mock` and `hex-gui-mock` tasks, or by `--mock`). In mock mode:

- devices connect with ophyd-async's mock backends instead of EPICS;
- the RunEngine metadata is an in-memory dictionary holding a mock proposal
  (`pass-123456`) instead of the Redis-backed beamline metadata;
- data is written to a temporary local Tiled server instead of
  `tiled.nsls2.bnl.gov`;
- proposal changes accept any credentials.

Mock mode is the safest way to try a plan, learn the GUI, or develop new code.

## Quick reference

```bash
# Command line (IPython) session
pixi run start                 # real hardware
pixi run start-mock            # simulated devices

# GUI
pixi run hex-gui               # real hardware, plans run in-process
pixi run hex-gui-mock          # simulated devices

# Queue Server
pixi run qs-backend            # RE Manager
pixi run qs-server             # HTTP server on localhost:60610
pixi run qserver-gui           # upstream bluesky-widgets queue monitor

# Development
pixi run tests
pixi run lint
pixi run type-check
```
