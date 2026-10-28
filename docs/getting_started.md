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
| `terminal`     | The IPython command-line session on the beamline (`pixi run start`).       |
| `gui`          | The HEX Data Acquisition GUI against real hardware (`pixi run hex-gui`).   |
| `qs`           | The Bluesky Queue Server backend and HTTP server.                          |
| `dev`          | Development tools, plus the terminal and GUI with simulated devices (`pixi run start-mock`, `pixi run hex-gui-mock`). |

Pixi picks the environment that defines a task automatically, so in most cases
`pixi run <task>` is all you need. Pass `-e <environment>` to choose one
explicitly.

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
pixi run hex-gui               # real hardware, plans run in an IPython session
pixi run hex-gui-mock          # simulated devices

# Queue Server
pixi run qs-backend            # RE Manager
pixi run qs-server             # HTTP server on localhost:60610
pixi run qserver-gui           # upstream bluesky-widgets queue monitor

# Development
pixi run -e py313 tests
pixi run -e py313 lint
pixi run -e py313 type-check
```
