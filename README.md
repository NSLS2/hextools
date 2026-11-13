# hextools

[![CI](https://github.com/NSLS2/hextools/actions/workflows/ci.yml/badge.svg)](https://github.com/NSLS2/hextools/actions/workflows/ci.yml) ![PyPI - Version](https://img.shields.io/pypi/v/hextools)


`hextools` is the data acquisition package for the **27-ID-1 HEX** (High Energy
Engineering X-ray scattering) beamline at NSLS-II, Brookhaven National Laboratory.

## Overview

- **Devices** — [ophyd-async](https://blueskyproject.io/ophyd-async) classes for
  the beamline's shutters, slits, monochromator, filters, motor stages, and
  detectors (Kinetix, Phantom, GeRM, Perkin Elmer, Vimba cameras, PandABox).
- **Plans** — [bluesky](https://blueskyproject.io/bluesky) plans for tomography,
  radiography, energy-dispersive (EDXD) and angle-dispersive (XRD) diffraction,
  and beamline setup (beam mode, energy, motor moves).
- **Collection profile** — an IPython startup script that builds the RunEngine,
  connects every device, and sends data to
  [Tiled](https://blueskyproject.io/tiled).
- **HEX Data Acquisition GUI** — a Qt application that generates a form for each
  plan and runs it either in-process or through the Bluesky Queue Server.
- **Exporters** — automatic export of acquired data from Tiled into beamline
  file formats.

## Installation

The project is managed with [Pixi](https://pixi.sh), which solves and locks all
dependencies, including the conda-only EPICS stack:

```bash
git clone https://github.com/NSLS2/hextools.git
cd hextools
pixi install
```

Python 3.11–3.13 are supported and tested.

## Usage

Start an IPython session with the collection profile loaded:

```bash
pixi run start            # against real beamline hardware
pixi run start-mock       # against simulated devices
```

Start the data acquisition GUI:

```bash
pixi run hex-gui          # against real beamline hardware
pixi run hex-gui-mock     # against simulated devices
```

The GUI can also be launched directly, which exposes additional options:

```bash
pixi run -e dev python -m hextools.gui [--mock] [--profile NAME] [--queueserver-uri URI] [--history-file PATH]
```

Queue Server components and the auto-exporter service have their own tasks:

```bash
pixi run qs-backend       # start-re-manager
pixi run qs-server        # bluesky-httpserver
pixi run auto-exporter    # Tiled auto-export service
```

## Development

```bash
pixi install -e dev
```

Common tasks:

```bash
pixi run tests            # unit tests (excluding GUI)
pixi run gui-tests        # GUI tests, run separately to avoid Qt teardown issues
pixi run all-tests        # run all unit tests
pixi run tests-cov        # run all unit tests with coverage
pixi run lint             # pre-commit (ruff, formatting, import-linter)
pixi run type-check       # pyright
```

Build the documentation:

```bash
pixi run -e docs build-docs           # output in docs/_build/html
pixi run -e docs build-docs true      # live-reloading preview
```

## Documentation

- [Getting started](docs/getting_started.md) — environments and installation
- [Command line](docs/cli.md) — running plans from IPython
- [GUI](docs/gui.md) — running plans from the Qt application
- [Collection profile](docs/profile.md) — what the startup script sets up
- [Developer guide](docs/developer.md) — repository layout and plan conventions

## License

BSD 3-Clause. See [LICENSE](LICENSE) for details.
