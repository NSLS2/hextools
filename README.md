# hextools

[![CI](https://github.com/NSLS2/hextools/actions/workflows/ci.yml/badge.svg)](https://github.com/NSLS2/hextools/actions/workflows/ci.yml) ![PyPI - Version](https://img.shields.io/pypi/v/hextools)


Tools for the NSLS-II HEX beamline at Brookhaven National Laboratory.

## Overview

`hextools` provides [ophyd-async](https://github.com/bluesky/ophyd-async) device
definitions for motors and detectors used at the HEX beamline, including:

- **Motors** — Filters, slits, monochromators, mirrors, and sample stages
- **Detectors** — GeRM and Phantom camera support

## Installation

```bash
pip install hextools
```

## Development

```bash
git clone https://github.com/NSLS2/hextools.git
cd hextools
uv sync
```

Run tests:

```bash
uv run pytest
```

## License

BSD 3-Clause. See [LICENSE](LICENSE) for details.
