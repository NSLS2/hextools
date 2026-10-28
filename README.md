# hextools

[![CI](https://github.com/NSLS2/hextools/actions/workflows/ci.yml/badge.svg)](https://github.com/NSLS2/hextools/actions/workflows/ci.yml) ![PyPI - Version](https://img.shields.io/pypi/v/hextools)


Tools for the NSLS-II HEX beamline at Brookhaven National Laboratory.

## Overview

`hextools` provides [ophyd-async](https://github.com/bluesky/ophyd-async) device
definitions for motors and detectors used at the HEX beamline, including:

- **Motors** — Filters, slits, monochromators, mirrors, and sample stages
- **Detectors** — GeRM and Phantom camera support

The collection profile exposes a `SampleSmall` stage as `sample_small`, with
ordinary ophyd-async motor components `sample_small.x` and `sample_small.y`.
These replace the standalone `sample_small_x` and `sample_small_y` identifiers.
The GUI displays the axes as **Sample Small X** and **Sample Small Y** while plan
submission retains their dotted Python paths. Their motor records are
`XF:27IDF-OP:1{MC:3-Ax:SampleSX}Mtr` and
`XF:27IDF-OP:1{MC:3-Ax:SampleSY}Mtr`, respectively (from the legacy HEX MCF3 screen).

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
