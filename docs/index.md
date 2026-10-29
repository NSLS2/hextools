# hextools

`hextools` is the data acquisition package for the **27-ID-1 HEX** (High Energy
Engineering X-ray scattering) beamline at NSLS-II. It provides:

- [ophyd-async](https://blueskyproject.io/ophyd-async) device classes for the
  beamline's shutters, slits, monochromator, motor stages, filters and
  detectors (Kinetix, Phantom, GeRM, Perkin Elmer, Vimba cameras, PandABox);
- [bluesky](https://blueskyproject.io/bluesky) plans for tomography,
  radiography, energy-dispersive (EDXD) and angle-dispersive (XRD) diffraction,
  and beamline setup (beam mode, energy, motor moves);
- the **collection profile**, an IPython startup script that builds the
  RunEngine, connects every device and sends data to
  [Tiled](https://blueskyproject.io/tiled);
- the **HEX Data Acquisition GUI**, a Qt application that generates a form for
  each plan and runs it either in-process or through the Bluesky Queue Server.

```{toctree}
:maxdepth: 2
:caption: User guide

getting_started
cli
gui
profile
```

```{toctree}
:maxdepth: 2
:caption: Developer guide

developer
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
