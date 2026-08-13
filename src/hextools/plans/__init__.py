"""
HEX beamline Bluesky plans.

Ported from NSLS2/hex-ob ``plans/``, the staging ground where each legacy
pyepics script family is rewritten and proven against the mock + simulated
beamline tiers before promotion here.  Detectors and motors are passed as
arguments (configure, don't copy): the plans drive the devices in
``hextools.detectors``.

Currently the tomography family (the Kinetix script generation).  The
Phantom plan family stays in hex-ob until it is reconciled with this
package's ``PhantomDetector`` surface.
"""

from . import tomography
from .shutter import close_photon_shutter, open_photon_shutter

__all__ = [
    "close_photon_shutter",
    "open_photon_shutter",
    "tomography",
]
