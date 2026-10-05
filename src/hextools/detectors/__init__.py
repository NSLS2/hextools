"""Ophyd async support for detectors at HEX."""

from .phantom import (
    PhantomAuxPinMode,
    PhantomDetector,
    PhantomDownloadFrameMode,
    PhantomDownloadSpeed,
    PhantomExtSyncType,
    PhantomFanState,
    PhantomPixelDataFormat,
    PhantomReadySignal,
    PhantomSettingsSlot,
    PhantomTrigEdge,
)

# Readout headroom (s) added to exposure_time when frame_period is unset; the
# same margin the beamline's deployed PandA plan kept between step and exposure.
#
# TODO: this is ONE constant carrying the Kinetix value while three other
# detectors want their own (phantom 0.000005, GeRM not applicable, PE 0.05).
# Whichever plan imports it gets the Kinetix number regardless of the camera.
FRAME_PERIOD_MARGIN = 0.0125

__all__ = [
    "FRAME_PERIOD_MARGIN",
    "PhantomDetector",
    "PhantomAuxPinMode",
    "PhantomExtSyncType",
    "PhantomDownloadFrameMode",
    "PhantomDownloadSpeed",
    "PhantomFanState",
    "PhantomPixelDataFormat",
    "PhantomReadySignal",
    "PhantomTrigEdge",
    "PhantomSettingsSlot",
]
