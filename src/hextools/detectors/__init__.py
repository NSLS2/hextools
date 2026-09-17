"""Ophyd async support for detectors at HEX."""

# Readout headroom (s) added to exposure_time when frame_period is unset;
# same margin the beamline's deployed PandA plan kept between step and exposure.
FRAME_PERIOD_MARGIN = 0.0125 #Kinetix
                    # 0.000005 #TODO for phantom
                    # GeRM N/A
                    # 0.05 # PE

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

__all__ = [
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
