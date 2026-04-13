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
