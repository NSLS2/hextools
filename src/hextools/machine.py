"""NSLS-II accelerator and storage ring devices for HEX beamline."""

from ophyd_async.core import SignalR, StandardReadable, StrictEnum, SignalDatatypeT
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice, epics_signal_r
from typing import TypeVar


from bluesky.suspenders import SuspendFloor

class NSLS2OpsMode(StrictEnum):
    OPERATIONS = "Operations"
    SETUP = "Setup"
    ACCEL_STUDIES = "Accel Studies"
    BEAMLINE_STUDIES = "Beamline Studies"
    FAILURE = "Failure"
    MAINTENANCE = "Maintenance"
    SHUTDOWN = "Shutdown"
    UNSCHEDULED_OPS = "Unscheduled Ops"
    DECAY_MODE_OPS = "Decay Mode Ops"



class NSLS2StorageRing(StandardReadable, EpicsDevice):
    """NSLS-II storage ring device."""

    def __init__(self):
        self.operating_mode = epics_signal_r(NSLS2OpsMode, "SR-OPS{}Mode-Sts")
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.beam_current = epics_signal_r(float, "SR:OPS-BI{DCCT:1}I:Real-I")
        super().__init__()
