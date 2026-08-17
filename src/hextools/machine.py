from ophyd_async.core import StandardReadable
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice, epics_signal_r


class NSLS2StorageRing(StandardReadable, EpicsDevice):
    def __init__(self):
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.beam_current = epics_signal_r(float, "SR:OPS-BI{DCCT:1}I:Real-I")
