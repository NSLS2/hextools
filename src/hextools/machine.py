"""NSLS-II accelerator and storage ring devices for HEX beamline."""

from ophyd_async.core import StandardReadable, DeviceMock, set_mock_value, default_mock_class
from ophyd_async.core import StandardReadable, StrictEnum
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice, epics_signal_r


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



class MockStorageRing(DeviceMock["NSLS2StorageRing"]):
    """Mock storage ring device.

    Sets the beam current to 450 mA, to not trip a suspender by default in mock mode.
    """

    async def connect(self, device: "NSLS2StorageRing"):
        set_mock_value(device.beam_current, 450)
        set_mock_value(device.operating_mode, NSLS2OpsMode.OPERATIONS)


@default_mock_class(MockStorageRing)
class NSLS2StorageRing(StandardReadable, EpicsDevice):
    """NSLS-II storage ring device.

    Attributes
    ----------
    operating_mode : SignalR[NSLS2OpsMode]
        The current operating mode of the storage ring.
    beam_current : SignalR[float]
        The current beam current of the storage ring.
    """

    def __init__(self):
        self.operating_mode = epics_signal_r(NSLS2OpsMode, "SR-OPS{}Mode-Sts")
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.beam_current = epics_signal_r(float, "SR:OPS-BI{DCCT:1}I:Real-I")
        super().__init__()
