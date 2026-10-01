"""NSLS-II accelerator and storage ring devices for HEX beamline."""

from ophyd_async.core import StandardReadable, DeviceMock, set_mock_value, default_mock_class
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice, epics_signal_r


class MockStorageRing(DeviceMock["NSLS2StorageRing"]):
    """Mock storage ring device.

    Sets the beam current to 450 mA, to not trip a suspender in mock mode.
    """

    async def connect(self, device: "NSLS2StorageRing"):
        set_mock_value(device.beam_current, 450)


@default_mock_class(MockStorageRing)
class NSLS2StorageRing(StandardReadable, EpicsDevice):
    """NSLS-II storage ring device."""

    def __init__(self):
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.beam_current = epics_signal_r(float, "SR:OPS-BI{DCCT:1}I:Real-I")
        super().__init__()
