"""Shutter/GV device for the photon delivery system."""

from ophyd_async.core import AsyncMovable, AsyncStatus, wait_for_value
from ophyd_async.epics.core import (
    EpicsDevice,
    epics_signal_r,
    epics_triggerable_command,
)


class Shutter(EpicsDevice, AsyncMovable[bool]):
    """Photon shutter device."""

    def __init__(self, prefix: str, name: str = ""):

        super().__init__(prefix, name=name)
        self.status = epics_signal_r(bool, f"{prefix}Pos-Sts")
        self.open_cmd = epics_triggerable_command(f"{prefix}Cmd:Opn-Cmd")
        self.close_cmd = epics_triggerable_command(f"{prefix}Cmd:Cls-Cmd")

    @AsyncStatus.wrap
    async def set(self, value: bool):
        if value:
            cmd_sig = self.open_cmd
        else:
            cmd_sig = self.close_cmd

        await cmd_sig.execute()
        await wait_for_value(self.status, value, timeout=10)
