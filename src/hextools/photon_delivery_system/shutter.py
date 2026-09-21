"""Shutter/GV device for the photon delivery system."""

from typing import Hashable

from ophyd_async.core import AsyncMovable, AsyncStatus, StrictEnum, wait_for_value
from bluesky import plan_stubs as bps
from ophyd_async.epics.core import (
    EpicsDevice,
    epics_signal_r,
    epics_triggerable_command,
)

class ShutterStatus(StrictEnum):
    OPEN = "Open"
    CLOSED = "Not Open"

class Shutter(EpicsDevice, AsyncMovable[bool]):
    """Photon shutter device.

    Attributes
    ----------
    status : EpicsSignalRO[bool]
        Readback of the shutter status (True for open, False for closed)
    open_cmd : EpicsTriggerableCommand
        Command to open the shutter
    close_cmd : EpicsTriggerableCommand
        Command to close the shutter
    """

    def __init__(self, prefix: str, name: str = ""):

        super().__init__(prefix, name=name)
        self.status = epics_signal_r(ShutterStatus, f"{prefix}Pos-Sts")
        self.open_cmd = epics_triggerable_command(f"{prefix}Cmd:Opn-Cmd")
        self.close_cmd = epics_triggerable_command(f"{prefix}Cmd:Cls-Cmd")

    @AsyncStatus.wrap
    async def set(self, value: bool):
        """Set the state of the shutter.

        Parameters
        ----------
        value : bool
            The desired state of the shutter (True for open, False for closed)

        Returns
        -------
        AsyncStatus
            An object representing the status of the set operation.
        """

        if value:
            cmd_sig = self.open_cmd
        else:
            cmd_sig = self.close_cmd

        await cmd_sig.execute()
        await wait_for_value(self.status, ShutterStatus.OPEN if value else ShutterStatus.CLOSED, timeout=10)


def ensure_shutter_state(
    shutter: Shutter,
    desired_state: bool,
    allow_actuation: bool = False,
    group: Hashable | None = None,
    wait: bool = True,
):
    """Plan stub to guarantees that the shutter is in the desired state.

    Parameters
    ----------
    shutter : Shutter
        shutter to guarantee the state of.
    desired_state : bool
        the state of the shutter (True for open, False for closed)
    allow_actuation : bool, default False
        whether to allow the plan to actuate the shutter if it is not in the desired state
    group : Hashable | None, optional
        the Bluesky group to use for the actuation, if any
    wait : bool, default True
        whether to wait for the shutter to reach the desired state after actuation
    """

    shutter_status = yield from bps.rd(shutter.status)
    if shutter_status != desired_state:
        if allow_actuation:
            yield from bps.abs_set(shutter, desired_state, group=group, wait=wait)
        else:
            raise RuntimeError(f"Shutter {shutter.name} is not in the desired state!")


def ensure_shutter_open(
    shutter: Shutter,
    allow_actuation: bool = False,
    group: Hashable | None = None,
    wait: bool = True,
):
    """Plan stub to guarantee that the shutter is open.

    Parameters
    ----------
    shutter : Shutter
        shutter to guarantee the state of.
    allow_actuation : bool, default False
        whether to allow the plan to actuate the shutter if it is not in the desired state
    group : Hashable | None, optional
        the Bluesky group to use for the actuation, if any
    wait : bool, default True
        whether to wait for the shutter to reach the desired state after actuation
    """

    yield from ensure_shutter_state(
        shutter, True, allow_actuation=allow_actuation, wait=wait, group=group
    )


def ensure_shutter_closed(
    shutter: Shutter,
    allow_actuation: bool = True,
    group: Hashable | None = None,
    wait: bool = True,
):
    """Plan stub to guarantee that the shutter is closed.

    Parameters
    ----------
    shutter : Shutter
        shutter to guarantee the state of.
    allow_actuation : bool, default True
        whether to allow the plan to actuate the shutter if it is not in the desired state
    group : Hashable | None, optional
        the Bluesky group to use for the actuation, if any
    wait : bool, default True
            whether to wait for the shutter to reach the desired state after actuation
    """

    yield from ensure_shutter_state(
        shutter, False, allow_actuation=allow_actuation, wait=wait, group=group
    )
