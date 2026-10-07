"""Shutter/GV device for the photon delivery system."""

import asyncio
import random
from typing import Callable, Hashable

from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    AsyncMovable,
    AsyncStatus,
    DeviceMock,
    StrictEnum,
    callback_on_mock_execute,
    default_mock_class,
    set_mock_value,
    wait_for_value,
)
from bluesky.protocols import Reading, Subscribable
from bluesky import plan_stubs as bps
from ophyd_async.epics.core import (
    EpicsDevice,
    epics_signal_r,
    epics_triggerable_command,
)

DEFAULT_SHUTTER_TIMEOUT = 10

class ShutterStatus(StrictEnum):
    OPEN = "Open"
    CLOSED = "Not Open"


class MockShutter(DeviceMock["Shutter"]):
    """Mock shutter that flips status after a random 0-3 s actuation delay."""

    async def connect(self, device: "Shutter") -> None:
        set_mock_value(device.status, ShutterStatus.CLOSED)

        async def _actuate(target: ShutterStatus) -> None:
            await asyncio.sleep(random.uniform(0.0, 3.0))
            set_mock_value(device.status, target)

        async def _open() -> None:
            await _actuate(ShutterStatus.OPEN)

        async def _close() -> None:
            await _actuate(ShutterStatus.CLOSED)

        callback_on_mock_execute(device.open_cmd, _open)
        callback_on_mock_execute(device.close_cmd, _close)


@default_mock_class(MockShutter)
class Shutter(EpicsDevice, Subscribable[bool], AsyncMovable[bool]):
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
        # Maps a user callback to the status-signal wrapper registered for it.
        self._sub_translators: dict[
            Callable[[dict[str, Reading[bool]]], None],
            Callable[[dict[str, Reading[ShutterStatus]]], None],
        ] = {}

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
        await wait_for_value(self.status, ShutterStatus.OPEN if value else ShutterStatus.CLOSED, timeout=DEFAULT_SHUTTER_TIMEOUT)


    def subscribe_reading(self, function: Callable[[dict[str, Reading[bool]]], None]) -> None:
        """Subscribe to changes in the shutter status.

        Parameters
        ----------
        function : Callable[[dict[str, Reading[bool]]], None]
            A function to call with the new shutter status (as a dictionary of readings)
        """

        def _translate(readings: dict[str, Reading[ShutterStatus]]) -> None:
            """Translate readings from ShutterStatus to bool."""

            function(
                {
                    self.name: {
                        **reading,
                        "value": reading["value"] == ShutterStatus.OPEN,
                    }
                    for reading in readings.values()
                }
            )

        self._sub_translators[function] = _translate
        self.status.subscribe_reading(_translate)


    def clear_sub(self, function: Callable[[dict[str, Reading[bool]]], None]) -> None:
        """Remove a subscription previously passed to `subscribe_reading`.

        Parameters
        ----------
        function : Callable[[dict[str, Reading[bool]]], None]
            The callback to remove.
        """

        translator = self._sub_translators.pop(function)
        self.status.clear_sub(translator)


def ensure_shutter_state(
    shutter: Shutter | list[Shutter],
    desired_state: bool,
    allow_actuation: bool = False,
    group: Hashable | None = None,
    wait: bool = True,
):
    """Plan stub to guarantees that the shutter(s) is/are in the desired state.

    Parameters
    ----------
    shutter : Shutter | list[Shutter]
        shutter(s) to guarantee the state of.
    desired_state : bool
        the state of the shutter (True for open, False for closed)
    allow_actuation : bool, default False
        whether to allow the plan to actuate the shutter if it is not in the desired state
    group : Hashable | None, optional
        the Bluesky group to use for the actuation, if any
    wait : bool, default True
        whether to wait for the shutter to reach the desired state after actuation
    """

    shutters = shutter if isinstance(shutter, list) else [shutter]

    shutter_statuses = {}
    for s in shutters:
        shutter_statuses[s.name] = yield from bps.rd(s.status)

    actuated = False
    for s in shutters:
        if (shutter_statuses[s.name] == ShutterStatus.OPEN) != desired_state:
            if allow_actuation:
                yield from bps.abs_set(s, desired_state, group=group, wait=False)
                actuated = True
            else:
                raise RuntimeError(f"Shutter {s.name} is not in the desired state!")

    if wait and actuated:
        yield from bps.wait(group=group, timeout=DEFAULT_SHUTTER_TIMEOUT)


def ensure_shutter_open(
    shutter: Shutter | list[Shutter],
    allow_actuation: bool = False,
    group: Hashable | None = None,
    wait: bool = True,
):
    """Plan stub to guarantee that the shutter(s) is/are open.

    Parameters
    ----------
    shutter : Shutter | list[Shutter]
        shutter(s) to guarantee the state of.
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
    shutter: Shutter | list[Shutter],
    allow_actuation: bool = True,
    group: Hashable | None = None,
    wait: bool = True,
):
    """Plan stub to guarantee that the shutter is closed.

    Parameters
    ----------
    shutter : Shutter | list[Shutter]
        shutter(s) to guarantee the state of.
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