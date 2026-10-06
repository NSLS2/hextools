"""RunEngine suspenders for the HEX beamline."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import Any

from bluesky.protocols import Subscribable
from bluesky.suspenders import SuspendFloor


def _call_on_loop(loop, func: Callable[[], None]) -> None:
    """Run ``func`` on the RunEngine's event loop and wait for it."""
    if threading.get_ident() == getattr(loop, "_thread_id", None):
        func()
        return
    done = threading.Event()
    errors: list[BaseException] = []

    def call():
        try:
            func()
        except BaseException as err:
            errors.append(err)
        finally:
            done.set()

    loop.call_soon_threadsafe(call)
    done.wait()
    if errors:
        raise errors[0]


class SuspendFloorUnlessOpsMode(SuspendFloor):
    """Suspend when a signal falls below a threshold, unless in a skipped ring mode.

    Install it whatever the ring's mode: the mode is checked every time the
    suspender is evaluated, so a ring entering maintenance releases a tripped
    suspender, and a ring returning to operations with beam still low trips it.
    Until the mode has been read, the suspender acts like a plain `SuspendFloor`.

    Parameters
    ----------
    signal : Signal
        The signal to watch, e.g. the storage ring beam current.
    suspend_thresh : float
        Suspend if the signal falls below this value.
    ops_mode : Signal
        The ring's operating mode. Like ``signal``, either an ophyd-async signal
        or one following ophyd's subscription pattern.
    skip_modes : Iterable
        Modes in which a low signal does not suspend the RunEngine.
    **kwargs
        Passed to `SuspendFloor` (``resume_thresh``, ``sleep``, ...).
    """

    def __init__(
        self,
        signal,
        suspend_thresh: float,
        *,
        ops_mode,
        skip_modes: Iterable[Any],
        **kwargs,
    ):
        super().__init__(signal, suspend_thresh, **kwargs)
        self._ops_mode = ops_mode
        self._skip_modes = frozenset(skip_modes)
        self._mode = None
        self._mode_cid = None
        # The base _lock is not reentrant, and _set_mode calls back into __call__.
        self._eval_lock = threading.RLock()

    @property
    def _skipping(self) -> bool:
        return self._mode in self._skip_modes

    def _should_suspend(self, value):
        return not self._skipping and super()._should_suspend(value)

    def _should_resume(self, value):
        return self._skipping or super()._should_resume(value)

    def _on_mode_reading(self, reading):
        self._set_mode(reading[self._ops_mode.name]["value"])

    def _on_mode_value(self, value=None, **kwargs):
        self._set_mode(value)

    def __call__(self, value, **kwargs):
        with self._eval_lock:
            super().__call__(value, **kwargs)

    def _set_mode(self, mode):
        with self._eval_lock:
            self._mode = mode
            if self._last_value is None:
                return
            value = self._last_value
            if self._implements_protocol:
                value = {self._sig.name: {"value": value}}
            self(value=value)

    def install(self, RE, *, event_type=None):
        # The mode first, so the first beam reading is judged against it.
        if isinstance(self._ops_mode, Subscribable):
            _call_on_loop(
                RE.loop, lambda: self._ops_mode.subscribe_reading(self._on_mode_reading)
            )
        else:
            self._mode_cid = self._ops_mode.subscribe(self._on_mode_value, run=True)
        super().install(RE, event_type=event_type)

    def remove(self):
        RE = self.RE
        super().remove()
        if self._mode_cid is not None:
            self._ops_mode.unsubscribe(self._mode_cid)
            self._mode_cid = None
        elif RE is not None and isinstance(self._ops_mode, Subscribable):
            _call_on_loop(
                RE.loop, lambda: self._ops_mode.clear_sub(self._on_mode_reading)
            )

    def _get_justification(self):
        just = super()._get_justification()
        if just and self._mode is not None:
            just = f"{just} (ring mode: {getattr(self._mode, 'value', self._mode)})"
        return just
