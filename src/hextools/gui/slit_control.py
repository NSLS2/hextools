"""Slit Control tab: front-end Slit1 aperture plan and its live position."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import IPython
from bluesky import RunEngine
from bluesky_widgets.models.run_engine_client import RunEngineClient
from qtpy.QtCore import Signal, Slot
from qtpy.QtWidgets import QFormLayout, QGroupBox, QLabel, QVBoxLayout, QWidget

from hextools.gui.plan_widget import QtPlanWidget
from hextools.photon_delivery_system.fe_slit import (
    FrontEndSlit,
    set_fe_slit1_aperture,
)


class _SlitReadout(QGroupBox):
    """Live Y position and aperture of one front-end slit."""

    _signal_y = Signal(float)
    _signal_aperture = Signal(str)

    def __init__(
        self, var_name: str, slit: FrontEndSlit, loop: asyncio.AbstractEventLoop
    ):
        super().__init__(var_name)
        self.slit = slit
        self._loop = loop
        self.y_label = QLabel("--")
        self.aperture_label = QLabel("--")
        form = QFormLayout()
        form.addRow("Y (mm)", self.y_label)
        form.addRow("Aperture", self.aperture_label)
        self.setLayout(form)
        self._signal_y.connect(self._set_y)
        self._signal_aperture.connect(self._set_aperture)
        # Subscriptions run on the RunEngine's loop, like the shutter indicators
        loop.call_soon_threadsafe(slit.y.user_readback.subscribe_reading, self._on_y)
        loop.call_soon_threadsafe(slit.aperture.subscribe_reading, self._on_aperture)

    def _on_y(self, readings: Mapping[str, Any]):
        for reading in readings.values():
            self._signal_y.emit(float(reading["value"]))

    def _on_aperture(self, readings: Mapping[str, Any]):
        for reading in readings.values():
            self._signal_aperture.emit(str(reading["value"]))

    @Slot(float)
    def _set_y(self, y: float):
        self.y_label.setText(f"{y:.3f}")

    @Slot(str)
    def _set_aperture(self, aperture: str):
        self.aperture_label.setText(aperture)

    def unsubscribe(self):
        def _clear():
            for signal, callback in (
                (self.slit.y.user_readback, self._on_y),
                (self.slit.aperture, self._on_aperture),
            ):
                try:
                    signal.clear_sub(callback)
                except KeyError:
                    pass

        self._loop.call_soon_threadsafe(_clear)


class QtFrontEndSlitStatus(QWidget):
    """Live readouts for every :class:`FrontEndSlit` in ``namespace``.

    Parameters
    ----------
    re : RunEngine
        The in-process RunEngine.
    namespace : Mapping[str, object]
        Namespace searched for slits, typically IPython's ``user_ns``.
    """

    def __init__(self, re: RunEngine, namespace: Mapping[str, Any], parent=None):
        super().__init__(parent)
        slits: dict[int, tuple[str, FrontEndSlit]] = {}
        for name, obj in sorted(namespace.items()):
            if not name.startswith("_") and isinstance(obj, FrontEndSlit):
                slits.setdefault(id(obj), (name, obj))
        self.readouts = [
            _SlitReadout(name, slit, re.loop) for name, slit in slits.values()
        ]
        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        for readout in self.readouts:
            vbox.addWidget(readout)
        self.setLayout(vbox)

    def closeEvent(self, event):
        for readout in self.readouts:
            readout.unsubscribe()
        super().closeEvent(event)


class QtSlitControl(QWidget):
    """The aperture plan for front-end Slit1, with its live position below.

    The live readout needs the in-process RunEngine (local mode); with the
    Queue Server only the plan form is shown.
    """

    def __init__(self, re_client: RunEngineClient | RunEngine, parent=None):
        super().__init__(parent)
        vbox = QVBoxLayout()
        vbox.addWidget(QtPlanWidget(re_client, set_fe_slit1_aperture))
        ipython = IPython.get_ipython()
        if isinstance(re_client, RunEngine) and ipython is not None:
            self.status = QtFrontEndSlitStatus(re_client, ipython.user_ns)
            vbox.addWidget(self.status)
        vbox.addStretch()
        self.setLayout(vbox)
