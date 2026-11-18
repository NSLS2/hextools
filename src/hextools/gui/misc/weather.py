"""A small Qt widget showing the date, time, and current weather for a location.

The clock (date and time) updates every second from the local machine, using the
location's timezone so it matches the site. Weather (temperature and a
short condition) is fetched periodically from the free Open-Meteo API, which
needs no API key. Display is best effort: a failed request leaves the previous
reading in place and shows a placeholder until the next successful fetch.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from qtpy.QtCore import Qt, QTimer, Signal, Slot
from qtpy.QtWidgets import QFormLayout, QGroupBox, QLabel, QVBoxLayout, QWidget

from hextools.gui._threading import FunctionWorker


@dataclass(frozen=True)
class Location:
    """A named site for the weather widget.

    ``timezone`` is an IANA name (e.g. ``"America/New_York"``).
    """

    name: str
    latitude: float
    longitude: float
    timezone: str

    @property
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


# Brookhaven National Laboratory.
UPTON_NY = Location(
    name="Upton, NY",
    latitude=40.8677,
    longitude=-72.8814,
    timezone="America/New_York",
)

_PLACEHOLDER = "\u2014"  # em dash

# Unit letter -> (Open-Meteo temperature_unit, display suffix).
_TEMPERATURE_UNITS: dict[str, tuple[str, str]] = {
    "F": ("fahrenheit", "\u00b0F"),
    "C": ("celsius", "\u00b0C"),
}

_API_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes -> (description, emoji).
_WMO_CODES: dict[int, tuple[str, str]] = {
    0: ("Clear sky", "\u2600\ufe0f"),
    1: ("Mainly clear", "\U0001f324\ufe0f"),
    2: ("Partly cloudy", "\u26c5"),
    3: ("Overcast", "\u2601\ufe0f"),
    45: ("Fog", "\U0001f32b\ufe0f"),
    48: ("Rime fog", "\U0001f32b\ufe0f"),
    51: ("Light drizzle", "\U0001f327\ufe0f"),
    53: ("Drizzle", "\U0001f327\ufe0f"),
    55: ("Dense drizzle", "\U0001f327\ufe0f"),
    56: ("Freezing drizzle", "\U0001f327\ufe0f"),
    57: ("Freezing drizzle", "\U0001f327\ufe0f"),
    61: ("Light rain", "\U0001f327\ufe0f"),
    63: ("Rain", "\U0001f327\ufe0f"),
    65: ("Heavy rain", "\U0001f327\ufe0f"),
    66: ("Freezing rain", "\U0001f327\ufe0f"),
    67: ("Freezing rain", "\U0001f327\ufe0f"),
    71: ("Light snow", "\U0001f328\ufe0f"),
    73: ("Snow", "\U0001f328\ufe0f"),
    75: ("Heavy snow", "\U0001f328\ufe0f"),
    77: ("Snow grains", "\U0001f328\ufe0f"),
    80: ("Rain showers", "\U0001f326\ufe0f"),
    81: ("Rain showers", "\U0001f326\ufe0f"),
    82: ("Violent rain showers", "\u26c8\ufe0f"),
    85: ("Snow showers", "\U0001f328\ufe0f"),
    86: ("Heavy snow showers", "\U0001f328\ufe0f"),
    95: ("Thunderstorm", "\u26c8\ufe0f"),
    96: ("Thunderstorm, hail", "\u26c8\ufe0f"),
    99: ("Thunderstorm, hail", "\u26c8\ufe0f"),
}


def _describe(code: int) -> str:
    description, emoji = _WMO_CODES.get(code, ("Unknown", ""))
    return f"{emoji} {description}".strip()


class QtWeatherWidget(QWidget):
    """Display the current date, time, temperature, and weather for a location.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    location : Location, optional
        Site to show the clock and weather for. Default: ``UPTON_NY``.
    weather_period : float, optional
        Interval between weather requests, in seconds. Default: 600 (10 minutes).
    temperature_unit : {"F", "C"}, optional
        Temperature unit to request and display. Default: ``"F"``.
    title : str, optional
        Group box title. Default: ``location.name``.
    """

    signal_weather_updated = Signal(object)

    def __init__(
        self,
        parent=None,
        *,
        location: Location = UPTON_NY,
        weather_period: float = 600.0,
        temperature_unit: Literal["F", "C"] = "F",
        title: str | None = None,
    ):
        super().__init__(parent)
        unit = temperature_unit.upper()
        if unit not in _TEMPERATURE_UNITS:
            raise ValueError(
                f"temperature_unit must be 'F' or 'C', got {temperature_unit!r}"
            )
        self._worker = None
        self._location = location
        self._tzinfo = location.tzinfo
        self._api_temperature_unit, self._temperature_suffix = _TEMPERATURE_UNITS[unit]

        self._date_label = QLabel(_PLACEHOLDER)
        self._time_label = QLabel(_PLACEHOLDER)
        self._temp_label = QLabel(_PLACEHOLDER)
        self._condition_label = QLabel(_PLACEHOLDER)
        for label in (
            self._date_label,
            self._time_label,
            self._temp_label,
            self._condition_label,
        ):
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        form = QFormLayout()
        form.setContentsMargins(8, 6, 8, 6)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        form.addRow("Date:", self._date_label)
        form.addRow("Time:", self._time_label)
        form.addRow("Temperature:", self._temp_label)
        form.addRow("Condition:", self._condition_label)

        group_box = QGroupBox(title if title is not None else location.name)
        group_box.setLayout(form)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(group_box)
        self.setLayout(vbox)

        self.signal_weather_updated.connect(self._slot_weather_updated)

        # Clock ticks once a second; weather refreshes on a slower cadence.
        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(1000)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start()
        self._update_clock()

        self._weather_timer = QTimer(self)
        self._weather_timer.setInterval(int(weather_period * 1000))
        self._weather_timer.timeout.connect(self._start_worker)
        self._weather_timer.start()
        self._start_worker()

    def _update_clock(self):
        now = datetime.now(self._tzinfo)
        self._date_label.setText(now.strftime("%A, %B %d, %Y"))
        self._time_label.setText(now.strftime("%I:%M:%S %p %Z"))

    def _start_worker(self):
        # Skip if a request is still in flight to avoid overlapping polls.
        if self._worker is not None:
            return
        self._worker = FunctionWorker(self._request_weather)
        self._worker.returned.connect(self._on_worker_returned)
        self._worker.start()

    def _request_weather(self) -> dict:
        """Return the current-weather dict, or an empty dict on any failure."""
        query = urllib.parse.urlencode(
            {
                "latitude": self._location.latitude,
                "longitude": self._location.longitude,
                "current": "temperature_2m,weather_code",
                "temperature_unit": self._api_temperature_unit,
                "timezone": self._location.timezone,
            }
        )
        try:
            with urllib.request.urlopen(f"{_API_URL}?{query}", timeout=10) as response:
                payload = json.load(response)
        except Exception:
            return {}
        return payload.get("current") or {}

    def _on_worker_returned(self, current):
        self._worker = None
        self.signal_weather_updated.emit(current)

    @Slot(object)
    def _slot_weather_updated(self, current: dict):
        temperature = current.get("temperature_2m")
        if temperature is not None:
            self._temp_label.setText(f"{temperature:g} {self._temperature_suffix}")
        else:
            self._temp_label.setText(_PLACEHOLDER)

        code = current.get("weather_code")
        if code is not None:
            self._condition_label.setText(_describe(int(code)))
        else:
            self._condition_label.setText(_PLACEHOLDER)

    def closeEvent(self, event):
        self._clock_timer.stop()
        self._weather_timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication

    app = QApplication([])
    widget = QtWeatherWidget()
    widget.resize(280, 160)
    widget.show()
    app.exec_()
