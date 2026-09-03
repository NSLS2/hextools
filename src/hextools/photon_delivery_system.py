"""Device classes and plans for the photon delivery system (PDS) at the HEX beamline."""

import asyncio
from dataclasses import dataclass
from enum import IntEnum
from importlib import resources
from typing import Final

import numpy as np
import yaml
from bluesky import plan_stubs as bps
from bluesky import plans as bp
from bluesky.callbacks.fitting import PeakStats
from bluesky.preprocessors import subs_decorator
from ophyd_async.core import (
    AsyncMovable,
    AsyncStatus,
    StandardReadable,
    StrictEnum,
    derived_signal_r,
    wait_for_value,
)
from ophyd_async.core import (
    StandardReadableFormat as Format,
)
from ophyd_async.epics.adcore import AreaDetector, NDStatsIO
from ophyd_async.epics.core import (
    EpicsDevice,
    epics_signal_r,
    epics_triggerable_command,
)
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor


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


class BeamMode(StrictEnum):
    """Beam modes."""

    MONOCHROMATIC = "Monochromatic"
    WHITE = "White"


@dataclass
class FilterSetting:
    """Description and motor position for a filter setting."""

    position: float
    description: str | None = None


class FilterPosition(IntEnum):
    """Filter positions."""

    UPPER_LIMIT = 0
    PASS_THROUGH = 1
    POS_1 = 2  # Thickest filter
    POS_2 = 3
    POS_3 = 4
    POS_4 = 5  # Thinnest filter
    LOWER_LIMIT = 6


class Filter(StandardReadable, EpicsDevice, AsyncMovable[FilterPosition]):
    """Filter device with multiple positions.

    This class represents a filter device with multiple positions, each corresponding
    to a specific filter setting. It provides methods to set the filter position and
    retrieve the description of the current filter setting.

    Attributes
    ----------
    in_position : epics_signal_r
        Signal indicating whether the filter is in position.
    positions : dict
        Dictionary mapping FilterPosition to FilterSetting.
    filter_motor : AsyncEpicsMotor
        Motor controlling the filter position.
    description : derived_signal_r
        Derived signal providing the description of the current filter setting.
    """

    def __init__(
        self,
        motor_pv: str,
        positions: dict[FilterPosition, FilterSetting],
        in_pos_switch_pv: str,
        name: str = "",
    ):
        self.in_position = epics_signal_r(bool, in_pos_switch_pv)
        self.positions = positions
        self.filter_motor = AsyncEpicsMotor(motor_pv)
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.description = derived_signal_r(
                self._get_description,
                in_pos=self.in_position,
                motor_pos=self.filter_motor.user_readback,
            )
        super().__init__(name=name)

    def _get_description(self, in_pos: int, motor_pos: float) -> str:
        """Get the description of the current filter setting based on motor position.

        Parameters
        ----------
        in_pos : int
            Indicates whether the filter is in position.
        motor_pos : float
            Current position of the filter motor.

        Returns
        -------
        str
            Description of the current filter setting.
        """
        if not in_pos:
            return "out of position"
        closest = min(
            self.positions.items(), key=lambda item: abs(item[1].position - motor_pos)
        )
        return closest[1].description or closest[0].name.lower().replace("_", " ")

    @AsyncStatus.wrap
    async def set(self, value: FilterPosition):
        """Set the filter to the specified position.

        Parameters
        ----------
        value : FilterPosition
            The desired filter position to set.
        """
        if value not in self.positions:
            raise ValueError(
                f"Invalid filter position: {value}. "
                f"Defined positions are: {list(self.positions.keys())}"
            )

        await self.filter_motor.set(self.positions[value].position)
        await wait_for_value(self.in_position, True, timeout=10)


FILTERS_CONFIG_RESOURCE: str = "filters.yml"


def load_filters() -> list[Filter]:
    """Create Filter devices from the packaged YAML configuration file.

    The ``filters.yml`` file bundled with the ``hextools`` package is used.
    Each top-level key defines a filter with its ``motor_pv``,
    ``in_position_switch`` PV, and a ``positions`` mapping. The ``positions``
    mapping must contain exactly one entry per :class:`FilterPosition`, keyed by
    the lowercased enum name (``upper_limit``, ``pass_through``, ``pos_1`` ..
    ``pos_4``, ``lower_limit``). Each value is either a number (the motor
    position) or a mapping with ``position`` and optional ``description`` keys.

    Returns
    -------
    list[Filter]
        One configured Filter device per top-level entry, named after its key.
    """
    source = resources.files(__name__).joinpath(FILTERS_CONFIG_RESOURCE)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))

    filters: list[Filter] = []
    for name, spec in config["filters"].items():
        positions_spec = spec["positions"]
        if len(positions_spec) != len(FilterPosition):
            raise ValueError(
                f"Filter {name!r} must define exactly {len(FilterPosition)} "
                f"positions (one per FilterPosition), got {len(positions_spec)}."
            )
        positions: dict[FilterPosition, FilterSetting] = {}
        for key, value in positions_spec.items():
            filter_position = FilterPosition[key.upper()]
            if isinstance(value, dict):
                setting = FilterSetting(value["position"], value.get("description"))
            else:
                setting = FilterSetting(value)
            positions[filter_position] = setting
        filters.append(
            Filter(
                motor_pv=spec["motor_pv"],
                positions=positions,
                in_pos_switch_pv=spec["in_position_switch"],
                name=name,
            )
        )
    return filters


class Slits(StandardReadable, EpicsDevice):
    """Generic slits with inboard, outboard, bottom, and top motors."""

    def __init__(self, prefix: str, num: int, name: str = ""):
        super().__init__(f"{prefix}{{Slt:{num}-Ax:", name=name or f"slits{num}")
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.inboard = AsyncEpicsMotor(prefix + "I}Mtr", name="inboard")
            self.outboard = AsyncEpicsMotor(prefix + "O}Mtr", name="outboard")
            self.bottom = AsyncEpicsMotor(prefix + "B}Mtr", name="bottom")
            self.top = AsyncEpicsMotor(prefix + "T}Mtr", name="top")


class DCLM(StandardReadable, EpicsDevice, AsyncMovable[BeamMode]):
    """Double crystal Laue monochromator device.

    This class represents a double crystal laue monochromator (DCLM) device,
    which is used to select a specific energy of X-rays from a white beam.
    It provides methods to set the beam mode (white or monochromatic) and to
    change the energy of the monochromatic beam.

    Attributes
    ----------
    xtal1_vertical_trans : AsyncEpicsMotor
        Motor for the vertical translation of the first crystal.
    cooled_beam_stop : AsyncEpicsMotor
        Motor for the cooled beam stop.
    xtal1_pitch : AsyncEpicsMotor
        Motor for the pitch of the first crystal.
    xtal2_pitch : AsyncEpicsMotor
        Motor for the pitch of the second crystal.
    flourescence_screen : AsyncEpicsMotor
        Motor for the fluorescence screen.
    xtal2_z : AsyncEpicsMotor
        Motor for the z-position of the second crystal.
    beam_mode : derived_signal_r
        Derived signal representing the current beam mode (white or monochromatic).
    energy : derived_signal_r
        Derived signal representing the current energy of the monochromatic beam.
    beam_stop_in : float
        Position of the beam stop when it is in the beam path.
    beam_stop_out : float
        Position of the beam stop when it is out of the beam path.
    crystal_1_in : float
        Position of the first crystal when it is in the beam path.
    crystal_1_out : float
        Position of the first crystal when it is out of the beam path.
    fluo_y_direct : float
        Position of the fluorescence screen when it is in the direct beam path.
    fluo_y_direct_out : float
        Position of the fluorescence screen when it is out of the beam path.

    Methods
    -------
    set(value: BeamMode)
        Set the beam mode to either white or monochromatic.
    """

    # Constants for in/out positions of the monochromator components
    beam_stop_in: Final[float] = 0.0
    beam_stop_out: Final[float] = 48.3
    xtal1_in: Final[float] = 0.0
    xtal1_out: Final[float] = -36.0
    fs_in: Final[float] = 25.0
    fs_out: Final[float] = 46.0

    # Constants for the monochromator geometry
    bragg_factor: Final[float] = 1.977  # Energy in keV for Si(111) at 2d = 6.271 Å
    bragg_angle_offset: Final[float] = (
        35.2544  # Bragg angle offset in degrees for Si(111)
    )
    fixed_beam_offset: Final[float] = (
        25.0  # Crystal 2 z offset in mm for fixed beam offset
    )
    fs_distance: Final[float] = 1428.0  # Distance to fluorescence screen in mm

    def __init__(self, prefix: str, name: str = ""):

        self.xtal1_vertical_trans = AsyncEpicsMotor(prefix + "C1Y}Mtr")
        self.cooled_beam_stop = AsyncEpicsMotor(
            prefix + "BS}Mtr",
        )
        self.xtal1_pitch = AsyncEpicsMotor(prefix + "C1P}Mtr")
        self.xtal2_pitch = AsyncEpicsMotor(prefix + "C2P}Mtr")
        self.flourescence_screen = AsyncEpicsMotor(prefix + "FS}Mtr")
        self.xtal2_z = AsyncEpicsMotor(prefix + "Z2}Mtr")
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.beam_mode = derived_signal_r(
                self._get_beam_mode,
                xtal1_pos=self.xtal1_vertical_trans.user_readback,
                beam_stop_pos=self.cooled_beam_stop.user_readback,
            )
            self.energy = derived_signal_r(
                self._get_energy,
                pitch_angle=self.xtal1_pitch.user_readback,
            )

        super().__init__(name=name)

    def _get_beam_mode(self, xtal1_pos: float, beam_stop_pos: float) -> BeamMode:
        """Determine the current beam mode."""
        xtal1_out = abs(xtal1_pos - self.xtal1_out) < 0.01
        beam_stop_out = abs(beam_stop_pos - self.beam_stop_out) < 0.01
        if xtal1_out and beam_stop_out:
            return BeamMode.WHITE
        return BeamMode.MONOCHROMATIC

    def _get_energy(self, pitch_angle: float) -> float:
        """Compute the energy of the monochromatic beam based on the pitch angle."""
        bragg_angle = np.deg2rad(self.bragg_angle_offset - pitch_angle)
        return self.bragg_factor / np.sin(bragg_angle)

    @AsyncStatus.wrap
    async def set(self, value: BeamMode):
        if value == BeamMode.WHITE:
            xtal1_target, bs_target = self.xtal1_out, self.beam_stop_out
        else:
            xtal1_target, bs_target = self.xtal1_in, self.beam_stop_in
        coros = (
            self.xtal1_vertical_trans.set(xtal1_target),
            self.cooled_beam_stop.set(bs_target),
        )
        await asyncio.gather(*coros)


def change_energy(
    dclm: DCLM,
    energy: float = 0.0,
    fs_camera: AreaDetector | None = None,
    coarse_angle_range: float = 0.1,
    coarse_num_steps: int = 41,
    fine_angle_range: float = 0.025,
    fine_num_steps: int = 26,
):
    """Bluesky plan to change monochromator energy for Si(111).

    Computes Bragg geometry and moves all crystal and beam stop motors.
    If an area detector is provided, performs coarse and fine pitch scans
    to auto-tune the crystal 2 pitch to the fluorescence peak.

    Parameters
    ----------
    monochromator : DCLM
        Must be in monochromatic mode.
    energy : float
        Target energy in keV.
    fs_camera : AreaDetector, optional
        Fluorescence screen camera for auto-tuning. If None, motors are
        moved without feedback.
    coarse_angle_range : float
        Half-width of the coarse pitch scan in degrees.
    coarse_num_steps : int
        Number of points in the coarse scan.
    fine_angle_range : float
        Half-width of the fine pitch scan in degrees.
    fine_num_steps : int
        Number of points in the fine scan.

    Raises
    ------
    RuntimeError
        If the monochromator is not in monochromatic mode.
    """
    mode = yield from bps.rd(dclm.beam_mode)
    if mode != BeamMode.MONOCHROMATIC:
        raise RuntimeError("Monochromator is not in monochromatic mode.")

    if not np.isfinite(energy) or energy <= dclm.bragg_factor:
        raise ValueError(
            f"Energy must be finite and greater than {dclm.bragg_factor} keV."
        )
    bragg_angle = np.arcsin(
        dclm.bragg_factor / energy
    )  # Si(111), 2d = 6.271 Å, energy in keV
    angle = dclm.bragg_angle_offset - np.rad2deg(
        bragg_angle
    )  # Bragg angle to motor coordinate
    z = dclm.fixed_beam_offset / np.tan(
        2 * bragg_angle
    )  # Crystal 2 z for fixed beam offset of 25 mm
    if fs_camera is not None:
        fluo_y = dclm.fs_distance * np.tan(
            2 * bragg_angle
        )  # Fluorescence screen y at 1428 mm downstream
    else:
        fluo_y = dclm.fs_in

    # fmt: off
    yield from bps.mv(
        dclm.xtal2_z, z,
        dclm.xtal2_pitch, angle,
        dclm.xtal1_pitch, angle,
        dclm.xtal1_vertical_trans, dclm.xtal1_in,
        dclm.cooled_beam_stop, dclm.beam_stop_in,
        dclm.flourescence_screen, fluo_y,
    )
    # fmt: on

    # No feedback available without a camera, so just move the motors and return.
    if fs_camera is None:
        return

    # Create a PeakStats object to monitor the fluorescence screen camera signal
    # and find the position of the crystal 2 pitch that produces a peak.
    ps = PeakStats(
        dclm.xtal2_pitch.name,
        fs_camera.get_plugin_by_name("stats1", NDStatsIO).total.name,
    )

    # Perform a scan around the current position of the crystal 2 pitch,
    # and feed the produced events into the PeakStats object to find the peak position.
    @subs_decorator(ps)
    def auto_tune(angle_range: float, num_steps: int):
        yield from bp.scan(
            [fs_camera],
            dclm.xtal2_pitch,
            angle - angle_range,
            angle + angle_range,
            num_steps,
            md={"plan_name": "change_energy_auto_tune"},
        )

    yield from auto_tune(coarse_angle_range, coarse_num_steps)  # Coarse scan

    peak: float | None = ps.com  # ty: ignore[unresolved-attribute]
    if peak is None:
        raise RuntimeError(
            "No peak found in coarse scan. Check the fluorescence screen."
        )

    # Move to the peak found by the coarse scan
    yield from bps.mv(dclm.xtal2_pitch, peak)

    # Clear coarse scan events so the fine scan computes from its own data only
    ps.reset()

    yield from auto_tune(fine_angle_range, fine_num_steps)  # Fine scan

    peak = ps.com  # ty: ignore[unresolved-attribute]
    if peak is None:
        raise RuntimeError("No peak found in fine scan. Check the fluorescence screen.")

    # To minimize issues with backlash, always approach the final position from below.
    yield from bps.mv(dclm.xtal2_pitch, peak - 0.05)
    yield from bps.mv(dclm.xtal2_pitch, peak)

    # Move the fluorescence screen out of the beam path
    yield from bps.mv(dclm.flourescence_screen, dclm.fs_out)
