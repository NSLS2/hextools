"""Ophyd async support for the GeRM detector at HEX."""

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast
from typing import Annotated as A

import numpy as np
from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    DetectorAcquireLogic,
    DetectorDataLogic,
    DetectorTrigger,
    DetectorTriggerLogic,
    DeviceMock,
    PathProvider,
    SignalDataProvider,
    SignalR,
    SignalRW,
    StandardDetector,
    StandardReadable,
    StrictEnum,
    TriggerInfo,
    callback_on_mock_put,
    default_mock_class,
    set_and_wait_for_value,
    set_mock_value,
    soft_signal_r_and_setter,
    wait_for_value,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.adcore import (
    ADBaseColorMode,
    ADBaseDataType,
    ADHDFDataLogic,
    ADMultipartDataLogic,
    NDArrayDescription,
    NDFileHDF5IO,
    NDPluginBaseIO,
    NDPluginFileIO,
    NDStatsIO,
)
from ophyd_async.epics.core import EpicsDevice, PvSuffix

if TYPE_CHECKING:
    from hextools.photon_delivery_system.shutter import Shutter

logger = logging.getLogger(__name__)


class TDCSlopeTimes(StrictEnum):
    """TDC slope time setting."""

    ONE_US = "1us"
    TWO_US = "2us"
    THREE_US = "3us"
    FOUR_US = "4us"
    SIX_US = "6us"
    NINE_US = "9us"
    TWELVE_US = "12us"


class TDCMode(StrictEnum):
    """TDC mode: time of arrival or time over threshold."""

    TOA = "ToA"
    TOT = "ToT"


class InputPolarity(StrictEnum):
    """Input signal polarity."""

    POSITIVE = "Positive"
    NEGATIVE = "Negative"


class MonitorMode(StrictEnum):
    """Signal routed to the monitor output."""

    OFF = "Off"
    TEMPERATURE = "Temperature"
    BASELINE = "Baseline"
    THRESHOLD = "Threshold"
    TEST_PULSE = "TestPulse"
    CHANNEL = "Channel"


class Gain(StrictEnum):
    """Detector gain, given as the full-scale energy range."""

    GAIN_240KEV = "240keV"
    GAIN_120KEV = "120keV"
    GAIN_60KEV = "60keV"
    GAIN_30KEV = "30keV"


class ShapingTime(StrictEnum):
    """Pulse shaping time."""

    ST_0_125_US = "0.125us"
    ST_0_25_US = "0.25us"
    ST_0_5_US = "0.5us"
    ST_1_0_US = "1.0us"
    ST_2_0_US = "2.0us"
    ST_4_0_US = "4.0us"
    ST_8_0_US = "8.0us"
    ST_16_0_US = "16.0us"


class CountMode(StrictEnum):
    """Counting mode: timed or continuous."""

    TIMED = "Timed"
    CONTINUOUS = "Continuous"


class ShotMode(StrictEnum):
    """Acquisition shot mode: one shot or auto count."""

    ONE_SHOT = "OneShot"
    AUTO_COUNT = "AutoCount"


class LeakagePulseMode(StrictEnum):
    """Leakage pulse source: real or simulated."""

    REAL = "Real"
    SIMULATED = "Simulated"


class InternalLeakCurrent(StrictEnum):
    """Internal leakage current setting."""

    OFF = "Off"
    TWO_PA = "2pA"
    EIGHT_PA = "8pA"


class GeRMDetectorIO(EpicsDevice, StandardReadable):
    """IO class for GeRM detector."""

    num_elements: A[SignalR[int], PvSuffix("NumElements"), Format.CONFIG_SIGNAL]
    num_chips: A[SignalR[int], PvSuffix("NumChips"), Format.CONFIG_SIGNAL]
    ip_address: A[SignalR[str], PvSuffix("IPAddress_RBV"), Format.CONFIG_SIGNAL]

    acquire: A[SignalRW[bool], PvSuffix.rbv("Acquire")]
    # TODO: Get this fixed in the IOC - readback doesn't reflect setpoint
    acquire_time: A[SignalRW[float], PvSuffix("AcquireTime"), Format.CONFIG_SIGNAL]

    # Todo: Make into device vector
    model: A[SignalR[str], PvSuffix("DetectorModel"), Format.CONFIG_SIGNAL]
    firmware_version: A[SignalR[str], PvSuffix("FirmwareVersion"), Format.CONFIG_SIGNAL]

    temperature1: A[SignalR[float], PvSuffix("Temperature1")]
    temperature2: A[SignalR[float], PvSuffix("Temperature2")]
    temperature3: A[SignalR[float], PvSuffix("Temperature3")]
    zynq_temperature: A[SignalR[float], PvSuffix("ZynqTemperature")]
    high_voltage: A[SignalRW[float], PvSuffix.rbv("HighVoltage"), Format.CONFIG_SIGNAL]
    hv_current: A[SignalR[float], PvSuffix("HighVoltageCurrent")]
    udp_reachable: A[SignalR[bool], PvSuffix("UDPReachable_RBV")]
    udp_data_file_write_enable: A[SignalR[bool], PvSuffix("UDPDataFileWriteEnable")]

    chip: A[SignalRW[int], PvSuffix("Chip")]
    channel: A[SignalRW[int], PvSuffix("Channel")]
    monitor_channel: A[SignalRW[int], PvSuffix("MonitorChannel")]
    test_pulse_amplitude: A[SignalRW[int], PvSuffix.rbv("TestPulseAmplitude")]
    test_pulse_frequency: A[SignalRW[int], PvSuffix.rbv("TestPulseFrequency")]
    test_pulse_count: A[SignalRW[int], PvSuffix.rbv("TestPulseCount")]
    tdc_slope: A[SignalRW[TDCSlopeTimes], PvSuffix("TDCSlope")]
    tdc_mode: A[SignalRW[TDCMode], PvSuffix("TDCMode")]
    test_pulse_enable: A[SignalRW[bool], PvSuffix("TestPulseEnable")]
    input_polarity: A[SignalRW[InputPolarity], PvSuffix("Polarity")]

    # Acquisition settings
    gain: A[SignalRW[Gain], PvSuffix("Gain")]
    shaping_time: A[SignalRW[ShapingTime], PvSuffix.rbv("ShapingTime")]
    count_mode: A[SignalRW[CountMode], PvSuffix("CountMode")]
    shot_mode: A[SignalRW[ShotMode], PvSuffix("AutoCount")]
    elapsed_time: A[SignalR[float], PvSuffix("ElapsedTime")]
    auto_time: A[SignalRW[float], PvSuffix("AcquireTime1")]
    run_number: A[SignalRW[int], PvSuffix.rbv("RunNumber")]

    # Calibration settings
    multi_fire_suppression_enable: A[SignalRW[bool], PvSuffix("MultiFire")]
    leakage_pulse_mode: A[SignalRW[LeakagePulseMode], PvSuffix("SimEventSel")]
    internal_leak_current: A[SignalRW[InternalLeakCurrent], PvSuffix("LeakageCurrent")]
    pileup_rejection_enable: A[SignalRW[bool], PvSuffix("PileupReject")]
    pipeline_delay: A[SignalRW[int], PvSuffix.rbv("PipelineDelay")]
    readout_delay: A[SignalRW[int], PvSuffix.rbv("ReadoutDelay")]

    enable_test_pulse: A[SignalRW[bool], PvSuffix.rbv("TestPulseEnable")]
    enable_tsen: A[SignalRW[bool], PvSuffix("TsenSel")]
    enable_all_tsen: A[SignalRW[bool], PvSuffix("TsenAll")]
    enable_channels: A[SignalRW[bool], PvSuffix("ChenSel")]

    # TODO: Double check this
    mca: A[
        SignalR[np.ndarray[tuple[int], np.dtype[np.int32]]],
        PvSuffix("MCA"),
        Format.UNCACHED_SIGNAL,
    ]
    tdc: A[
        SignalR[np.ndarray[tuple[int], np.dtype[np.int32]]],
        PvSuffix("TDC"),
        Format.UNCACHED_SIGNAL,
    ]
    spectrum: A[
        SignalR[np.ndarray[tuple[int], np.dtype[np.int32]]],
        PvSuffix("Spectrum"),
        Format.UNCACHED_SIGNAL,
    ]
    spectrum_x: A[
        SignalR[np.ndarray[tuple[int], np.dtype[np.float64]]],
        PvSuffix("SpectrumX"),
        Format.UNCACHED_SIGNAL,
    ]
    intensity: A[
        SignalR[np.ndarray[tuple[int], np.dtype[np.int32]]],
        PvSuffix("Intensity"),
        Format.UNCACHED_SIGNAL,
    ]

    thresholds: A[
        SignalRW[np.ndarray[tuple[int], np.dtype[np.int32]]],
        PvSuffix("Threshold"),
        Format.CONFIG_SIGNAL,
    ]
    threshold_trims: A[
        SignalRW[np.ndarray[tuple[int], np.dtype[np.uint8]]],
        PvSuffix("ThresholdTrim"),
        Format.CONFIG_SIGNAL,
    ]
    pileup_trims: A[
        SignalRW[np.ndarray[tuple[int], np.dtype[np.uint8]]],
        PvSuffix("PileupTrim"),
        Format.CONFIG_SIGNAL,
    ]

    file_path: A[SignalRW[str], PvSuffix("FilePath"), Format.CONFIG_SIGNAL]
    file_name: A[SignalRW[str], PvSuffix("FileName"), Format.CONFIG_SIGNAL]
    file_size: A[SignalRW[int], PvSuffix("FileSize"), Format.CONFIG_SIGNAL]

    # TODO: Device Vector
    peltier1: A[SignalRW[float], PvSuffix("Peltier1"), Format.CONFIG_SIGNAL]
    peltier1_current: A[SignalR[float], PvSuffix("Peltier1Current")]
    peltier2: A[SignalRW[float], PvSuffix("Peltier2"), Format.CONFIG_SIGNAL]
    peltier2_current: A[SignalR[float], PvSuffix("Peltier2Current")]

    adc0_clock_skew: A[SignalRW[int], PvSuffix("Adc0ClkSkew"), Format.CONFIG_SIGNAL]
    adc1_clock_skew: A[SignalRW[int], PvSuffix("Adc1ClkSkew"), Format.CONFIG_SIGNAL]
    adc2_clock_skew: A[SignalRW[int], PvSuffix("Adc2ClkSkew"), Format.CONFIG_SIGNAL]

    # count: A[SignalRW[bool], PvSuffix(".CNT")]
    # mca: A[SignalR[float], PvSuffix(".MCA")]
    # number_of_channels: A[SignalRW[int], PvSuffix(".NELM"), Format.CONFIG_SIGNAL]
    # energy: A[SignalR[float], PvSuffix(".SPCTX")]
    # gain: A[SignalRW[float], PvSuffix(".GAIN"), Format.CONFIG_SIGNAL]
    # shaping_time: A[SignalRW[float], PvSuffix(".SHPT"), Format.CONFIG_SIGNAL]
    # count_time: A[SignalRW[float], PvSuffix(".TP"), Format.CONFIG_SIGNAL]
    # auto_time: A[SignalRW[float], PvSuffix(".TP1"), Format.CONFIG_SIGNAL]
    # run_num: A[SignalRW[int], PvSuffix(".RUNNO")]
    # fast_data_filename: A[SignalRW[str], PvSuffix(".FNAM"), Format.CONFIG_SIGNAL]
    # operating_mode: A[SignalRW[int], PvSuffix(".MODE")]
    # single_auto_toggle: A[SignalRW[int], PvSuffix(".CONT")]
    # gmon: A[SignalR[float], PvSuffix(".GMON")]
    # ip_addr: A[SignalRW[str], PvSuffix(".IPADDR")]
    # temp_1: A[SignalR[float], PvSuffix(":Temp1")]
    # temp_2: A[SignalR[float], PvSuffix(":Temp2")]
    # fpga_cpu_temp: A[SignalR[float], PvSuffix(":ztmp")]
    # calibration_file: A[SignalRW[str], PvSuffix(".CALF")]
    # multi_file_supression: A[SignalRW[int], PvSuffix(".MFS")]
    # tdc: A[SignalRW[int], PvSuffix(".TDC")]
    # leakage_pulse: A[SignalRW[int], PvSuffix(".LOAO")]
    # internal_leak_curr: A[SignalRW[int], PvSuffix(".EBLK")]
    # pileup_rejection: A[SignalRW[int], PvSuffix(".PUEN")]
    # test_pulse_aplitude: A[SignalRW[float], PvSuffix(".TPAMP")]
    # channel: A[SignalRW[int], PvSuffix(".MONCH")]
    # tdc_slope: A[SignalRW[float], PvSuffix(".TDS")]
    # test_pulse_freq: A[SignalRW[float], PvSuffix(".TPFRQ")]
    # tdc_mode: A[SignalRW[int], PvSuffix(".TDM")]
    # test_pulce_enable: A[SignalRW[int], PvSuffix(".TPENB")]
    # test_pulse_count: A[SignalRW[int], PvSuffix(".TPCNT")]
    # input_polarity: A[SignalRW[int], PvSuffix(".POL")]
    # voltage: A[SignalR[float], PvSuffix(":HV_RBV"), Format.CONFIG_SIGNAL]
    # current: A[SignalR[float], PvSuffix(":HV_CUR")]
    # peltier_2: A[SignalRW[float], PvSuffix(":P2")]
    # peliter_2_current: A[SignalR[float], PvSuffix(":P2_CUR")]
    # peltier_1: A[SignalRW[float], PvSuffix(":P1")]
    # peltier_1_current: A[SignalR[float], PvSuffix(":P1_CUR")]
    # hv_bias: A[SignalRW[float], PvSuffix(":HV"), Format.CONFIG_SIGNAL]
    # ring_hi: A[SignalRW[float], PvSuffix(":DRFTHI")]
    # ring_lo: A[SignalRW[float], PvSuffix(":DRFTLO")]
    # channel_enabled: A[SignalRW[int], PvSuffix(".TSEN")]
    # write_dir: A[SignalRW[str], PvSuffix(":write_dir"), Format.CONFIG_SIGNAL]
    # file_name: A[SignalRW[str], PvSuffix(":file_name"), Format.CONFIG_SIGNAL]
    # frame_num: A[SignalR[int], PvSuffix(":frame_num")]
    # frame_shape: A[SignalR[int], PvSuffix(":frame_shape")]
    # ioc_stage: A[SignalRW[str], PvSuffix(":stage")]
    # count: A[SignalRW[bool], PvSuffix(":count")]

    def __init__(self, prefix: str, name: str = "") -> None:
        with self.add_children_as_readables(Format.CONFIG_SIGNAL):
            self.num_energy_bins, _ = soft_signal_r_and_setter(
                int, 4096, name="num_energy_bins"
            )
            self.data_type, _ = soft_signal_r_and_setter(
                ADBaseDataType, ADBaseDataType.INT32, name="data_type"
            )
            self.color_mode, _ = soft_signal_r_and_setter(
                ADBaseColorMode, ADBaseColorMode.MONO, name="color_mode"
            )
        super().__init__(prefix, name=name)


class GeRMTriggerLogic(DetectorTriggerLogic):
    """The trigger logic for GeRM detector."""

    driver: GeRMDetectorIO

    def __init__(self, driver: GeRMDetectorIO) -> None:
        self.driver = driver

    async def prepare_internal(self, num: int, livetime: float, deadtime: float):
        """Prepare the detector for an internal trigger.

        Parameters
        ----------
        livetime : float
            How long the exposure should be, 0 means what is currently set.
        deadtime : float
            How long between exposures, 0 means the shortest possible.
        """
        if num != 1:
            raise ValueError(
                "Only a single collection with a single exposure is supported."
            )
        await self.driver.acquire_time.set(livetime)

    async def default_trigger_info(self) -> TriggerInfo:
        livetime = await self.driver.acquire_time.get_value()
        return TriggerInfo(
            trigger=DetectorTrigger.INTERNAL,
            livetime=livetime,
            deadtime=0,
            exposures_per_collection=1,
            collections_per_event=1,
            number_of_events=1,
        )


class GeRMAcquireLogic(DetectorAcquireLogic):
    """The arm logic for GeRM detector."""

    def __init__(self, driver: GeRMDetectorIO) -> None:
        self.driver = driver

    async def start_acquiring(self):
        await set_and_wait_for_value(self.driver.acquire, True)

    async def wait_for_idle(self):
        count_time = await self.driver.acquire_time.get_value()
        await wait_for_value(
            self.driver.acquire, False, timeout=count_time + DEFAULT_TIMEOUT
        )

    async def ensure_stopped(self):
        await set_and_wait_for_value(self.driver.acquire, False)


@dataclass
class GeRMStatsDataLogic(DetectorDataLogic):
    """Data logic that reads a single stats signal from the GeRM detector."""

    signal: SignalR
    hinted: bool = True

    async def prepare_single(self, datakey_name: str) -> SignalDataProvider:
        return SignalDataProvider(self.signal)

    def get_hinted_fields(self, datakey_name: str) -> Sequence[str]:
        return [self.signal.name] if self.hinted else []


#: Full-scale energy (keV) of each gain setting, for the mock's energy axis.
_GAIN_FULL_SCALE_KEV = {
    Gain.GAIN_240KEV: 240.0,
    Gain.GAIN_120KEV: 120.0,
    Gain.GAIN_60KEV: 60.0,
    Gain.GAIN_30KEV: 30.0,
}


class GeRMDetectorMock(DeviceMock["GeRMDetector"]):
    """Simulates a GeRM count, writing each spectrum frame to HDF5 or TIFF.

    Setting ``acquire`` counts for ``acquire_time`` (updating ``elapsed_time``),
    then publishes a fresh ``(num_elements, num_energy_bins)`` frame to the live
    signals and ``stats1.total``. Whichever file plugin is capturing writes the frame
    to disk as the IOC would: appended to ``<file_path><file_name>.h5`` at
    ``/entry/data/data``, or as ``<file_path><file_name>_NNNNNN.tiff``.

    The frame is the sum of the diffraction signal, counted only while every shutter
    given to :meth:`set_shutters` is open, and a fixed test pulse peak while
    ``test_pulse_enable`` is set; with neither it is all zeros. Functions given to
    :meth:`register_frame_callback` are called with the accumulated frame once per
    second while counting, and with the final frame.
    """

    #: The HEX GeRM is a 192-strip detector.
    num_elements = 192
    #: Energies (keV) and rates (counts/s/bin) of the simulated diffraction lines.
    lines_kev = ((11.7, 6.0), (12.4, 3.0), (28.3, 3.0), (30.6, 4.0), (35.0, 10.0))
    #: Full-scale test pulse amplitude, in DAC counts.
    test_pulse_full_scale = 1024
    #: Seconds between frame callbacks while counting.
    frame_update_period = 1.0

    def __init__(self, name: str = "", parent: DeviceMock | None = None) -> None:
        super().__init__(name, parent)
        self._acquisition: asyncio.Task | None = None
        self._rng = np.random.default_rng(0)
        self._frame_callbacks: list[Callable[[np.ndarray], None]] = []
        self._shutters: tuple[Shutter, ...] = ()
        self._frame_shape: tuple[int, int] | None = None

    def register_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Call ``callback`` with each ``(num_elements, num_energy_bins)`` frame.

        It is first called with an all-zero frame, once the detector is connected.
        Later calls come from the event loop the detector is connected in.
        """
        self._frame_callbacks.append(callback)
        if self._frame_shape is not None:
            callback(np.zeros(self._frame_shape, dtype=np.int32))

    def unregister_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Stop calling a callback given to :meth:`register_frame_callback`."""
        self._frame_callbacks.remove(callback)

    def set_shutters(self, *shutters: "Shutter") -> None:
        """Only count the diffraction signal while all of ``shutters`` are open."""
        self._shutters = shutters

    async def connect(self, device: "GeRMDetector") -> None:
        driver = device.driver
        set_mock_value(driver.num_elements, self.num_elements)
        set_mock_value(driver.num_chips, self.num_elements // 32)
        set_mock_value(driver.model, "Mock GeRM")
        set_mock_value(driver.firmware_version, "mock")
        set_mock_value(driver.ip_address, "127.0.0.1")
        set_mock_value(driver.acquire_time, 1.0)
        set_mock_value(driver.gain, Gain.GAIN_120KEV)
        set_mock_value(driver.shaping_time, ShapingTime.ST_1_0_US)
        set_mock_value(driver.count_mode, CountMode.TIMED)
        set_mock_value(driver.shot_mode, ShotMode.ONE_SHOT)
        set_mock_value(driver.high_voltage, -500.0)
        set_mock_value(driver.temperature1, -30.0)
        set_mock_value(driver.temperature2, -30.0)
        set_mock_value(driver.temperature3, -30.0)
        set_mock_value(driver.zynq_temperature, 45.0)
        set_mock_value(driver.udp_reachable, True)
        set_mock_value(driver.test_pulse_amplitude, 100)
        set_mock_value(driver.test_pulse_frequency, 1000)
        set_mock_value(driver.test_pulse_count, 99999)
        set_mock_value(device.hdf.num_frames_chunks, 1)

        for writer in (device.hdf, device.tiff):
            set_mock_value(writer.file_path_exists, True)
            self._reset_count_on_capture(writer)

        def _on_acquire(value: bool) -> None:
            counting = self._acquisition is not None and not self._acquisition.done()
            if counting and not value:
                # Stopped early: abandon the count without writing a frame.
                self._acquisition.cancel()  # type: ignore[union-attr]
            elif value and not counting:
                self._acquisition = asyncio.ensure_future(self._acquire(device))

        callback_on_mock_put(driver.acquire, _on_acquire)

        self._frame_shape = (
            self.num_elements,
            await driver.num_energy_bins.get_value(),
        )
        self._notify(np.zeros(self._frame_shape, dtype=np.int32))

    @staticmethod
    def _reset_count_on_capture(writer: NDPluginFileIO) -> None:
        def _on_capture(value: bool) -> None:
            if value:
                set_mock_value(writer.num_captured, 0)

        callback_on_mock_put(writer.capture, _on_capture)

    async def _acquire(self, device: "GeRMDetector") -> None:
        driver = device.driver
        acquire_time = await driver.acquire_time.get_value()
        num_elements = await driver.num_elements.get_value()
        num_bins = await driver.num_energy_bins.get_value()
        full_scale = _GAIN_FULL_SCALE_KEV[await driver.gain.get_value()]
        energies = np.linspace(0.0, full_scale, num_bins)
        set_mock_value(driver.spectrum_x, energies)
        rate = self._beam_rate(energies, num_elements)

        signal = np.zeros((num_elements, num_bins), dtype=np.int32)
        pulse_time = 0.0
        since_update = 0.0
        step = min(0.1, acquire_time)
        elapsed = 0.0
        set_mock_value(driver.elapsed_time, 0.0)
        while elapsed < acquire_time:
            await asyncio.sleep(step)
            previous, elapsed = elapsed, min(elapsed + step, acquire_time)
            dt = elapsed - previous
            # Shutters and test pulses are checked each step, as they can change.
            if await self._shutters_open():
                signal += self._rng.poisson(rate * dt).astype(np.int32)
            if await driver.test_pulse_enable.get_value():
                pulse_time += dt
            set_mock_value(driver.elapsed_time, elapsed)
            since_update += dt
            if since_update >= self.frame_update_period - 1e-9 and (
                elapsed < acquire_time
            ):
                since_update = 0.0
                self._notify(await self._frame(driver, signal, pulse_time))

        frame = await self._frame(driver, signal, pulse_time)
        set_mock_value(driver.mca, frame.ravel())
        set_mock_value(driver.spectrum, frame.sum(axis=0, dtype=np.int32))
        set_mock_value(driver.intensity, frame.sum(axis=1, dtype=np.int32))
        set_mock_value(device.stats1.total, float(frame.sum()))

        for writer in (device.hdf, device.tiff):
            if await writer.capture.get_value():
                await self._write_frame(writer, frame)
                captured = await writer.num_captured.get_value()
                set_mock_value(writer.num_captured, captured + 1)

        self._notify(frame)
        set_mock_value(driver.acquire, False)

    def _notify(self, frame: np.ndarray) -> None:
        for callback in list(self._frame_callbacks):
            try:
                callback(frame)
            except Exception:
                logger.exception("GeRM mock frame callback %r failed", callback)

    async def _frame(
        self, driver: GeRMDetectorIO, signal: np.ndarray, pulse_time: float
    ) -> np.ndarray:
        """Add the test pulses for ``pulse_time`` seconds to a copy of ``signal``."""
        frame = signal.copy()
        if pulse_time > 0:
            frame += await self._test_pulses(driver, frame.shape, pulse_time)
        return frame

    async def _shutters_open(self) -> bool:
        from hextools.photon_delivery_system.shutter import ShutterStatus

        for shutter in self._shutters:
            if await shutter.status.get_value() != ShutterStatus.OPEN:
                return False
        return True

    def _beam_rate(self, energies: np.ndarray, num_elements: int) -> np.ndarray:
        """Return the expected counts/s per strip and bin with beam.

        The continuum runs from a ragged low-energy cutoff near 6.5 keV, rising to a
        sharp edge near 35.5 keV, with a sparse tail above it.
        """
        # Fixed seed: the per-strip geometry must not change between counts.
        geometry = np.random.default_rng(1)
        strips = np.arange(num_elements)[:, np.newaxis]
        low = 6.5 + geometry.normal(0.0, 0.35, (num_elements, 1)) * (
            1.0 + strips / num_elements
        )
        high = (
            35.5
            + 0.2 * np.sin(strips / 4.0)
            + 0.3 * strips / num_elements
            + geometry.normal(0.0, 0.08, (num_elements, 1))
        )
        position = np.clip((energies - low) / (high - low), 0.0, 1.0)
        above_low = np.clip(energies - low, 0.0, None)
        rate = 2.0 + 6.0 * position**3 + 3.0 * np.exp(-above_low / 0.8)
        for energy, line_rate in self.lines_kev:
            rate = rate + line_rate * np.exp(-0.5 * ((energies - energy) / 0.25) ** 2)
        # Logistic edges, written with tanh so they cannot overflow.
        rate *= 0.5 * (1.0 + np.tanh((energies - low) / 0.2))
        rate *= 0.5 * (1.0 + np.tanh((high - energies) / 0.16))
        above = np.clip(energies - high, 0.0, None)
        rate += np.where(energies > high, 0.01 * np.exp(-above / 12.0), 0.0)
        return rate

    async def _test_pulses(
        self, driver: GeRMDetectorIO, shape: tuple[int, ...], acquire_time: float
    ) -> np.ndarray:
        """Put the same narrow peak in every strip, at the bin set by the amplitude."""
        num_bins = shape[1]
        amplitude = await driver.test_pulse_amplitude.get_value()
        frequency = await driver.test_pulse_frequency.get_value()
        max_pulses = await driver.test_pulse_count.get_value()
        pulses = min(frequency * acquire_time, max_pulses)
        center = amplitude / self.test_pulse_full_scale * (num_bins - 1)
        peak = np.exp(-0.5 * ((np.arange(num_bins) - center) / 2.0) ** 2)
        peak *= pulses / peak.sum() if peak.sum() > 0 else 0.0
        return np.broadcast_to(np.rint(peak).astype(np.int32), shape).copy()

    @staticmethod
    async def _write_frame(writer: NDPluginFileIO, frame: np.ndarray) -> None:
        directory = Path(await writer.file_path.get_value())
        directory.mkdir(parents=True, exist_ok=True)
        file_name = await writer.file_name.get_value()
        if isinstance(writer, NDFileHDF5IO):
            # Imported here so the detector does not require h5py unless mocked.
            import h5py

            with h5py.File(directory / f"{file_name}.h5", "a") as f:
                if "/entry/data/data" in f:
                    dataset = cast(h5py.Dataset, f["/entry/data/data"])
                    dataset.resize(dataset.shape[0] + 1, axis=0)
                    dataset[-1] = frame
                else:
                    f.create_dataset(
                        "/entry/data/data",
                        data=frame[np.newaxis],
                        maxshape=(None, *frame.shape),
                        chunks=(1, *frame.shape),
                    )
        else:
            import tifffile

            file_number = await writer.file_number.get_value()
            tifffile.imwrite(directory / f"{file_name}_{file_number:06d}.tiff", frame)
            set_mock_value(writer.file_number, file_number + 1)


@default_mock_class(GeRMDetectorMock)
class GeRMDetector(StandardDetector):
    """The ophyd class for GeRM detector."""

    def __init__(
        self,
        prefix: str,
        path_provider: PathProvider,
        plugins: Mapping[str, NDPluginBaseIO] | None = None,
        config_sigs: Sequence[SignalR] = (),
        name: str = "",
    ) -> None:
        self._path_provider = path_provider
        self.driver = GeRMDetectorIO(prefix, name=name)
        if plugins is not None:
            for plugin_name, plugin in plugins.items():
                setattr(self, plugin_name, plugin)
        self.add_detector_logics(GeRMTriggerLogic(self.driver))
        self.add_detector_logics(GeRMAcquireLogic(self.driver))
        self.hdf = NDFileHDF5IO(prefix + "MCA1:HDF1:", name="hdf")
        self.tiff = NDPluginFileIO(prefix + "MCA1:TIFF1:", name="tiff")
        self.stats1 = NDStatsIO("XF:27ID1-ES{GeRM-Det:1}MCA1:Stats1:", name="stats1")
        self.save_as_hdf()
        self.add_config_signals(*config_sigs)
        super().__init__(name=name)

    def save_as_tiff(self) -> None:
        if self._data_logics and isinstance(self._data_logics[0], ADMultipartDataLogic):
            return
        self._data_logics = ()
        self.add_detector_logics(
            ADMultipartDataLogic(
                NDArrayDescription(
                    [self.driver.num_elements, self.driver.num_energy_bins],
                    self.driver.data_type,
                    self.driver.color_mode,
                ),
                self._path_provider,
                self.tiff,
                ".tiff",
                "multipart/related;type=image/tiff",
                hinted=False,
            ),
            GeRMStatsDataLogic(self.stats1.total),
        )

    def save_as_hdf(self) -> None:
        if self._data_logics and isinstance(self._data_logics[0], ADHDFDataLogic):
            return
        self._data_logics = ()
        self.add_detector_logics(
            ADHDFDataLogic(
                NDArrayDescription(
                    [self.driver.num_elements, self.driver.num_energy_bins],
                    self.driver.data_type,
                    self.driver.color_mode,
                ),
                self._path_provider,
                self.hdf,
                hinted=False,
            ),
            GeRMStatsDataLogic(self.stats1.total),
        )
