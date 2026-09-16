"""Ophyd async support for the GeRM detector at HEX."""

from collections.abc import Mapping, Sequence
from tkinter.font import names
from typing import Annotated as A

from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    DetectorAcquireLogic,
    DetectorDataLogic,
    DetectorTrigger,
    DetectorTriggerLogic,
    PathProvider,
    SignalR,
    SignalRW,
    StandardDetector,
    StandardReadable,
    StrictEnum,
    TriggerInfo,
    set_and_wait_for_value,
    soft_signal_r_and_setter,
    wait_for_value,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice, PvSuffix
from ophyd_async.epics.adcore import (
    ADBaseColorMode,
    ADBaseDataType,
    ADBaseIO,
    ADHDFDataLogic,
    ADWriterFactory,
    NDArrayDescription,
    NDFileHDF5IO,
    NDPluginBaseIO,
    NDProcessIO,
)
import numpy as np


class TDCSlopeTimes(StrictEnum):
    ONE_US = "1us"
    TWO_US = "2us"
    THREE_US = "3us"
    FOUR_US = "4us"
    SIX_US = "6us"
    NINE_US = "9us"
    TWELVE_US = "12us"


class TDCMode(StrictEnum):
    TOA = "ToA"
    TOT = "ToT"


class InputPolarity(StrictEnum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"


class MonitorMode(StrictEnum):
    OFF = "Off"
    TEMPERATURE = "Temperature"
    BASELINE = "Baseline"
    THRESHOLD = "Threshold"
    TEST_PULSE = "TestPulse"
    CHANNEL = "Channel"


class Gain(StrictEnum):
    GAIN_240KEV = "240keV"
    GAIN_120KEV = "120keV"
    GAIN_60KEV = "60keV"
    GAIN_30KEV = "30keV"


class ShapingTime(StrictEnum):
    ST_0_125_US = "0.125us"
    ST_0_25_US = "0.25us"
    ST_0_5_US = "0.5us"
    ST_1_0_US = "1.0us"
    ST_2_0_US = "2.0us"
    ST_4_0_US = "4.0us"
    ST_8_0_US = "8.0us"
    ST_16_0_US = "16.0us"


class CountMode(StrictEnum):
    TIMED = "Timed"
    CONTINUOUS = "Continuous"


class ShotMode(StrictEnum):
    ONE_SHOT = "OneShot"
    AUTO_COUNT = "AutoCount"


class LeakagePulseMode(StrictEnum):
    REAL = "Real"
    SIMULATED = "Simulated"


class InternalLeakCurrent(StrictEnum):
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
    model: A[SignalR[str], PvSuffix("DetectorModel")]
    firmware_version: A[SignalR[int], PvSuffix("FirmwareVersion")]

    temperature1: A[SignalR[float], PvSuffix("Temperature1")]
    temperature2: A[SignalR[float], PvSuffix("Temperature2")]
    temperature3: A[SignalR[float], PvSuffix("Temperature3")]
    zynq_temperature: A[SignalR[float], PvSuffix("ZynqTemperature")]
    high_voltage: A[SignalRW[float], PvSuffix.rbv("HighVoltage")]
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
        print(livetime)
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
        self.driver = GeRMDetectorIO(prefix, name=name)
        if plugins is not None:
            for plugin_name, plugin in plugins.items():
                setattr(self, plugin_name, plugin)
        self.add_detector_logics(GeRMTriggerLogic(self.driver))
        self.add_detector_logics(GeRMAcquireLogic(self.driver))
        self.hdf = NDFileHDF5IO(prefix + "MCA1:HDF1:", name="hdf")
        self.add_detector_logics(
            ADHDFDataLogic(
                NDArrayDescription(
                    [self.driver.num_energy_bins, self.driver.num_elements],
                    self.driver.data_type,
                    self.driver.color_mode,
                ),
                path_provider,
                self.hdf,
            )
        )
        self.add_config_signals(*config_sigs)
        super().__init__(name=name)
