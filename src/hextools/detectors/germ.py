"""Ophyd async support for the GeRM detector at HEX."""

import asyncio
from typing import Annotated as A

from ophyd_async.core import (
    DetectorArmLogic,
    DetectorDataLogic,
    DetectorTrigger,
    DetectorTriggerLogic,
    SignalR,
    SignalRW,
    StandardDetector,
    StandardReadable,
    TriggerInfo,
    set_and_wait_for_value,
)
from ophyd_async.core import StandardReadableFormat as Format
from ophyd_async.epics.core import EpicsDevice, PvSuffix


class GeRMDetectorIO(EpicsDevice, StandardReadable):
    """IO class for GeRM detector."""

    # count: A[SignalRW[bool], PvSuffix(".CNT")]
    mca: A[SignalR[float], PvSuffix(".MCA")]
    number_of_channels: A[SignalRW[int], PvSuffix(".NELM"), Format.CONFIG_SIGNAL]
    energy: A[SignalR[float], PvSuffix(".SPCTX")]
    gain: A[SignalRW[float], PvSuffix(".GAIN"), Format.CONFIG_SIGNAL]
    shaping_time: A[SignalRW[float], PvSuffix(".SHPT"), Format.CONFIG_SIGNAL]
    count_time: A[SignalRW[float], PvSuffix(".TP"), Format.CONFIG_SIGNAL]
    auto_time: A[SignalRW[float], PvSuffix(".TP1"), Format.CONFIG_SIGNAL]
    run_num: A[SignalRW[int], PvSuffix(".RUNNO")]
    fast_data_filename: A[SignalRW[str], PvSuffix(".FNAM"), Format.CONFIG_SIGNAL]
    operating_mode: A[SignalRW[int], PvSuffix(".MODE")]
    single_auto_toggle: A[SignalRW[int], PvSuffix(".CONT")]
    gmon: A[SignalR[float], PvSuffix(".GMON")]
    ip_addr: A[SignalRW[str], PvSuffix(".IPADDR")]
    temp_1: A[SignalR[float], PvSuffix(":Temp1")]
    temp_2: A[SignalR[float], PvSuffix(":Temp2")]
    fpga_cpu_temp: A[SignalR[float], PvSuffix(":ztmp")]
    calibration_file: A[SignalRW[str], PvSuffix(".CALF")]
    multi_file_supression: A[SignalRW[int], PvSuffix(".MFS")]
    tdc: A[SignalRW[int], PvSuffix(".TDC")]
    leakage_pulse: A[SignalRW[int], PvSuffix(".LOAO")]
    internal_leak_curr: A[SignalRW[int], PvSuffix(".EBLK")]
    pileup_rejection: A[SignalRW[int], PvSuffix(".PUEN")]
    test_pulse_aplitude: A[SignalRW[float], PvSuffix(".TPAMP")]
    channel: A[SignalRW[int], PvSuffix(".MONCH")]
    tdc_slope: A[SignalRW[float], PvSuffix(".TDS")]
    test_pulse_freq: A[SignalRW[float], PvSuffix(".TPFRQ")]
    tdc_mode: A[SignalRW[int], PvSuffix(".TDM")]
    test_pulce_enable: A[SignalRW[int], PvSuffix(".TPENB")]
    test_pulse_count: A[SignalRW[int], PvSuffix(".TPCNT")]
    input_polarity: A[SignalRW[int], PvSuffix(".POL")]
    voltage: A[SignalR[float], PvSuffix(":HV_RBV"), Format.CONFIG_SIGNAL]
    current: A[SignalR[float], PvSuffix(":HV_CUR")]
    peltier_2: A[SignalRW[float], PvSuffix(":P2")]
    peliter_2_current: A[SignalR[float], PvSuffix(":P2_CUR")]
    peltier_1: A[SignalRW[float], PvSuffix(":P1")]
    peltier_1_current: A[SignalR[float], PvSuffix(":P1_CUR")]
    hv_bias: A[SignalRW[float], PvSuffix(":HV"), Format.CONFIG_SIGNAL]
    ring_hi: A[SignalRW[float], PvSuffix(":DRFTHI")]
    ring_lo: A[SignalRW[float], PvSuffix(":DRFTLO")]
    channel_enabled: A[SignalRW[int], PvSuffix(".TSEN")]
    write_dir: A[SignalRW[str], PvSuffix(":write_dir"), Format.CONFIG_SIGNAL]
    file_name: A[SignalRW[str], PvSuffix(":file_name"), Format.CONFIG_SIGNAL]
    frame_num: A[SignalR[int], PvSuffix(":frame_num")]
    frame_shape: A[SignalR[int], PvSuffix(":frame_shape")]
    ioc_stage: A[SignalRW[str], PvSuffix(":stage")]
    count: A[SignalRW[bool], PvSuffix(":count")]


class GeRMArmLogic(DetectorArmLogic):
    """The arm logic for GeRM detector."""

    def __init__(self, driver: GeRMDetectorIO) -> None:
        self._driver = driver

    async def arm(self):
        await set_and_wait_for_value(self._driver.count, True, True)

    async def wait_for_idle(self):
        pass

    async def disarm(self):
        pass


class GeRMTriggerLogic(DetectorTriggerLogic):
    """The trigger logic for GeRM detector."""

    def __init__(self, driver: GeRMDetectorIO) -> None:
        self._driver = driver

    async def prepare_internal(self, num: int, livetime: float, deadtime: float):
        """Prepare the detector for an internal trigger.

        Parameters
        ----------
        livetime : float
            How long the exposure should be, 0 means what is currently set.
        deadtime : float
            How long between exposures, 0 means the shortest possible.
        """
        await self._driver.count_time.set(livetime)

    async def default_trigger_info(self) -> TriggerInfo:
        return TriggerInfo(
            trigger=DetectorTrigger.INTERNAL,
            livetime=0,
            deadtime=0,
            exposures_per_collection=1,
            collections_per_event=1,
            number_of_events=1,
        )


class GeRMDataLogic(DetectorDataLogic):
    """The data logic for GeRM detector."""


class GeRMDetector(StandardDetector):
    """The ophyd class for GeRM detector."""

    def __init__(self, prefix: str, name: str) -> None:
        self.driver = GeRMDetectorIO(prefix, name=name)
        self._trigger_logic = GeRMTriggerLogic(self.driver)
        self._arm_logic = GeRMArmLogic(self.driver)
        # self._data_logics = [GeRMDataLogic(self.driver)]
