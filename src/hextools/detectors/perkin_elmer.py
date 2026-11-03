"""PerkinElmer detector support for HEX beamline."""

from collections.abc import Sequence
from typing import Annotated as A

from ophyd_async.core import EnableDisable, SignalR, SignalRW, SignalX, StrictEnum
from ophyd_async.epics.adcore import (
    ADBaseIO,
    ADContAcqAcquireLogic,
    ADContAcqTriggerLogic,
    ADMultipartDataLogic,
    ADWriterFactory,
    AreaDetector,
    NDArrayDescription,
    NDCircularBuffIO,
    NDPluginBaseIO,
    NDProcessIO,
)
from ophyd_async.epics.core import PvSuffix


class PerkinElmerTriggerMode(StrictEnum):
    """Trigger mode for ADPerkinElmer detector."""

    INTERNAL = "Internal"
    EXTERNAL = "External"
    FREE_RUNNING = "Free Running"
    SOFT_TRIGGER = "Soft Trigger"


class PerkinElmerIO(ADBaseIO):
    """Mirrors the interface provided by ADPerkinElmer/db/PerkinElmer.template."""

    trigger_mode: A[SignalRW[PerkinElmerTriggerMode], PvSuffix.rbv("TriggerMode")]

    num_offset_frames: A[SignalRW[int], PvSuffix.rbv("PENumOffsetFrames")]
    use_offset: A[SignalRW[EnableDisable], PvSuffix.rbv("PEUseOffset")]

    acquire_gain: A[SignalX, PvSuffix("PEAcquireGain")]
    num_gain_frames: A[SignalRW[int], PvSuffix.rbv("PENumGainFrames")]
    use_gain: A[SignalRW[EnableDisable], PvSuffix.rbv("PEUseGain")]

    use_pixel_correction: A[
        SignalRW[EnableDisable], PvSuffix.rbv("PEUsePixelCorrection")
    ]


class PerkinElmerDetector(AreaDetector[PerkinElmerIO]):
    """Continuously acquiring ADPerkinElmer detector using a circular buffer plugin.

    :param prefix: EPICS PV prefix for the detector
    :param writer_factories: Factories for file writer plugins and their data logics
    :param driver_suffix: Suffix for the driver PV, defaults to "cam1:"
    :param cb_suffix: Suffix for the circular buffer plugin PV, defaults to "CB1:"
    :param proc_suffix: If provided, an NDProcessIO plugin is created at this suffix
    :param plugins: Additional areaDetector plugins to include
    :param config_sigs: Additional signals to include in configuration
    :param name: Name for the detector device
    """

    def __init__(
        self,
        prefix: str,
        *writer_factories: ADWriterFactory,
        driver_suffix: str = "cam1:",
        cb_suffix: str = "CB1:",
        proc_suffix: str | None = None,
        plugins: dict[str, NDPluginBaseIO] | None = None,
        config_sigs: Sequence[SignalR] = (),
        name: str = "",
    ) -> None:
        driver = PerkinElmerIO(prefix + driver_suffix)
        cb_plugin = NDCircularBuffIO(prefix + cb_suffix)
        proc_plugin = NDProcessIO(prefix + proc_suffix) if proc_suffix else None
        super().__init__(
            driver,
            prefix,
            *writer_factories,
            acquire_logic=ADContAcqAcquireLogic(driver, cb_plugin),
            trigger_logic=ADContAcqTriggerLogic(driver, cb_plugin, proc_plugin),
            plugins=(plugins or {})
            | {"cb": cb_plugin}
            | ({"proc": proc_plugin} if proc_plugin else {}),
            config_sigs=config_sigs,
            name=name,
        )
    def save_as_tiff(self) -> None:
        if self._data_logics and isinstance(self._data_logics[0], ADMultipartDataLogic):
            return
        self._data_logics = ()
        self.add_detector_logics(
            ADWriterFactory.tiff()
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
            GeRMStatsDataLogic(self.stats1.total, self.stats1),
        )
