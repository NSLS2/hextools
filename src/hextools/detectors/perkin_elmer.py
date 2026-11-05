"""PerkinElmer detector support for HEX beamline."""

from collections.abc import Sequence
from typing import Annotated as A

from ophyd_async.core import (
    EnableDisable,
    PathProvider,
    SignalR,
    SignalRW,
    StrictEnum,
    TriggerableCommand,
)
from ophyd_async.epics.adcore import (
    ADBaseIO,
    ADContAcqAcquireLogic,
    ADContAcqTriggerLogic,
    ADHDFDataLogic,
    ADMultipartDataLogic,
    ADWriterFactory,
    AreaDetector,
    NDArrayDescription,
    NDCircularBuffIO,
    NDFileHDF5IO,
    NDPluginBaseIO,
    NDPluginFileIO,
    NDProcessIO,
)
from ophyd_async.epics.core import PvSuffix


class PerkinElmerTriggerMode(StrictEnum):
    """Trigger mode for ADPerkinElmer detector."""

    INTERNAL = "Internal"
    EXTERNAL = "External"
    FREE_RUNNING = "Free Running"
    SOFT_TRIGGER = "Soft Trigger"


class PerkinElmerSyncMode(StrictEnum):
    """Sync mode for ADPerkinElmer detector."""

    DDD_CLEAR = "DDD Clear"
    DDD_NO_CLEAR = "DDD No Clear"
    LINEWISE = "Linewise"
    FRAMEWISE = "Framewise"
    AUTO_TRIGGER = "AutoTrigger"


class PerkinElmerGain(StrictEnum):
    """Gain mode for ADPerkinElmer detector."""

    PF_0_25 = "0.25pF"
    PF_0_5 = "0.5pF"
    PF_1 = "1pF"
    PF_2 = "2pF"
    PF_4 = "4pF"
    PF_8 = "8pF"


class PerkinElmerIO(ADBaseIO):
    """Mirrors the interface provided by ADPerkinElmer/db/PerkinElmer.template."""

    trigger_mode: A[SignalRW[PerkinElmerTriggerMode], PvSuffix.rbv("TriggerMode")]

    corrections_directory: A[SignalRW[str], PvSuffix("PECorrectionsDir")]

    num_offset_frames: A[SignalRW[int], PvSuffix("PENumOffsetFrames")]
    offset_constant: A[SignalRW[int], PvSuffix("PEOffsetConstant")]
    offset_available: A[SignalR[bool], PvSuffix("PEOffsetAvailable")]
    use_offset: A[SignalRW[EnableDisable], PvSuffix("PEUseOffset")]
    gain: A[SignalRW[PerkinElmerGain], PvSuffix.rbv("PEGain")]

    gain_available: A[SignalR[bool], PvSuffix("PEGainAvailable")]
    acquire_gain: A[TriggerableCommand, PvSuffix("PEAcquireGain")]
    num_gain_frames: A[SignalRW[int], PvSuffix("PENumGainFrames")]
    use_gain: A[SignalRW[EnableDisable], PvSuffix("PEUseGain")]
    gain_file: A[SignalRW[str], PvSuffix("PEGainFile")]
    load_gain_file: A[TriggerableCommand, PvSuffix("PELoadGainFile")]

    use_pixel_correction: A[SignalRW[EnableDisable], PvSuffix("PEUsePixelCorrection")]
    bad_pixel_file: A[SignalRW[str], PvSuffix("PEPixelCorrectionFile")]
    load_bad_pixel_file: A[TriggerableCommand, PvSuffix("PELoadPixelCorrection")]

    skip_frames: A[SignalRW[EnableDisable], PvSuffix.rbv("PESkipFrames")]
    num_frames_to_skip: A[SignalRW[int], PvSuffix.rbv("PENumFramesToSkip")]
    num_frame_buffers: A[SignalRW[int], PvSuffix.rbv("PENumFrameBuffers")]
    sync_mode: A[SignalRW[PerkinElmerSyncMode], PvSuffix.rbv("PESyncMode")]


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
        path_provider: PathProvider,
        driver_suffix: str = "cam1:",
        cb_suffix: str = "CB1:",
        proc_suffix: str = "Proc1:",
        tiff_suffix: str = "TIFF1:",
        plugins: dict[str, NDPluginBaseIO] | None = None,
        config_sigs: Sequence[SignalR] = (),
        name: str = "",
    ) -> None:
        self._path_provider = path_provider
        driver = PerkinElmerIO(prefix + driver_suffix)
        cb_plugin = NDCircularBuffIO(prefix + cb_suffix)
        proc_plugin = NDProcessIO(prefix + proc_suffix, tiff_suffix="TIFF:")
        trans_plugin = NDPluginBaseIO(prefix + "Trans1:")
        tiff_plugin = NDPluginFileIO(prefix + tiff_suffix)

        self._array_description = NDArrayDescription(
            shape_signals=[
                driver.array_size_z,
                driver.array_size_y,
                driver.array_size_x,
            ],
            data_type_signal=driver.data_type,
            color_mode_signal=driver.color_mode,
        )

        super().__init__(
            driver,
            prefix,
            ADWriterFactory.hdf(path_provider, hinted=False),
            acquire_logic=ADContAcqAcquireLogic(driver, cb_plugin),
            trigger_logic=ADContAcqTriggerLogic(driver, cb_plugin, proc_plugin),
            plugins=(plugins or {})
            | {"cb": cb_plugin}
            | {"proc": proc_plugin}
            | {"trans": trans_plugin}
            | {"tiff": tiff_plugin},
            config_sigs=config_sigs,
            name=name,
        )

    def save_as_tiff(self) -> None:
        if self._data_logics and isinstance(self._data_logics[0], ADMultipartDataLogic):
            return
        sans_writer = self._data_logics[1:]
        self._data_logics = ()
        self.add_detector_logics(
            ADMultipartDataLogic(
                self._array_description,
                self._path_provider,
                self.get_plugin("tiff", NDPluginFileIO),
                ".tiff",
                "multipart/related;type=image/tiff",
                hinted=False,
            )
        )
        self.add_detector_logics(*sans_writer)

    def save_as_hdf(self) -> None:
        if self._data_logics and isinstance(self._data_logics[0], ADHDFDataLogic):
            return
        sans_writer = self._data_logics[1:]
        self._data_logics = ()
        self.add_detector_logics(
            ADHDFDataLogic(
                self._array_description,
                self._path_provider,
                self.get_plugin("hdf", NDFileHDF5IO),
                hinted=False,
            )
        )
        self.add_detector_logics(*sans_writer)
