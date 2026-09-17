"""Ophyd async support for Phantom VEO camera at HEX."""

import asyncio
from collections.abc import Sequence
from typing import Annotated as A

from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    DetectorTrigger,
    DetectorTriggerLogic,
    DeviceVector,
    OnOff,
    SignalR,
    SignalRW,
    StandardReadable,
    StrictEnum,
    SubsetEnum,
    TriggerInfo,
    YesNo,
    derived_signal_r,
    observe_value,
    set_and_wait_for_other_value,
)
from ophyd_async.epics.adcore import (
    ADAcquireLogic,
    ADBaseDataType,
    ADBaseIO,
    ADHDFDataLogic,
    ADMultipartDataLogic,
    ADWriterFactory,
    AreaDetector,
    NDFileHDF5IO,
    NDPluginBaseIO,
    NDProcessIO,
    prepare_exposures_per_collection,
    default_trigger_info_from_detector_settings,
)
from ophyd_async.epics.core import EpicsDevice, PvSuffix, epics_signal_rw_rbv


class PhantomDownloadFrameMode(StrictEnum):
    """Mode for selecting frames to download from the camera."""

    FIRST_AND_LAST = "First and Last"
    UNIVERSAL = "Universal"


class PhantomDownloadSpeed(StrictEnum):
    """Speed for downloading frames from the camera."""

    ONE_GIGABIT = "1 Gb/s"
    TEN_GIGABIT = "10 Gb/s"


class PhantomSettingsSlot(StrictEnum):
    """Settings slots available on the camera for saving and loading configurations."""

    SLOT_1 = "1"
    SLOT_2 = "2"
    SLOT_3 = "3"
    SLOT_4 = "4"
    SLOT_5 = "5"


class PhantomExtSyncType(StrictEnum):
    """Types of external synchronization available for triggering the camera."""

    FREE_RUN = "FREE-RUN"  # Software internal trigger
    FSYNC = "FSYNC"  # External frame sync signal, typically used for edge triggering
    IRIG = "IRIG"
    VIDEO = "VIDEO"


class PhantomTrigEdge(StrictEnum):
    """Trigger edge for external triggering."""

    FALLING = "FALLING"
    RISING = "RISING"


class PhantomReadySignal(StrictEnum):
    """Defines behavior of the camera's ready signal output."""

    RECORDING = "RECORDING"
    TRIGGER = "TRIGGER"


class PhantomAuxPinMode(SubsetEnum):
    """Defines possible modes for camera's auxiliary pins."""

    FSYNCIO = "FSYNCIO"
    STROBE = "STROBE"
    EVENT = "EVENT"
    MEMGATE = "MEMGATE"
    READY = "READY"
    TCOUT = "TCOUT"
    PRETRIG = "PRETRIG"
    AUXTRIG = "AUXTRIG"
    MSTROBE = "MSTROBE"
    AUTOTRIG = "AUTOTRIG"
    SWTRIG = "SWTRIG"
    RECORDING = "RECORDING"


class PhantomFanState(StrictEnum):
    """Defines possible states for the camera's fan."""

    ON = "Fan On"
    QUIET = "Fan Quiet"


class PhantomPixelDataFormat(StrictEnum):
    """Defines the pixel data format for the camera."""

    EIGHT = "8"
    EIGHT_R = "8R"
    P_SIXTEEN = "P16"
    P_SIXTEEN_R = "P16R"
    P_TEN = "P10"
    P_TWELVE_L = "P12L"


class PhantomCineIO(EpicsDevice, StandardReadable):
    cine_name: A[SignalR[str], PvSuffix("Name_RBV")]
    width: A[SignalR[int], PvSuffix("Width_RBV")]
    height: A[SignalR[int], PvSuffix("Height_RBV")]
    frame_count: A[SignalR[int], PvSuffix("FrameCount_RBV")]
    first_frame: A[SignalR[int], PvSuffix("FirstFrame_RBV")]
    last_frame: A[SignalR[int], PvSuffix("LastFrame_RBV")]
    invalid: A[SignalR[bool], PvSuffix("State_RBV.B0")]
    complete_and_valid: A[SignalR[bool], PvSuffix("State_RBV.B1")]
    waiting_for_trigger: A[SignalR[bool], PvSuffix("State_RBV.B2")]
    trigger_recieved: A[SignalR[bool], PvSuffix("State_RBV.B3")]
    ready: A[SignalR[bool], PvSuffix("State_RBV.B4")]
    default_copy: A[SignalR[bool], PvSuffix("State_RBV.B5")]
    can_accept_trigger: A[SignalR[bool], PvSuffix("State_RBV.B6")]
    preview_cine: A[SignalR[bool], PvSuffix("State_RBV.B7")]
    active_cine: A[SignalR[bool], PvSuffix("State_RBV.B8")]
    cine_content_saved: A[SignalR[bool], PvSuffix("State_RBV.B9")]


class PhantomIO(ADBaseIO):
    """IO class for ADPhantom driver.

    Mirrors API found in ADPhantomApp/Db/phantomCamera.template
    """

    acq_notify: A[SignalR[float], PvSuffix("AcqNotify")]
    acquire_time_ms: A[SignalRW[float], PvSuffix.rbv("AcquireTimeMs")]
    partition_cines: A[SignalRW[int], PvSuffix("PartitionCines")]
    cine_count: A[SignalR[int], PvSuffix("CineCount_RBV")]
    max_frame_count: A[SignalR[int], PvSuffix("MaxFrameCount_RBV")]
    total_frame_count: A[SignalR[int], PvSuffix("TotalFrameCount_RBV")]
    post_trig_frames: A[SignalRW[int], PvSuffix.rbv("PostTrigFrames")]
    auto_advance: A[SignalRW[OnOff], PvSuffix.rbv("AutoAdvance")]
    auto_save: A[SignalRW[OnOff], PvSuffix.rbv("AutoSave")]
    auto_restart: A[SignalRW[OnOff], PvSuffix.rbv("AutoRestart")]
    auto_bref: A[SignalRW[OnOff], PvSuffix.rbv("AutoBref")]
    cine_name: A[SignalRW[str], PvSuffix.rbv("CineName")]
    selected_cine: A[SignalRW[int], PvSuffix.rbv("SelectedCine")]
    cine_frame_count: A[SignalR[int], PvSuffix("CineFrameCount_RBV")]
    cine_first_frame: A[SignalR[int], PvSuffix("CineFirstFrame_RBV")]
    cine_last_frame: A[SignalR[int], PvSuffix("CineLastFrame_RBV")]
    download_start_frame: A[SignalRW[int], PvSuffix.rbv("DownloadStartFrame")]
    download_end_frame: A[SignalRW[int], PvSuffix.rbv("DownloadEndFrame")]
    download_start_cine: A[SignalRW[int], PvSuffix.rbv("DownloadStartCine")]
    download_end_cine: A[SignalRW[int], PvSuffix.rbv("DownloadEndCine")]
    download_frame_mode: A[
        SignalRW[PhantomDownloadFrameMode], PvSuffix.rbv("DownloadFrameMode")
    ]
    download_speed: A[SignalRW[PhantomDownloadSpeed], PvSuffix.rbv("DownloadSpeed")]
    dropped_packets: A[SignalR[int], PvSuffix("DroppedPackets_RBV")]
    mark_cine_saved: A[SignalRW[YesNo], PvSuffix.rbv("MarkCineSaved")]
    download: A[SignalRW[int], PvSuffix("Download")]
    abort_download: A[SignalRW[int], PvSuffix("AbortDownload")]
    delete_start_cine: A[SignalRW[int], PvSuffix.rbv("DeleteStartCine")]
    delete_end_cine: A[SignalRW[int], PvSuffix.rbv("DeleteEndCine")]
    download_count: A[SignalR[int], PvSuffix("DownloadCount_RBV")]
    delete: A[SignalRW[int], PvSuffix("Delete")]
    send_software_trigger: A[SignalRW[int], PvSuffix("SendSoftwareTrigger")]
    perform_csr: A[SignalRW[int], PvSuffix("PerformCSR")]
    csr_count: A[SignalR[int], PvSuffix("CSRCount_RBV")]
    settings_save: A[SignalRW[int], PvSuffix("SettingsSave")]
    settings_load: A[SignalRW[int], PvSuffix("SettingsLoad")]
    settings_slot: A[SignalRW[PhantomSettingsSlot], PvSuffix.rbv("SettingsSlot")]
    preview: A[SignalRW[int], PvSuffix.rbv("Preview")]
    sensor_temp: A[SignalR[int], PvSuffix("SensorTemp_RBV")]
    thermo_power: A[SignalR[int], PvSuffix("ThermoPower_RBV")]
    camera_temp: A[SignalR[int], PvSuffix("CameraTemp_RBV")]
    fan_power: A[SignalR[int], PvSuffix("FanPower_RBV")]
    ext_sync_type: A[SignalRW[PhantomExtSyncType], PvSuffix.rbv("ExtSyncType")]
    edr: A[SignalRW[int], PvSuffix.rbv("EDR")]
    frame_delay: A[SignalRW[int], PvSuffix.rbv("FrameDelay")]
    trigger_edge: A[SignalRW[PhantomTrigEdge], PvSuffix.rbv("TriggerEdge")]
    trigger_filter: A[SignalRW[int], PvSuffix.rbv("TriggerFilter")]
    ready_signal: A[SignalRW[PhantomReadySignal], PvSuffix.rbv("ReadySignal")]
    quiet_fan: A[SignalRW[PhantomFanState], PvSuffix.rbv("QuietFan")]
    sync_clock: A[SignalRW[int], PvSuffix("SyncClock")]
    select_pixel_data_format: A[
        SignalRW[PhantomPixelDataFormat], PvSuffix.rbv("SelectPixelDataFormat")
    ]
    frame_read_speed: A[SignalR[int], PvSuffix("FrameReadSpeed_RBV")]
    invalid: A[SignalR[bool], PvSuffix("State_RBV.B0")]
    complete_and_valid: A[SignalR[bool], PvSuffix("State_RBV.B1")]
    waiting_for_trigger: A[SignalR[bool], PvSuffix("State_RBV.B2")]
    trigger_received: A[SignalR[bool], PvSuffix("State_RBV.B3")]

    def __init__(self, prefix: str, name: str = "", num_cines: int = 63):
        super().__init__(prefix, name=name)
        # self.aux_pins = DeviceVector(
        #     {
        #         i: epics_signal_rw_rbv(
        #             PhantomAuxPinMode, prefix + f"Aux{i}PinMode", name=f"aux_pin{i}"
        #         )
        #         for i in (1, 2, 4)
        #     },
        #     name="aux_pins",
        # )

        self.cines = DeviceVector(
            {i: PhantomCineIO(prefix + f"C{i}:") for i in range(1, num_cines + 1)},
            name="cines",
        )

        # IOC does not provide these signals, so make them derived here
        self.total_download_frames = derived_signal_r(
            self.get_total_downloaded_frames,
            start=self.download_start_frame,
            end=self.download_end_frame,
        )
        self.download_data_type = derived_signal_r(
            self.get_downloaded_dtype, pixel_token=self.select_pixel_data_format
        )

    def get_downloaded_dtype(
        self, pixel_token: PhantomPixelDataFormat
    ) -> ADBaseDataType:
        """Convert the pixel data format token to the corresponding ADBaseDataType.

        Parameters
        ----------
        pixel_token : PhantomPixelDataFormat
            The pixel data format token read from the camera.

        Returns
        -------
        ADBaseDataType
            The corresponding ADBaseDataType to use for the downloaded frames.
        """
        if pixel_token in (
            PhantomPixelDataFormat.EIGHT,
            PhantomPixelDataFormat.EIGHT_R,
        ):
            return ADBaseDataType.UINT8
        return ADBaseDataType.UINT16

    def get_total_downloaded_frames(self, start: int, end: int) -> int:
        """Get the number of frames to download based on start/end frame indices.

        Parameters
        ----------
        start : int
            The starting frame index for the download.
        end : int
            The ending frame index for the download.

        Returns
        -------
        int
            The total number of frames to download, calculated as end - start + 1.
            If end is less than start, returns 0.
        """
        if end < start:
            return 0

        return end - start + 1


class PhantomTriggerLogic(DetectorTriggerLogic):
    """Trigger logic for the Phantom camera."""

    def __init__(self, driver: PhantomIO, process_plugin: NDProcessIO | None = None):
        self.driver = driver
        self.process_plugin = process_plugin

    def config_sigs(self) -> set[SignalR]:
        """Return the signals that should appear in read_configuration.

        Returns
        -------
        set[SignalR]
            The set of signals to include in the configuration.
        """
        return {
            self.driver.acquire_time_ms,
            self.driver.ext_sync_type,
            self.driver.model,
            self.driver.manufacturer,
            self.driver.post_trig_frames,
            self.driver.cine_name,
            self.driver.selected_cine,
            self.driver.download_start_frame,
            self.driver.download_end_frame,
            self.driver.select_pixel_data_format,
        }

    async def setup_download(self, num: int):
        """Given a total number of frames to download, set up the start/end indices.

        If post trigger frames are greater than or equal to the total number of frames
        to download, then the download will be set up to download all frames from the
        event trigger until num. If post trigger frames are less than the total number
        of frames to download, then the download will be set up to download the
        specified number of post trigger frames, and the remaining pre trigger frames.

        Parameters
        ----------
        num : int
            The total num of frames to download, incl both pre and post trigger frames.

        Raises
        ------
        ValueError
            If num is less than or equal to 0.
        """
        if num <= 0:
            raise ValueError("Number of frames to download must be greater than 0!")

        post_trig_frames = await self.driver.post_trig_frames.get_value()

        if num < post_trig_frames:
            await asyncio.gather(
                self.driver.download_start_frame.set(0),
                self.driver.download_end_frame.set(num - 1),
            )
        else:
            await asyncio.gather(
                self.driver.download_start_frame.set((num - post_trig_frames) * -1),
                self.driver.download_end_frame.set(post_trig_frames - 1),
            )

    async def prepare_internal(self, num: int, livetime: float, deadtime: float):
        """Prepare the detector to take internally triggered exposures.

        Parameters
        ----------
        num : int
            The total num of frames to download, incl both pre and post trigger frames.
        livetime : float
            How long the exposure should be, in s, 0 means what is currently set.
        deadtime : float
            How long between exposures, in s, 0 means the shortest possible.
        """
        coros = [self.driver.ext_sync_type.set(PhantomExtSyncType.FREE_RUN)]
        if livetime != 0:
            coros.append(self.driver.acquire_time_ms.set(livetime * 1000))
        await asyncio.gather(*coros)
        await self.setup_download(num)

    async def prepare_edge(self, num: int, livetime: float):
        """Prepare the detector to take external edge triggered exposures.

        Parameters
        ----------
        num : int
            The total num of frames to download, incl both pre and post trigger frames.
        livetime : float
            How long the exposure should be, in s, 0 means what is currently set.
        """
        coros = [self.driver.ext_sync_type.set(PhantomExtSyncType.FSYNC)]
        if livetime != 0:
            coros.append(self.driver.acquire_time_ms.set(livetime * 1000))
        await asyncio.gather(*coros)
        await self.setup_download(num)

    async def prepare_exposures_per_collection(self, exposures_per_collection: int):
        """Prepare the process plugin for the specified number of exposures per collection.

        Parameters
        ----------
        exposures_per_collection : int
            The number of exposures to take per collection.
        """
        if self.process_plugin is not None:
            await prepare_exposures_per_collection(
                self.process_plugin, exposures_per_collection=exposures_per_collection
            )

    async def default_trigger_info(self) -> TriggerInfo:
        """Fallback for the default TriggerInfo in plans without prepare.

        Returns
        -------
        TriggerInfo
            A TriggerInfo with default values for the Phantom camera.
        """
        trigger_mode = DetectorTrigger.INTERNAL
        if await self.driver.ext_sync_type.get_value() == PhantomExtSyncType.FSYNC:
            trigger_mode = DetectorTrigger.EXTERNAL_EDGE
        default_trig_info = await default_trigger_info_from_detector_settings(
            self.driver.total_download_frames,
            self.process_plugin,
            detector_trigger=trigger_mode,
        )
        return default_trig_info


class PhantomAcquireLogic(ADAcquireLogic):
    """Acquire logic for the Phantom camera."""

    driver: PhantomIO

    def __init__(self, driver: PhantomIO):
        super().__init__(driver, driver.acquire)
        self.driver = driver

    async def start_acquiring(self):
        """Start the acquisition and the image data download.

        Start the acquisition, wait for the event trigger,
        wait for any post trigger frames, and start the download.

        Raises
        ------
        RuntimeError
            If acquisition stops before we receive the event trigger.
        TimeoutError
            If we timeout waiting for the cine to be marked as valid.
        ValueError
            If the number of post trigger frames recorded is not what was expected.
        """
        # Start the acquisition, and wait for waiting for trigger to be True
        await set_and_wait_for_other_value(
            self.driver.acquire,
            True,
            self.driver.waiting_for_trigger,
            True,
            timeout=DEFAULT_TIMEOUT,
        )

        # Wait for trigger_received to go to 1. If trigger_received does not go
        # to 1 within the timeout, check if acquisition stopped, and if so raise
        # a timeout error indicating acquisition stopped while waiting for trigger.
        # Otherwise, if acq is still running, keep waiting for trigger_received to
        # go to 1.
        while True:
            try:
                async for trigger_received in observe_value(
                    self.driver.trigger_received, done_timeout=DEFAULT_TIMEOUT
                ):
                    if trigger_received:
                        break
                break
            except TimeoutError as exc:
                acquiring = await self.driver.acquire.get_value()
                if not acquiring:
                    raise RuntimeError(
                        "Acquisition stopped while waiting for event trigger!"
                    ) from exc

        # After we recieve the event trigger, we want to wait until the number of
        # post trigger frames have been recorded before we start the download.
        target_post_trig = await self.driver.post_trig_frames.get_value()
        try:
            async for actual_post_trig in observe_value(
                self.driver.array_counter, done_timeout=DEFAULT_TIMEOUT
            ):
                if target_post_trig == actual_post_trig:
                    break
        except TimeoutError as exc:
            (
                actual_post_trig,
                complete_and_valid,
                trigger_received,
            ) = await asyncio.gather(
                self.driver.array_counter.get_value(),
                self.driver.complete_and_valid.get_value(),
                self.driver.trigger_received.get_value(),
            )
            if trigger_received == 1 and complete_and_valid != 1:
                raise TimeoutError(
                    "Received event trigger, but writing to cine was not completed!"
                ) from exc
            elif actual_post_trig != target_post_trig:
                raise ValueError(
                    f"Expected number of post trig frames {target_post_trig} "
                    f"does not match actual number {actual_post_trig}"
                ) from exc

        # If we recieved the trigger, we know at this point how many frames we'll have access to,
        # and how many we want to download. If we are trying to DL more than we have available,
        # raise a RuntimeError.
        available_frames, total_download_frames = await asyncio.gather(
            self.driver.total_frame_count.get_value(),
            self.driver.total_download_frames.get_value(),
        )
        if total_download_frames > available_frames:
            raise RuntimeError(
                f"Requested {total_download_frames} frames to download, but only {available_frames} are available!"
            )

        # Finally, start the download
        await self.driver.download.set(True)

    async def wait_for_idle(self):
        """Wait for the camera to finish downloading and return to idle state.

        Raises
        ------
        TimeoutError
            If we timeout waiting for the download to complete.
        """
        # First, make sure our arm process is complete.
        if self.acquire_status:
            await self.acquire_status

        selected_cine_num, last_download_count = await asyncio.gather(
            self.driver.selected_cine.get_value(),
            self.driver.download_count.get_value(),
        )
        selected_cine = self.driver.cines[selected_cine_num]

        # As long as our download counter is counting up and has not reached the target
        # number of frames, keep waiting. If we timeout, check if the download count
        # has increased since the last time we checked, and if so keep waiting,
        # otherwise raise a timeout error.
        while True:
            try:
                async for saved in observe_value(
                    selected_cine.cine_content_saved, done_timeout=DEFAULT_TIMEOUT
                ):
                    if saved:
                        return
            except TimeoutError as err:
                # download_counter = await self.driver.download_count.get_value()
                # if download_counter == 0:
                #     raise TimeoutError(
                #         "Download counter stopped incrementing and cine was not marked as saved!"
                #     ) from err

                current, saved = await asyncio.gather(
                    self.driver.download_count.get_value(),
                    selected_cine.cine_content_saved.get_value(),
                )
                print(
                    f"Last value: {last_download_count}, Current value: {current}, Saved: {saved}"
                )
                if saved:
                    return
                if current <= last_download_count:
                    raise TimeoutError(
                        "Download counter stopped incrementing and cine was not marked as saved!"
                    ) from err
                last_download_count = current


class PhantomDetector(AreaDetector[PhantomIO]):
    """Detector class for Phantom cameras."""

    hdf: NDFileHDF5IO | None = None

    def __init__(
        self,
        prefix: str,
        *writer_factories: ADWriterFactory,
        driver_suffix: str = "cam1:",
        proc_suffix: str | None = "Proc1:",
        plugins: dict[str, NDPluginBaseIO] | None = None,
        config_sigs: Sequence[SignalR] = (),
        name: str = "",
    ) -> None:
        driver = PhantomIO(prefix + driver_suffix)
        process_plugin = None if not proc_suffix else NDProcessIO(prefix + proc_suffix)
        if plugins is None:
            plugins = {}
        if process_plugin is not None:
            plugins["proc"] = process_plugin
        super().__init__(
            driver,
            prefix,
            *writer_factories,
            acquire_logic=PhantomAcquireLogic(driver),
            trigger_logic=PhantomTriggerLogic(driver, process_plugin=process_plugin),
            plugins=plugins,
            config_sigs=config_sigs,
            name=name,
        )

        # Override default data type signal in datalogic description
        for data_logic in self._data_logics:
            if isinstance(data_logic, (ADHDFDataLogic, ADMultipartDataLogic)):
                data_logic.array_description.data_type_signal = (
                    self.driver.download_data_type
                )
