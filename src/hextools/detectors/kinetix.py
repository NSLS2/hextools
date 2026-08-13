"""
HEX beamline Kinetix detector device definitions (ophyd-async).

Devices defined here:
    - kinetix1  : Kinetix camera #1  (XF:27ID1-BI{Kinetix-Det:1})
    - kinetix3  : Kinetix camera #3  (XF:27ID1-BI{Kinetix-Det:3})

Ported from NSLS2/hex-ob ``lib/detectors.py``, where it was developed and
mock/sim-tested together with the tomography plans.  Written against the
released 0.19 detector-assembly API; runs unchanged under the 0.21 line
this repo pins.  A detector is assembled from building blocks:

    KinetixDetector(prefix, *writer_factories)

The ``ADWriterFactory`` supplies the file-writer plugin IO (``NDFileHDF5IO``
on ``HDF1:``) and a "data logic" that decides how frames are captured and
described.  The ``PathProvider`` that decides *where* files land is captured
inside the data-logic factory — there is no ``set_path_provider()`` on the
detector, so plans that let the scientist choose an output directory use the
``SettablePathProvider`` below (see ``hextools/plans/tomography/alignment_scan.py``).

HEX specifics (same workarounds as hex-profile-collection
``startup/10-kinetix.py``):
    - SWMR mode is disabled (not supported at HEX).
    - The HDF flush-now signal is not used (it can lock the HDF plugin).
    - File close waits up to 60 s (closing on network storage routinely
      exceeds the default 10 s timeout).
    - After every scan (or abort) the camera is returned to
      Continuous / Internal mode so the live-view image stream restarts.
"""

from pathlib import Path

from ophyd_async.core import (
    AsyncStatus,
    PathInfo,
    StreamableDataProvider,
    set_and_wait_for_other_value,
    wait_for_value,
)
from typing import Annotated as A

from ophyd_async.core import SignalR, SignalRW
from ophyd_async.epics.adcore import (
    ADHDFDataLogic,
    ADImageMode,
    ADWriterFactory,
    NDFileHDF5IO,
    NDPluginBaseIO,
)
from ophyd_async.epics.core import PvSuffix
from ophyd_async.epics.adkinetix import KinetixDetector, KinetixTriggerMode
from ophyd_async.epics.core import stop_busy_record


class SettablePathProvider:
    """
    A PathProvider whose target directory / filename can be changed between
    scans.

    ophyd-async bakes the path provider into the detector at construction
    time; this class is the small extra piece that lets a *plan* accept an
    ``output_dir`` argument and point the already-built detector at it:

        detector.path_provider.set("/nsls2/data/hex/.../scan_00001", "proj")
    """

    def __init__(self, directory: str | Path = "/tmp", filename: str = "proj"):
        self._directory = Path(directory)
        self._filename = filename

    def set(self, directory: str | Path, filename: str | None = None) -> None:
        """Point subsequent scans at *directory* (and optionally rename files)."""
        self._directory = Path(directory)
        if filename is not None:
            self._filename = filename

    def __call__(self, device_name: str | None = None) -> PathInfo:
        # create_dir_depth=-4 lets the writer create up to 4 missing directory
        # levels (e.g. tomography/<base_folder>/scan_NNNNN under the proposal).
        return PathInfo(
            directory_path=self._directory,
            filename=self._filename,
            create_dir_depth=-4,
        )


def set_output_dir(detector, output_dir, filename: str = "proj") -> Path:
    """
    Point *detector*'s settable path provider at *output_dir*.

    The shared front half of every plan that takes an ``output_dir``
    argument (equivalent of the old ``camera.set_hdf_file_path(...)``).
    Raises TypeError with guidance if the detector wasn't built by
    make_kinetix (or doesn't otherwise carry a settable provider).
    """
    output_path = Path(output_dir)
    path_provider = getattr(detector, "path_provider", None)
    if path_provider is None or not hasattr(path_provider, "set"):
        detector_label = getattr(detector, "name", type(detector).__name__)
        raise TypeError(
            f"{detector_label} has no settable path provider; build it with "
            "hextools.detectors.kinetix.make_kinetix so the plan can direct its output "
            "to output_dir."
        )
    path_provider.set(output_path, filename)
    return output_path


class HEXProcPluginIO(NDPluginBaseIO):
    """NDPluginProcess with the recursive-filter (frame averaging) fields.

    Mirrors the PVs the legacy scripts drove on ``Proc1:`` for the
    frame-averaged variants (NDPluginProcess.template).
    """

    # Enum-backed fields are typed as their choice strings (Disable/Enable,
    # No/Yes, RecursiveAve/..., Every array/Array N only) — verified against
    # the live NDPluginProcess template; FilterType has no _RBV.
    enable_filter: A[SignalRW[str], PvSuffix.rbv("EnableFilter")]
    filter_type: A[SignalRW[str], PvSuffix("FilterType")]
    num_filter: A[SignalRW[int], PvSuffix.rbv("NumFilter")]
    reset_filter: A[SignalRW[str], PvSuffix("ResetFilter")]
    auto_reset_filter: A[SignalRW[str], PvSuffix.rbv("AutoResetFilter")]
    filter_callbacks: A[SignalRW[str], PvSuffix.rbv("FilterCallbacks")]
    # Flat-field correction (live-view normalization mode):
    save_flat_field: A[SignalRW[str], PvSuffix("SaveFlatField")]        # No/Yes
    enable_flat_field: A[SignalRW[str], PvSuffix.rbv("EnableFlatField")]  # Disable/Enable
    valid_flat_field: A[SignalR[str], PvSuffix("ValidFlatField_RBV")]   # Invalid/Valid


class HEXADHDFDataLogic(ADHDFDataLogic):
    """HDF data logic with the HEX workarounds: no SWMR, no flush-now, 60 s close."""

    async def prepare_unbounded(self, datakey_name: str) -> StreamableDataProvider:
        provider = await super().prepare_unbounded(datakey_name)
        await self.writer.swmr_mode.set(False)
        # Flushing via FlushNow occasionally locks the HDF plugin at HEX.
        provider.flush_signal = None
        # The plugin resets NumCaptured when capture starts (just done by
        # super()), but the monitor-cached readback can still hold the
        # PREVIOUS scan's count when ophyd-async takes its baseline — which
        # then fails kickoff with "requested N:M, but detector was only
        # prepared up to ...".  Wait for the reset to actually propagate.
        await wait_for_value(self.writer.num_captured, 0, timeout=10)
        return provider

    async def stop(self) -> None:
        # Closing the HDF5 file on network storage typically takes longer
        # than the default 10 s timeout.
        await stop_busy_record(self.writer.capture, timeout=60)


class HEXKinetixDetector(KinetixDetector):
    """
    HEX-specific Kinetix detector.

    After every scan (or abort) the camera is reset to Continuous / Internal
    mode and restarted so that the live-view image stream resumes — and,
    symmetrically, staging STOPS the live view first (the old scripts'
    ``stop_preview()``): a free-running camera leaks frames into the new
    HDF capture between prepare and kickoff, which shows up as
    "Kickoff requested N:M, but detector was only prepared up to ...".
    """

    @AsyncStatus.wrap
    async def stage(self) -> None:
        await stop_busy_record(self.driver.acquire, timeout=10)
        await super().stage()

    async def start_live_view(self) -> None:
        """Start Continuous/Internal free-run WITHOUT waiting for completion.

        Acquire is a busy record and continuous mode never "finishes" — a
        plain set()/abs_set leaves a completion-tracked status that times
        out.  Uses the ADAcquireLogic non-blocking idiom; plans call this
        via ``bps.wait_for([detector.start_live_view])`` — wait_for takes
        awaitable factories (callables returning a coroutine), so the bound
        method is passed UNCALLED and the RunEngine creates/awaits the
        coroutine in its own loop.
        """
        await self.driver.trigger_mode.set(KinetixTriggerMode.INTERNAL)
        await self.driver.image_mode.set(ADImageMode.CONTINUOUS)
        self._live_view_status = await set_and_wait_for_other_value(
            set_signal=self.driver.acquire,
            set_value=True,
            match_signal=self.driver.acquire,
            match_value=True,
            wait_for_set_completion=False,
        )

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        # Close the writer and disarm via the standard path first.
        await super().unstage()

        # Return to continuous free-run mode for live viewing.
        await self.start_live_view()


def make_kinetix(
    detector_id: int, path_provider: SettablePathProvider | None = None
) -> HEXKinetixDetector:
    """
    Instantiate a HEXKinetixDetector for the given detector number (1 or 3).

    Parameters
    ----------
    detector_id:
        Which Kinetix to connect to (1 or 3).
    path_provider:
        Decides where HDF files are written.  Defaults to a fresh
        ``SettablePathProvider`` so plans can retarget it per scan; it is also
        exposed as ``detector.path_provider``.
    """
    if detector_id not in (1, 3):
        raise ValueError(f"Kinetix detector id must be 1 or 3, got {detector_id}")

    if path_provider is None:
        path_provider = SettablePathProvider()

    prefix = f"XF:27ID1-BI{{Kinetix-Det:{detector_id}}}"
    detector = HEXKinetixDetector(
        prefix,
        ADWriterFactory(
            writer_cls=NDFileHDF5IO,
            writer_suffix="HDF1:",
            writer_name="hdf",
            datakey_suffix="",
            array_description=None,
            data_logic_factory=lambda writer, desc, driver, plugins: HEXADHDFDataLogic(
                array_description=desc,
                path_provider=path_provider,
                driver=driver,
                writer=writer,
                plugins=list(plugins),
                datakey_suffix="",
            ),
        ),
        plugins={
            "proc": HEXProcPluginIO(prefix + "Proc1:"),
            "pva": NDPluginBaseIO(prefix + "Pva1:"),
        },
        name=f"kinetix{detector_id}",
    )
    # The legacy default upstream for the file/view plugins: Det:1 routes
    # through the transform plugin, Det:3 straight off the camera.
    detector.default_source_port = "TRANS1" if detector_id == 1 else "KTX1"
    # Not an ophyd-async child device — just a handle so plans can retarget
    # the output directory (see SettablePathProvider above).
    detector.path_provider = path_provider
    return detector
