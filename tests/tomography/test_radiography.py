from collections.abc import Callable
from pathlib import Path
from typing import Any

import bluesky.plan_stubs
import pytest
from bluesky import Msg, RunEngine
from bluesky import plan_stubs as bps
from ophyd_async.core import (
    StaticPathProvider,
    UUIDFilenameProvider,
    callback_on_mock_execute,
    callback_on_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.adcore import ADBaseDataType, ADWriterFactory, NDPluginFileIO
from ophyd_async.epics.adkinetix import KinetixDetector

from hextools.photon_delivery_system import Shutter
from hextools.photon_delivery_system.shutter import ShutterStatus
from hextools.tomography.radiography import FRAME_PERIOD_MARGIN, take_radiograph


class _FrozenClock:
    """Stand-in for the ``time`` module whose clock never advances.

    ``bps.repeat`` (which ``bp.count`` builds on) emits a sleep only when
    ``delay - elapsed`` is still positive. With a real clock, whether the sleep
    appears at all is a race against how fast the machine ran the acquisition -
    which is why asserting an exact sleep count used to fail on CI, on a
    different pair of Python versions each run. Freezing elapsed time at zero
    makes the full delay survive every time. The only two uses of ``time`` in
    ``bluesky.plan_stubs`` are the pair inside that delay calculation, so
    nothing else is affected.
    """

    @staticmethod
    def time() -> float:
        return 0.0


# --- shutters: same shape as tests/tomography/test_alignment.py ---------------


@pytest.fixture
def shutter_factory() -> Callable[[str], Shutter]:
    def _factory(name: str) -> Shutter:
        with init_devices(mock=True):
            shutter = Shutter(name, name=name)
        # the only two arcs Shutter.set awaits: a command put flips the status readback
        callback_on_mock_execute(
            shutter.open_cmd,
            lambda *_: set_mock_value(shutter.status, ShutterStatus.OPEN),
        )
        callback_on_mock_execute(
            shutter.close_cmd,
            lambda *_: set_mock_value(shutter.status, ShutterStatus.CLOSED),
        )
        # A mock enum signal starts at its first member, which is OPEN
        set_mock_value(shutter.status, ShutterStatus.CLOSED)
        return shutter

    return _factory


@pytest.fixture
def two_shutters(shutter_factory: Callable[[str], Shutter]) -> tuple[Shutter, Shutter]:
    return shutter_factory("front_end_shutter"), shutter_factory("photon_shutter")


# --- detector: Kinetix with an HDF writer, minimum causal chain ---------------


@pytest.fixture
def static_path_provider(tmp_path: Path) -> StaticPathProvider:
    return StaticPathProvider(UUIDFilenameProvider(), tmp_path)


@pytest.fixture
def kinetix_hdf_factory(
    static_path_provider: StaticPathProvider,
) -> Callable[[int], KinetixDetector]:
    def _factory(num: int) -> KinetixDetector:
        with init_devices(mock=True):
            ktx = KinetixDetector(
                f"KTX{num}",
                ADWriterFactory.hdf(static_path_provider),
                name=f"kinetix{num}",
            )
            hdf = ktx.get_plugin_by_name("hdf", NDPluginFileIO)

        # what the descriptor needs to describe the image
        set_mock_value(ktx.driver.array_size_x, 3200)
        set_mock_value(ktx.driver.array_size_y, 3200)
        set_mock_value(ktx.driver.data_type, ADBaseDataType.UINT16)

        async def _one_burst_arrives(_):
            # one acquire produces a whole burst: num_images frames land in the file
            n = await ktx.driver.num_images.get_value()
            got = await hdf.num_captured.get_value()
            set_mock_value(hdf.num_captured, got + n)

        # prepare: setting the directory must make the IOC report it exists
        callback_on_mock_put(
            hdf.file_path, lambda _: set_mock_value(hdf.file_path_exists, True)
        )
        # prepare: starting capture resets the frame counter
        callback_on_mock_put(hdf.capture, lambda _: set_mock_value(hdf.num_captured, 0))
        # trigger: the writer waits on num_captured reaching the expected count
        callback_on_mock_put(ktx.driver.acquire, _one_burst_arrives)
        return ktx

    return _factory


# --- one row of the happy path, to prove the arcs before the table exists -----


async def test_take_radiograph_single_row(
    RE: RunEngine,
    kinetix_hdf_factory: Callable[[int], KinetixDetector],
    two_shutters: tuple[Shutter, Shutter],
    monkeypatch: pytest.MonkeyPatch,
):
    # the profile sets this; tests do not load the profile
    monkeypatch.setenv("OPHYD_ASYNC_PRESERVE_DETECTOR_STATE", "YES")
    # Make the inter-acquisition delay deterministic rather than a race against
    # runner speed - see _FrozenClock. Without this the sleep assertions below
    # pass or fail depending on how loaded the machine is.
    monkeypatch.setattr(bluesky.plan_stubs, "time", _FrozenClock)
    exposure_time, num_images, num_acquisitions, wait = 0.1, 10, 5, 0.01

    fe_shutter, photon_shutter = two_shutters
    ktx = kinetix_hdf_factory(1)
    RE(bps.mv(fe_shutter, True))  # precondition: front end already open

    docs: dict[str, list[dict[str, Any]]] = {}

    def cache_docs(name: str, doc: dict[str, Any]):
        docs.setdefault(name, []).append(doc)

    messages_by_type: dict[str, list[Msg]] = {}

    def msg_hook(msg: Msg):
        messages_by_type.setdefault(msg.command, []).append(msg)

    RE.msg_hook = msg_hook

    RE(
        take_radiograph(
            [ktx],
            exposure_time,
            num_images=num_images,
            num_acquisitions=num_acquisitions,
            time_gap=wait,
            use_shutter=True,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
        ),
        cache_docs,  # type: ignore
    )

    for kind in ("start", "descriptor", "stream_resource", "stop"):
        assert len(docs[kind]) == 1
    assert len(docs["event"]) == num_acquisitions
    assert len(docs["stream_datum"]) == num_acquisitions

    start = docs["start"][0]
    assert start["plan_name"] == "take_radiograph"

    sleeps = messages_by_type.get("sleep", [])
    # A gap between each pair of acquisitions and none after the last: the plan
    # hands bp.count a finite list of num_acquisitions - 1 delays, so bps.repeat
    # stops without the trailing sleep a scalar delay would add.
    assert len(sleeps) == num_acquisitions - 1
    # Exact, not "<= wait": with the clock frozen the whole gap survives. If
    # bluesky ever measures elapsed time some other way, this fails loudly
    # instead of quietly going back to being a race.
    assert all(m.args[0] == wait for m in sleeps)

    assert await ktx.driver.acquire_time.get_value() == exposure_time
    assert await ktx.driver.num_images.get_value() == num_images
    # finalizer closed it
    assert await photon_shutter.status.get_value() == ShutterStatus.CLOSED

    assert await ktx.driver.acquire_period.get_value() == pytest.approx(
        exposure_time + FRAME_PERIOD_MARGIN
    )
