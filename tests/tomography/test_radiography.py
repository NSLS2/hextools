from collections.abc import Callable
from typing import Any

import pytest
from bluesky import Msg, RunEngine
from ophyd_async.core import (
    Device,
    StaticPathProvider,
    callback_on_mock_put,
    get_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.adcore import NDPluginFileIO, NDProcessFilterType, NDProcessIO
from ophyd_async.epics.adkinetix import KinetixDetector, KinetixTriggerMode
from ophyd_async.epics.core import epics_signal_rw
from ophyd_async.fastcs.panda import DatasetTable, HDFPanda, PandaHdf5DatasetType

from hextools.photon_delivery_system import Shutter, ShutterStatus
from hextools.tomography.radiography import FRAME_PERIOD_MARGIN, take_radiograph


@pytest.mark.usefixtures("skip_long_sleeps")
async def test_take_radiograph_single_row(
    RE: RunEngine,
    kinetix_det_factory: Callable[[int], KinetixDetector],
    two_shutters: tuple[Shutter, Shutter],
):
    exposure_time, num_images, num_acquisitions, wait = 0.1, 10, 5, 0.01

    fe_shutter, photon_shutter = two_shutters
    set_mock_value(fe_shutter.status, ShutterStatus.OPEN)
    ktx = kinetix_det_factory(1)

    docs: dict[str, list[dict[str, Any]]] = {}

    def cache_docs(name: str, doc: dict[str, Any]):
        docs.setdefault(name, []).append(doc)

    messages_by_type: dict[str, list[Msg]] = {}

    def msg_hook(msg: Msg):
        messages_by_type.setdefault(msg.command, []).append(msg)

    RE.msg_hook = msg_hook  # type: ignore

    RE(
        take_radiograph(
            [ktx],
            exposure_time,
            num_images=num_images,
            num_iterations=num_acquisitions,
            external_trigger=False,
            time_gap=wait,
            use_shutter=True,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
        ),
        cache_docs,  # type: ignore
    )

    # Each acquisition is its own run, so each opens its own file
    for kind in ("start", "descriptor", "stream_resource", "stop", "event"):
        assert len(docs[kind]) == num_acquisitions
    assert len(docs["stream_datum"]) == num_acquisitions
    assert all(start["plan_name"] == "take_radiograph" for start in docs["start"])

    # the gap between runs; longer sleeps are shutter settling
    delay_sleeps = [m for m in messages_by_type.get("sleep", []) if m.args[0] == wait]
    assert len(delay_sleeps) == num_acquisitions - 1

    assert await ktx.driver.acquire_time.get_value() == exposure_time
    assert await ktx.driver.num_images.get_value() == num_images
    assert (
        await photon_shutter.status.get_value() == ShutterStatus.CLOSED
    )  # finalizer closed it

    assert await ktx.driver.acquire_period.get_value() == pytest.approx(
        exposure_time + FRAME_PERIOD_MARGIN
    )


# --- PandA-paced radiograph: HEX_DEFAULT wiring --------------------------------
# BITS.B -> LUT1 -> PULSE1 (enabled while PCAP is armed) -> TTLOUT1 (camera) and PCAP


class _Bits(Device):
    """The PandA BITS block; on the real PandA it is discovered from the IOC."""

    def __init__(self, prefix: str, name: str = ""):
        self.a = epics_signal_rw(int, f"{prefix}A")
        self.b = epics_signal_rw(int, f"{prefix}B")
        super().__init__(name=name)


class _PandaWithBits(HDFPanda):
    def __init__(self, prefix: str, path_provider, name: str = ""):
        self.bits = _Bits(f"{prefix}BITS:")
        super().__init__(prefix, path_provider, name=name)


@pytest.fixture
def panda_pacing(static_path_provider: StaticPathProvider):
    """A mock PandA whose BITS.B rising edge sends PULSE1's train to the cameras."""

    def _factory(cameras: list[KinetixDetector]) -> _PandaWithBits:
        with init_devices(mock=True):
            panda = _PandaWithBits("PANDA:", static_path_provider, name="panda1")
        set_mock_value(panda.data.directory_exists, True)
        set_mock_value(
            panda.data.datasets,
            DatasetTable(name=["Angle"], dtype=[PandaHdf5DatasetType.FLOAT_64]),
        )
        callback_on_mock_put(
            panda.pcap.arm, lambda v: set_mock_value(panda.pcap.active, v)
        )
        callback_on_mock_put(
            panda.data.capture, lambda _: set_mock_value(panda.data.num_captured, 0)
        )

        async def _pulse_train(value):
            if not value or not await panda.pcap.active.get_value():
                return
            n = await panda.pulse[1].pulses.get_value()
            for cam in cameras:
                hdf = cam.get_plugin("hdf", NDPluginFileIO)
                set_mock_value(hdf.num_captured, await hdf.num_captured.get_value() + n)
            captured = await panda.data.num_captured.get_value()
            set_mock_value(panda.data.num_captured, captured + n)

        callback_on_mock_put(panda.bits.b, _pulse_train)
        return panda

    return _factory


async def test_external_trigger_radiograph_is_paced_by_the_panda(
    RE: RunEngine,
    kinetix_det_factory: Callable[[int], KinetixDetector],
    panda_pacing,
    two_shutters: tuple[Shutter, Shutter],
):
    exposure_time, acquire_period, num_images, num_iterations = 0.1, 0.2, 19, 2
    fe_shutter, photon_shutter = two_shutters
    ktx = kinetix_det_factory(1)
    panda = panda_pacing([ktx])

    docs: dict[str, list[dict[str, Any]]] = {}

    def cache_docs(name: str, doc: dict[str, Any]):
        docs.setdefault(name, []).append(doc)

    RE(
        take_radiograph(
            [ktx],
            exposure_time,
            num_images=num_images,
            num_iterations=num_iterations,
            acquire_period=acquire_period,
            external_trigger=True,
            use_shutter=False,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
            panda=panda,
        ),
        cache_docs,  # type: ignore
    )

    # Each burst is its own run. The PandA paces the camera but is not recorded,
    # so no Angle (dropped in ecd0b0d; may be revisited).
    assert len(docs["start"]) == num_iterations
    assert all(stop["exit_status"] == "success" for stop in docs["stop"])
    assert {d["data_key"] for d in docs["stream_resource"]} == {"kinetix1"}
    assert len(docs["stream_datum"]) == num_iterations
    # armed for external edges during the bursts; cleanup returns it to preview
    modes = [
        call.args[0] for call in get_mock_put(ktx.driver.trigger_mode).call_args_list
    ]
    assert any(mode != KinetixTriggerMode.INTERNAL for mode in modes)
    assert await ktx.driver.trigger_mode.get_value() == KinetixTriggerMode.INTERNAL

    pulse = panda.pulse[1]
    assert await pulse.pulses.get_value() == num_images
    assert await pulse.step.get_value() == pytest.approx(acquire_period)
    assert await pulse.width.get_value() == pytest.approx(exposure_time)
    # BITS.A (fly-scan PCOMP enable) stays off; BITS.B rises once per burst and ends low
    assert await panda.bits.a.get_value() == 0
    bits_b = [call.args[0] for call in get_mock_put(panda.bits.b).call_args_list]
    assert bits_b.count(1) == num_iterations
    assert await panda.bits.b.get_value() == 0


def test_external_trigger_without_a_panda_fails_before_arming(
    RE: RunEngine,
    kinetix_det_factory: Callable[[int], KinetixDetector],
    two_shutters: tuple[Shutter, Shutter],
):
    """With nothing pulsing the camera, an external radiograph used to time out."""
    fe_shutter, photon_shutter = two_shutters
    ktx = kinetix_det_factory(1)
    with pytest.raises(ValueError, match="panda"):
        RE(
            take_radiograph(
                [ktx],
                0.1,
                num_images=19,
                external_trigger=True,
                use_shutter=False,
                fe_shutter=fe_shutter,
                photon_shutter=photon_shutter,
            )
        )
    assert not get_mock_put(ktx.driver.acquire).called


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, NDProcessFilterType.AVERAGE),
        ({"filter_type": "Sum"}, NDProcessFilterType.SUM),
    ],
)
async def test_averaged_radiograph_sets_the_proc_filter_type(
    RE: RunEngine,
    kinetix_det_factory: Callable[..., KinetixDetector],
    two_shutters: tuple[Shutter, Shutter],
    kwargs: dict[str, Any],
    expected: NDProcessFilterType,
):
    fe_shutter, photon_shutter = two_shutters
    ktx = kinetix_det_factory(1, with_proc=True)
    proc = ktx.get_plugin("proc", NDProcessIO)
    set_mock_value(proc.filter_type, NDProcessFilterType.DIFFERENCE)
    docs: dict[str, list[dict[str, Any]]] = {}

    def cache_docs(name: str, doc: dict[str, Any]):
        docs.setdefault(name, []).append(doc)

    RE(
        take_radiograph(
            [ktx],
            0.01,
            num_images=2,
            images_to_average=3,
            external_trigger=False,
            use_shutter=False,
            fe_shutter=fe_shutter,
            photon_shutter=photon_shutter,
            **kwargs,
        ),
        cache_docs,  # type: ignore
    )

    assert await proc.filter_type.get_value() == expected
    assert docs["start"][0]["filter_type"] == expected.value
