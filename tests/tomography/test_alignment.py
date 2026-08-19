import asyncio
import inspect
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, OrderedDict

from bluesky import Msg, RunEngine
from bluesky.run_engine import RunEngineResult
import numpy as np
from ophyd_async.core import (
    StaticPathProvider,
    UUIDFilenameProvider,
    callback_on_mock_put,
    init_devices,
    callback_on_mock_execute,
    set_mock_value,
)
import pytest
from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.epics.adcore import (
    ADBaseDataType,
    ADWriterFactory,
    NDFileIO,
    NDPluginFileIO,
)
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor
from pytest_mock import MockerFixture

from hextools.motors import RotationMotor

from hextools.photon_delivery_system import Shutter
from hextools.tomography.alignment import ensure_run_is_valid, identify_sign_tilt_angle, tomo_alignment_scan
from bluesky import plans as bp, plan_stubs as bps


@pytest.mark.parametrize(
    "x, y, expected_sign",
    [
        # Horizontal arc curving below -> positive sign
        (
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([0.0, -1.0, -2.0, -1.0, 0.0]),
            1,
        ),
        # Horizontal arc curving above -> negative sign
        (
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([0.0, 1.0, 2.0, 1.0, 0.0]),
            -1,
        ),
        # Tilted line with points sagging below -> positive sign
        (
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([0.0, 0.0, 0.0, 1.5, 2.0]),
            1,
        ),
        # Tilted line with points bulging above -> negative sign
        (
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([0.0, 1.0, 2.0, 2.5, 2.0]),
            -1,
        ),
        # Upper semicircle -> negative sign
        (
            np.cos(np.linspace(0, np.pi, 7)),
            np.sin(np.linspace(0, np.pi, 7)),
            -1,
        ),
        # Lower semicircle -> positive sign
        (
            np.cos(np.linspace(0, np.pi, 7)),
            -np.sin(np.linspace(0, np.pi, 7)),
            1,
        ),
    ],
)
def test_identify_sign_tilt_angle(x, y, expected_sign):
    assert identify_sign_tilt_angle(x, y) == expected_sign


@pytest.mark.parametrize(
    ("available_streams", "det_names", "motor_name", "proj_stream", "ff_stream", "expected_valid", "expected_msg"),
    [
        ({}, ["det"], "motor", "primary", None, False, "Stream 'primary' not found in the run"),
        ({"primary": ["motor"]}, ["det"], "motor", "primary", None, False, "Detector 'det' not found in the stream"),
        ({"primary": ["det"]}, ["det"], "motor", "primary", None, False, "Motor 'motor' not found in the stream"),
        ({"primary": ["det", "motor"]}, ["det"], "motor", "primary", "ff", False, "Stream 'ff' not found in the run"),
        ({"primary": ["det", "motor"]}, ["det"], "motor", "primary", None, True, "Detector 'det' not found in the stream"),
        ({"primary": ["det", "motor"], "ff": ["det"]}, ["det"], "motor", "primary", "ff", True, "Detector 'det' not found in the stream"),
    ]
)
def test_ensure_run_is_valid(mocker: MockerFixture, available_streams: dict[str, list[str]], det_names: list[str], motor_name: str, proj_stream: str, ff_stream: str, expected_valid: bool, expected_msg: str):

    mock_run = mocker.MagicMock()
    mock_run.__getitem__.side_effect = available_streams.__getitem__
    mock_run.get.side_effect = available_streams.get
    mock_run.__contains__.side_effect = available_streams.__contains__
    mock_run.keys.side_effect = available_streams.keys

    if not expected_valid:
        with pytest.raises(KeyError, match=expected_msg):
            ensure_run_is_valid(mock_run, det_names, motor_name, proj_stream=proj_stream, ff_stream=ff_stream)
    else:
        ensure_run_is_valid(mock_run, det_names, motor_name, proj_stream=proj_stream, ff_stream=ff_stream)
# @pytest.mark.parametrize(
#     ("top", "bottom", "left", "right"),
#     [
#         (0, 0, 0, -1),
#         (-1, 0, -2, 0),
#         (-100, 20, 40, 100)
#     ]
# )
# def test_check_alignment_fails_with_invalid_crop(top, bottom, left, right):

@pytest.fixture
def shutter_factory() -> Callable[[str], Shutter]:
    def _factory(name: str) -> Shutter:
        with init_devices(mock=True):
            shutter = Shutter(name, name=name)
        callback_on_mock_execute(
            shutter.open_cmd, lambda: set_mock_value(shutter.status, True)
        )
        callback_on_mock_execute(
            shutter.close_cmd, lambda: set_mock_value(shutter.status, False)
        )
        return shutter

    return _factory


@pytest.fixture
def two_shutters(shutter_factory: Callable[[str], Shutter]) -> tuple[Shutter, Shutter]:
    fe_shutter = shutter_factory("front_end_shutter")
    photon_shutter = shutter_factory("photon_shutter")
    return fe_shutter, photon_shutter


@pytest.fixture
def motors() -> tuple[RotationMotor, AsyncEpicsMotor]:
    with init_devices(mock=True):
        rotation_motor = RotationMotor("ROT")
        sample_stage_x = AsyncEpicsMotor("X")
    return rotation_motor, sample_stage_x


@pytest.fixture
def static_path_provider(tmp_path: Path) -> StaticPathProvider:
    return StaticPathProvider(UUIDFilenameProvider(), tmp_path)


@pytest.fixture
def kinetix_det_factory(
    static_path_provider: StaticPathProvider,
) -> Callable[[int], KinetixDetector]:
    def _factory(num: int) -> KinetixDetector:
        with init_devices(mock=True):
            ktx = KinetixDetector(
                f"KTX{num}",
                ADWriterFactory.tiff(static_path_provider),
                name=f"kinetix{num}",
            )
            tiff_plugin = ktx.get_plugin("tiff", NDPluginFileIO)

        set_mock_value(ktx.driver.array_size_x, 3200)
        set_mock_value(ktx.driver.array_size_y, 3200)
        set_mock_value(ktx.driver.data_type, ADBaseDataType.UINT16)

        async def _mock_increment_num_captured(_):
            current_num_cap = await tiff_plugin.num_captured.get_value()
            set_mock_value(tiff_plugin.num_captured, current_num_cap + 1)

        callback_on_mock_put(
            tiff_plugin.capture, lambda _: set_mock_value(tiff_plugin.num_captured, 0)
        )
        callback_on_mock_put(ktx.driver.acquire, _mock_increment_num_captured)
        callback_on_mock_put(
            tiff_plugin.file_path,
            lambda _: set_mock_value(tiff_plugin.file_path_exists, True),
        )
        return ktx

    return _factory


async def test_tomo_alignment_scan_fails_if_fe_shutter_closed(
    RE: RunEngine, two_shutters: tuple[Shutter, Shutter], motors: tuple[RotationMotor, AsyncEpicsMotor]
):

    fe_shutter, photon_shutter = two_shutters
    rotation_motor, _ = motors
    assert not any(
        await asyncio.gather(
            fe_shutter.status.get_value(), photon_shutter.status.get_value()
        )
    )

    with pytest.raises(ValueError, match="Front-end shutter is closed"):
        RE(tomo_alignment_scan([], rotation_motor, fe_shutter, photon_shutter, 0.1))


@pytest.mark.parametrize(
    (
        "exposure_time",
        "num_projections",
        "init_angle",
        "stop_angle",
        "base_x_offset",
        "include_sample_stage_x",
    ),
    [
        (0.1, 37, 0.0, 360.0, 0.0, True),
        (0.4, 11, 0.0, 360.0, 10.0, False),
        (1.0, 21, -50.0, 50.0, 10.0, True),
    ],
)
async def test_tomo_alignment_scan(
    RE: RunEngine,
    kinetix_det_factory: Callable[[int], KinetixDetector],
    two_shutters: tuple[Shutter, Shutter],
    motors: tuple[RotationMotor, AsyncEpicsMotor],
    exposure_time: float,
    num_projections: int,
    init_angle: float,
    stop_angle: float,
    base_x_offset: float,
    include_sample_stage_x: bool,
):
    rotation_motor, sample_stage_x = motors
    fe_shutter, photon_shutter = two_shutters
    ktx1 = kinetix_det_factory(1)
    RE(bps.mv(fe_shutter, True))

    set_mock_value(rotation_motor.max_velocity, 10000)
    max_velocity = await rotation_motor.max_velocity.get_value()
    assert not await photon_shutter.status.get_value()

    docs: dict[str, list[dict[str, Any]]] = {}

    def cache_docs(name: str, doc: dict[str, Any]):
        if name in docs:
            docs[name].append(doc)
        else:
            docs[name] = [doc]

    messages_by_type: dict[str, list[Msg]] = {}
    messages: list[Msg] = []

    def msg_hook(msg: Msg):
        messages.append(msg)
        if msg.command in messages_by_type:
            messages_by_type[msg.command].append(msg)
        else:
            messages_by_type[msg.command] = [msg]

    RE.msg_hook = msg_hook  # type:ignore

    runs: RunEngineResult = RE(
        tomo_alignment_scan(
            [ktx1],
            rotation_motor,
            fe_shutter,
            photon_shutter,
            exposure_time,
            num_projections=num_projections,
            init_angle=init_angle,
            stop_angle=stop_angle,
            base_x_offset=base_x_offset,
            sample_stage_x=None if not include_sample_stage_x else sample_stage_x,
        ),
        cache_docs,  # type:ignore
    )

    expecting_flat_run = base_x_offset > 0.0 and include_sample_stage_x

    assert await rotation_motor.velocity.get_value() == max_velocity
    assert await photon_shutter.status.get_value()

    for doc_type in ["start", "descriptor", "stream_resource", "stop"]:
        assert (
            len(docs[doc_type]) == 2
            if expecting_flat_run
            else 1
        )

    for doc_type in ["stream_datum", "event"]:
        expected_num_events = num_projections + 1 if expecting_flat_run else num_projections
        assert len(docs[doc_type]) == expected_num_events

    assert await ktx1.driver.acquire_time.get_value() == exposure_time
    assert await rotation_motor.user_readback.get_value() == stop_angle

    if not expecting_flat_run:
        assert len(runs.run_start_uids) == 1
    else:
        assert len(runs.run_start_uids) == 2
        assert docs["start"][1]["flat_uid"] == runs.run_start_uids[0]

        sample_staged_move_counter = 0
        for msg in messages_by_type["set"]:
            if msg.obj == sample_stage_x:
                if sample_staged_move_counter == 0:
                    assert msg.args == np.float64(base_x_offset)
                else:
                    assert msg.args == np.float64(0.0)
                sample_staged_move_counter += 1
        assert sample_staged_move_counter == 2
        assert await sample_stage_x.user_readback.get_value() == 0.0

    for i, msg in enumerate(messages):
        if msg.command == "open_run" and msg.kwargs["plan_name"] == "scan":
            send_motor_to_init_msg = messages[i + 2]
            assert send_motor_to_init_msg.command == "set"
            assert send_motor_to_init_msg.obj == rotation_motor
            assert send_motor_to_init_msg.args == np.float64(init_angle)

