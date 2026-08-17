import asyncio
from pathlib import Path
from pprint import pprint
from typing import Any, Callable

from bluesky import RunEngine
import numpy as np
from ophyd_async.core import StaticPathProvider, UUIDFilenameProvider, callback_on_mock_put, init_devices, callback_on_mock_execute, set_mock_value
import pytest
from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.epics.adcore import ADWriterFactory, NDFileIO, NDPluginFileIO

from hextools.motors import RotationMotor
from hextools.photon_delivery_system import Shutter
from hextools.tomography.alignment import identify_sign_tilt_angle, tomo_alignment_scan
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


# @pytest.fixture
# def kinetix_detector_factory()
#     def _factory(num: int):
#         with init_devices(mock=True):


@pytest.fixture
def shutter_factory() -> Callable[[str], Shutter]:
    def _factory(name: str) -> Shutter:
        with init_devices(mock=True):
            shutter = Shutter(name, name=name)
        callback_on_mock_execute(shutter.open_cmd, lambda: set_mock_value(shutter.status, True))
        callback_on_mock_execute(shutter.close_cmd, lambda: set_mock_value(shutter.status, False))
        return shutter
    return _factory

@pytest.fixture
def two_shutters(shutter_factory: Callable[[str], Shutter]) -> tuple[Shutter, Shutter]:
    fe_shutter = shutter_factory("front_end_shutter")
    photon_shutter = shutter_factory("photon_shutter")
    return fe_shutter, photon_shutter

@pytest.fixture
def rotation_motor() -> RotationMotor:
    with init_devices(mock=True):
        rotation_motor = RotationMotor("ROT")
    return rotation_motor

@pytest.fixture
def static_path_provider(tmp_path: Path) -> StaticPathProvider:
    return StaticPathProvider(UUIDFilenameProvider(), tmp_path)

@pytest.fixture
def kinetix_det_factory(static_path_provider: StaticPathProvider) -> Callable[[int], KinetixDetector]:
    def _factory(num: int) -> KinetixDetector:
        with init_devices(mock=True):

            ktx = KinetixDetector(f"KTX{num}", ADWriterFactory.tiff(static_path_provider), name=f"kinetix{num}")
            tiff_plugin = ktx.get_plugin("tiff", NDPluginFileIO)

        async def _mock_increment_num_captured(_):
            current_num_cap = await tiff_plugin.num_captured.get_value()
            set_mock_value(tiff_plugin.num_captured, current_num_cap + 1)

        callback_on_mock_put(tiff_plugin.capture, lambda _: set_mock_value(tiff_plugin.num_captured, 0))
        callback_on_mock_put(ktx.driver.acquire, _mock_increment_num_captured)
        callback_on_mock_put(tiff_plugin.file_path, lambda _: set_mock_value(tiff_plugin.file_path_exists, True))
        return ktx
    return _factory

async def test_tomo_alignment_scan_fails_if_fe_shutter_closed(RE: RunEngine, two_shutters: tuple[Shutter, Shutter], rotation_motor: RotationMotor):

    fe_shutter, photon_shutter = two_shutters

    assert not any(await asyncio.gather(fe_shutter.status.get_value(), photon_shutter.status.get_value()))

    with pytest.raises(ValueError, match="Front-end shutter is closed"):
        RE(tomo_alignment_scan([], rotation_motor, fe_shutter, photon_shutter, 0.1))

async def test_tomo_alignment_scan(RE: RunEngine, kinetix_det_factory: Callable[[int], KinetixDetector], two_shutters: tuple[Shutter, Shutter], rotation_motor: RotationMotor):
    fe_shutter, photon_shutter = two_shutters
    ktx1 = kinetix_det_factory(1)
    RE(bps.mv(fe_shutter, True))

    max_velocity = await rotation_motor.max_velocity.get_value()
    assert not await photon_shutter.status.get_value()

    docs: dict[str, list[dict[str, Any]]] = {}
    def cache_docs(name: str, doc: dict[str, Any]):
        if name in docs:
            docs[name].append(doc)
        else:
            docs[name] = [doc]

    RE(tomo_alignment_scan([ktx1], rotation_motor, fe_shutter, photon_shutter, 0.1), cache_docs)  # type:ignore

    assert await rotation_motor.velocity.get_value() == max_velocity
    assert await photon_shutter.status.get_value()

    pprint(docs)
    assert False