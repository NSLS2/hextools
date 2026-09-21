import bluesky.plan_stubs as bps
import pytest
from bluesky.run_engine import RunEngine
from bluesky.utils import FailedStatus
from ophyd_async.core import init_devices, set_mock_value

from hextools.motors import CameraObjective, FOV_20_40_mm_Camera, FOV_2_4_mm_Camera, HomeStatus


@pytest.fixture
def fov_2_4_mm_camera() -> FOV_2_4_mm_Camera:
    with init_devices(mock=True):
        camera = FOV_2_4_mm_Camera("TEST:CAM:")
    return camera


async def test_fov_2_4_mm_camera_raises_when_not_homed(
    RE: RunEngine, fov_2_4_mm_camera: FOV_2_4_mm_Camera
):
    set_mock_value(fov_2_4_mm_camera._obj_selector_home_sts, HomeStatus.NOT_HOMED)

    with pytest.raises(FailedStatus) as exc_info:
        RE(bps.mv(fov_2_4_mm_camera, CameraObjective.LEFT_4MM))
    assert "not homed" in str(exc_info.value.__cause__)


@pytest.mark.parametrize(
    "objective, readback_attr, other_readback_attr",
    [
        (CameraObjective.LEFT_4MM, "_at_left_objective", "_at_right_objective"),
        (CameraObjective.RIGHT_2MM, "_at_right_objective", "_at_left_objective"),
    ],
)
async def test_double_obj_camera_moves_to_objective(
    RE: RunEngine,
    fov_2_4_mm_camera: FOV_2_4_mm_Camera,
    objective: CameraObjective,
    readback_attr: str,
    other_readback_attr: str,
):
    set_mock_value(fov_2_4_mm_camera._obj_selector_home_sts, HomeStatus.HOMED)
    readback = getattr(fov_2_4_mm_camera, readback_attr)
    set_mock_value(readback, True)

    RE(bps.mv(fov_2_4_mm_camera, objective))

    assert await readback.get_value() is True
    assert await getattr(fov_2_4_mm_camera, other_readback_attr).get_value() is False
