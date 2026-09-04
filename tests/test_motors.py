import bluesky.plan_stubs as bps
import pytest
from bluesky.run_engine import RunEngine
from bluesky.utils import FailedStatus
from ophyd_async.core import init_devices, set_mock_value

from hextools.motors import CameraObjective, DoubleObjCamera, HomeStatus


@pytest.fixture
def double_obj_camera() -> DoubleObjCamera:
    with init_devices(mock=True):
        camera = DoubleObjCamera("TEST:CAM:")
    return camera


async def test_double_obj_camera_raises_when_not_homed(
    RE: RunEngine, double_obj_camera: DoubleObjCamera
):
    set_mock_value(double_obj_camera._obj_selector_home_sts, HomeStatus.NOT_HOMED)

    with pytest.raises(FailedStatus) as exc_info:
        RE(bps.mv(double_obj_camera, CameraObjective.LEFT_4MM))
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
    double_obj_camera: DoubleObjCamera,
    objective: CameraObjective,
    readback_attr: str,
    other_readback_attr: str,
):
    set_mock_value(double_obj_camera._obj_selector_home_sts, HomeStatus.HOMED)
    readback = getattr(double_obj_camera, readback_attr)
    set_mock_value(readback, True)

    RE(bps.mv(double_obj_camera, objective))

    assert await readback.get_value() is True
    assert await getattr(double_obj_camera, other_readback_attr).get_value() is False
