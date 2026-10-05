import bluesky.plan_stubs as bps
import pytest
from bluesky.run_engine import RunEngine
from bluesky.utils import FailedStatus
from ophyd_async.core import init_devices, set_mock_value

from hextools.motors import (
    CameraObjective,
    FOV_2_4_mm_Camera,
    HomeStatus,
    OpticsTable,
    RotationMotor,
    SampleTower,
)


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


@pytest.fixture
def sample_tower(RE) -> SampleTower:
    with init_devices(mock=True):
        tower = SampleTower("TEST:TOWER:")
    return tower


@pytest.fixture
def optics_table(RE) -> OpticsTable:
    with init_devices(mock=True):
        table = OpticsTable("TEST:OPT:")
    return table


async def test_sample_tower_reads_virtual_axes_not_the_real_motors(
    sample_tower: SampleTower,
):
    """y, pitch and roll are what a scan records; the real motors are not.

    Seven real motors combine to give those three, and putting all ten in
    every event document would bury the ones that mean something. The real
    motors stay reachable as attributes for alignment work.
    """
    reading = await sample_tower.read()
    axes = {key.rsplit("-", 1)[-1] for key in reading}
    assert axes == {"y", "pitch", "roll"}

    real_motors = ("x1", "x2", "z1", "z2", "inboard_y", "outboard_y", "downstream_y")
    for real_motor in real_motors:
        assert hasattr(sample_tower, real_motor)


async def test_optics_table_reads_all_ten_axes(optics_table: OpticsTable):
    reading = await optics_table.read()
    axes = {key.rsplit("-", 1)[-1] for key in reading}
    assert axes == {"x2", "y2", "rx3", "ry3", "x3", "y3", "ry4", "x4", "a1", "z0"}


@pytest.mark.parametrize(
    "encoder_resolution",
    (
        0.0009,  # the XPD d-hutch spinner
        0.0001,
        0.01,
        1.0,
    ),
)
def test_rotation_motor_counts_per_rev(RE, encoder_resolution: float):
    """A revolution's worth of counts, times the size of a count, is 360 degrees.

    Asserted as that INVARIANT rather than as specific outputs, deliberately.
    encoder_resolution is the motor record's ERES - "Encoder Step Size (EGU)",
    the size of one count in degrees - so counts per revolution divides by it.
    Until 2026-09-21 this multiplied instead, which is dimensionally
    degrees-squared per count and truncates to 0 for any encoder finer than
    about 0.0028 deg/count. A test asserting output values passed that formula
    happily; this one cannot, whatever the numbers are.

    The tolerance is one count, which is what int() truncation can cost.
    """
    with init_devices(mock=True):
        motor = RotationMotor("TEST:ROT:")

    counts_per_rev = motor.get_encoder_counts_per_rev(encoder_resolution)

    assert counts_per_rev > 0
    assert abs(counts_per_rev * encoder_resolution - 360.0) <= encoder_resolution
