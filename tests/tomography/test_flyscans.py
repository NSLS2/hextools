from collections.abc import Callable

import pytest
from bluesky import RunEngine
from bluesky import plan_stubs as bps
from ophyd_async.core import (
    Device,
    DeviceVector,
    StaticPathProvider,
    callback_on_mock_put,
    get_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.epics.core import epics_signal_r, epics_signal_rw
from ophyd_async.fastcs.panda import DatasetTable, HDFPanda, PandaHdf5DatasetType

from hextools.motors import RotationMotor
from hextools.photon_delivery_system import Shutter
from hextools.tomography import flyscans
from hextools.tomography.flyscans import tomo_flyscan


class _Bits(Device):
    def __init__(self, prefix: str, name: str = ""):
        self.a = epics_signal_rw(int, f"{prefix}A")
        self.b = epics_signal_rw(int, f"{prefix}B")
        super().__init__(name=name)


class _Calc(Device):
    def __init__(self, prefix: str, name: str = ""):
        self.out = epics_signal_r(float, f"{prefix}OUT")
        self.out_dataset = epics_signal_rw(str, f"{prefix}OUT:DATASET")
        self.out_units = epics_signal_rw(str, f"{prefix}OUT:UNITS")
        self.out_scale = epics_signal_rw(float, f"{prefix}OUT:SCALE")
        self.out_offset = epics_signal_rw(float, f"{prefix}OUT:OFFSET")
        super().__init__(name=name)


class _HexPanda(HDFPanda):
    """The blocks tomo_flyscan uses that the real PandA discovers from its IOC."""

    def __init__(self, prefix: str, path_provider, name: str = ""):
        self.bits = _Bits(f"{prefix}BITS:")
        self.calc = DeviceVector({1: _Calc(f"{prefix}CALC1:")})
        super().__init__(prefix, path_provider, name=name)


@pytest.fixture
def flyscan_devices(
    static_path_provider: StaticPathProvider,
) -> tuple[_HexPanda, RotationMotor]:
    with init_devices(mock=True):
        panda = _HexPanda("PANDA:", static_path_provider, name="panda1")
        rot_motor = RotationMotor("ROT:", name="rot")
    set_mock_value(panda.data.directory_exists, True)
    set_mock_value(
        panda.data.datasets,
        DatasetTable(name=["Angle"], dtype=[PandaHdf5DatasetType.FLOAT_64]),
    )
    callback_on_mock_put(panda.pcap.arm, lambda v: set_mock_value(panda.pcap.active, v))
    set_mock_value(rot_motor.encoder_resolution, 0.001)
    return panda, rot_motor


@pytest.mark.parametrize("fly_fails", [False, True])
async def test_tomo_flyscan_restores_rotation_velocity(
    RE: RunEngine,
    flyscan_devices: tuple[_HexPanda, RotationMotor],
    kinetix_det_factory: Callable[[int], KinetixDetector],
    two_shutters: tuple[Shutter, Shutter],
    monkeypatch: pytest.MonkeyPatch,
    fly_fails: bool,
):
    # The motor's fly prepare leaves VELO at the fly speed; the plan must put back
    # the operator's speed however it ends (Nghia Vo, 2026-10-23).
    panda, rot_motor = flyscan_devices
    fe_shutter, photon_shutter = two_shutters
    await rot_motor.velocity.set(5.0)

    def _fly(*_args, **_kwargs):
        if fly_fails:
            raise RuntimeError("fly failed")
        yield from bps.null()

    # The PandA and motor fly logic are not what is under test here.
    monkeypatch.setattr(flyscans, "_fly_stub", _fly)

    plan = tomo_flyscan(
        [kinetix_det_factory(1)],
        num_projections=10,
        exposure_time=0.01,
        start=0,
        stop=1,
        use_shutter=False,
        panda=panda,
        rot_motor=rot_motor,
        fe_shutter=fe_shutter,
        photon_shutter=photon_shutter,
    )
    if fly_fails:
        with pytest.raises(RuntimeError, match="fly failed"):
            RE(plan)
    else:
        RE(plan)

    speeds = [call.args[0] for call in get_mock_put(rot_motor.velocity).call_args_list]
    assert any(speed != 5.0 for speed in speeds)  # the fly prepare did change it
    assert await rot_motor.velocity.get_value() == 5.0
