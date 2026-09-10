import bluesky.plan_stubs as bps
import numpy as np
import pytest
from bluesky.run_engine import RunEngine
from ophyd_async.core import (
    callback_on_mock_execute,
    callback_on_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.adcore import (
    ADAcquireLogic,
    ADBaseIO,
    ADState,
    AreaDetector,
    NDStatsIO,
    PluginSignalDataLogic,
)

from hextools.photon_delivery_system.dclm import DCLM, BeamMode, change_energy
from hextools.photon_delivery_system.shutter import Shutter


@pytest.fixture
def dclm() -> DCLM:
    with init_devices(mock=True, child_name_separator="_"):
        dclm = DCLM("TEST:{")
    return dclm


@pytest.fixture
def photon_shutter() -> Shutter:
    with init_devices(mock=True):
        shutter = Shutter("TEST:PSH:", name="photon_shutter")
    set_mock_value(shutter.status, False)
    callback_on_mock_execute(
        shutter.open_cmd, lambda *_: set_mock_value(shutter.status, True)
    )
    callback_on_mock_execute(
        shutter.close_cmd, lambda *_: set_mock_value(shutter.status, False)
    )
    return shutter


@pytest.mark.parametrize(
    "mode, expected_xtal1, expected_bs",
    [
        (BeamMode.WHITE, DCLM.xtal1_out, DCLM.beam_stop_out),
        (
            BeamMode.MONOCHROMATIC,
            DCLM.xtal1_in,
            DCLM.beam_stop_in,
        ),
    ],
)
async def test_monochromator_beam_mode_switch(
    RE: RunEngine,
    dclm: DCLM,
    mode: BeamMode,
    expected_xtal1: float,
    expected_bs: float,
):
    RE(bps.mv(dclm, mode))

    xtal1_pos = await dclm.xtal1_vertical_trans.user_readback.get_value()
    bs_pos = await dclm.cooled_beam_stop.user_readback.get_value()
    assert xtal1_pos == pytest.approx(expected_xtal1)
    assert bs_pos == pytest.approx(expected_bs)

    beam_mode = await dclm.beam_mode.get_value()
    assert beam_mode == mode


@pytest.mark.parametrize("energy", [8.0, 10.0, 12.0])
async def test_change_energy_moves_motors(
    RE: RunEngine, dclm: DCLM, photon_shutter: Shutter, energy: float
):
    bragg_angle = np.arcsin(1.977 / energy)
    expected_angle = 35.2544 - np.rad2deg(bragg_angle)
    expected_z = 25.0 / np.tan(2 * bragg_angle)

    RE(change_energy(energy, dclm=dclm, photon_shutter=photon_shutter))

    assert await dclm.xtal1_pitch.user_readback.get_value() == pytest.approx(
        expected_angle
    )
    assert await dclm.xtal2_pitch.user_readback.get_value() == pytest.approx(
        expected_angle
    )
    assert await dclm.xtal2_z.user_readback.get_value() == pytest.approx(expected_z)
    # The reset finalizer moves the fluorescence screen out at the end.
    assert await dclm.flourescence_screen.user_readback.get_value() == pytest.approx(
        dclm.fs_out
    )
    assert await dclm.xtal1_vertical_trans.user_readback.get_value() == pytest.approx(
        dclm.xtal1_in
    )
    assert await dclm.cooled_beam_stop.user_readback.get_value() == pytest.approx(
        dclm.beam_stop_in
    )


async def test_change_energy_raises_if_not_monochromatic(
    RE: RunEngine, dclm: DCLM, photon_shutter: Shutter
):
    set_mock_value(dclm.xtal1_vertical_trans.user_readback, dclm.xtal1_out)
    set_mock_value(dclm.cooled_beam_stop.user_readback, dclm.beam_stop_out)

    with pytest.raises(RuntimeError, match="not in monochromatic mode"):
        RE(change_energy(10.0, dclm=dclm, photon_shutter=photon_shutter))


def _make_fs_camera() -> AreaDetector:
    with init_devices(mock=True, child_name_separator="_"):
        driver = ADBaseIO("TEST:CAM:cam1:")
        stats1 = NDStatsIO("TEST:CAM:Stats1:")
        fs_camera = AreaDetector(
            driver,
            acquire_logic=ADAcquireLogic(driver),
            plugins={"stats1": stats1},
        )
    fs_camera.add_detector_logics(
        PluginSignalDataLogic(driver=driver, signal=stats1.total)
    )
    set_mock_value(driver.detector_state, ADState.IDLE)
    return fs_camera


async def test_change_energy_with_auto_tune(
    RE: RunEngine, dclm: DCLM, photon_shutter: Shutter
):
    energy = 10.0
    bragg_angle = np.arcsin(1.977 / energy)
    expected_angle = 35.2544 - np.rad2deg(bragg_angle)
    sigma = 0.05

    fs_camera = _make_fs_camera()
    driver = fs_camera.driver
    stats1 = fs_camera.get_plugin_by_name("stats1", NDStatsIO)

    # Simulate a Gaussian peak: read motor position when acquire fires.
    async def on_acquire(value, **_):
        if value:
            pos = await dclm.xtal2_pitch.user_readback.get_value()
            intensity = float(np.exp(-((pos - expected_angle) ** 2) / (2 * sigma**2)))
            set_mock_value(stats1.total, intensity)

    callback_on_mock_put(driver.acquire, on_acquire)

    RE(
        change_energy(
            energy, dclm=dclm, fs_camera=fs_camera, photon_shutter=photon_shutter
        )
    )

    # Auto-tune should converge on the peak center.
    assert await dclm.xtal2_pitch.user_readback.get_value() == pytest.approx(
        expected_angle, abs=0.01
    )
    # Fluorescence screen moved out after tuning.
    assert await dclm.flourescence_screen.user_readback.get_value() == pytest.approx(
        dclm.fs_out
    )


async def test_change_energy_no_peak_coarse_scan(
    RE: RunEngine, dclm: DCLM, photon_shutter: Shutter, mocker
):
    fs_camera = _make_fs_camera()

    mock_ps = mocker.patch(
        "hextools.photon_delivery_system.dclm.PeakStats", autospec=True
    ).return_value
    mock_ps.com = None

    with pytest.raises(RuntimeError, match="No peak found in coarse scan"):
        RE(
            change_energy(
                10.0, dclm=dclm, fs_camera=fs_camera, photon_shutter=photon_shutter
            )
        )


async def test_change_energy_no_peak_fine_scan(
    RE: RunEngine, dclm: DCLM, photon_shutter: Shutter, mocker
):
    fs_camera = _make_fs_camera()

    mock_ps = mocker.patch(
        "hextools.photon_delivery_system.dclm.PeakStats", autospec=True
    ).return_value
    # Coarse scan succeeds, fine scan fails.
    mock_ps.com = 5.0
    mock_ps.reset.side_effect = lambda: setattr(mock_ps, "com", None)

    with pytest.raises(RuntimeError, match="No peak found in fine scan"):
        RE(
            change_energy(
                10.0, dclm=dclm, fs_camera=fs_camera, photon_shutter=photon_shutter
            )
        )


@pytest.mark.parametrize("energy", [8.0, 10.0, 12.0])
async def test_energy_derived_signal(RE: RunEngine, dclm: DCLM, energy: float):
    bragg_angle = np.arcsin(1.977 / energy)
    pitch_angle = 35.2544 - np.rad2deg(bragg_angle)
    set_mock_value(dclm.xtal1_pitch.user_readback, pitch_angle)

    result = await dclm.energy.get_value()
    assert result == pytest.approx(energy)
