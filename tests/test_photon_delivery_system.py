import asyncio
import time
from importlib import resources
from typing import Annotated as A

import bluesky.plan_stubs as bps
import numpy as np
import pytest
import yaml
from bluesky.run_engine import RunEngine
from bluesky.utils import FailedStatus
from ophyd_async.core import SignalR, callback_on_mock_execute, callback_on_mock_put, init_devices, set_mock_value
from ophyd_async.epics.adcore import (
    ADAcquireLogic,
    ADBaseIO,
    ADState,
    AreaDetector,
    NDStatsIO,
    PluginSignalDataLogic,
)
from ophyd_async.epics.core import PvSuffix

from hextools.photon_delivery_system import (
    DCLM,
    BeamMode,
    Filter,
    FilterPosition,
    FilterSetting,
    Shutter,
    change_energy,
    load_filters,
)


@pytest.fixture
def shutter() -> Shutter:
    with init_devices(mock=True):
        ps = Shutter("TEST:SHUTTER:", name="test_shutter")
    return ps


async def _delayed_readback(ps: Shutter, value: bool, delay: float):
    await asyncio.sleep(delay)
    set_mock_value(ps.status, value)


@pytest.mark.parametrize("delay", [0.0, 0.05, 0.1, 0.15])
async def test_shutter_open_close_behavior(
    RE: RunEngine, shutter: Shutter, delay: float
):
    set_mock_value(shutter.status, False)

    callback_on_mock_execute(
        shutter.open_cmd,
        lambda *_: asyncio.ensure_future(_delayed_readback(shutter, True, delay)),
    )
    callback_on_mock_execute(
        shutter.close_cmd,
        lambda *_: asyncio.ensure_future(_delayed_readback(shutter, False, delay)),
    )

    t0 = time.monotonic()
    RE(bps.mv(shutter, True))
    open_duration = time.monotonic() - t0
    assert await shutter.status.get_value() is True
    assert open_duration >= delay
    assert open_duration < delay + 0.1

    t0 = time.monotonic()
    RE(bps.mv(shutter, False))
    close_duration = time.monotonic() - t0
    assert await shutter.status.get_value() is False
    assert close_duration >= delay
    assert close_duration < delay + 0.1


_TEST_FILTER_POSITIONS = {
    FilterPosition.PASS_THROUGH: FilterSetting(0.0, "pass through"),
    FilterPosition.POS_1: FilterSetting(10.0, "Cu 100um"),
    FilterPosition.POS_2: FilterSetting(20.0, "Cu 50um"),
}


@pytest.fixture
def test_filter() -> Filter:
    with init_devices(mock=True):
        f = Filter(
            "TEST:FILTER:MTR",
            _TEST_FILTER_POSITIONS,
            "TEST:FILTER:IN_POS",
            name="test_filter",
        )
    set_mock_value(f.in_position, True)
    return f


@pytest.mark.parametrize(
    "position, expected_description",
    [
        (FilterPosition.PASS_THROUGH, "pass through"),
        (FilterPosition.POS_1, "Cu 100um"),
        (FilterPosition.POS_2, "Cu 50um"),
    ],
)
async def test_filter_description_changes_with_position(
    RE: RunEngine,
    test_filter: Filter,
    position: FilterPosition,
    expected_description: str,
):
    callback_on_mock_put(
        test_filter.filter_motor.user_setpoint,
        lambda value, **_: set_mock_value(
            test_filter.filter_motor.user_readback, value
        ),
    )

    RE(bps.mv(test_filter, position))
    desc = await test_filter.description.get_value()
    assert desc == expected_description


async def test_filter_rejects_undefined_position(RE: RunEngine, test_filter: Filter):
    with pytest.raises(FailedStatus) as exc_info:
        RE(bps.mv(test_filter, FilterPosition.POS_3))
    assert "Invalid filter position" in str(exc_info.value.__cause__)


@pytest.fixture
def dclm() -> DCLM:
    with init_devices(mock=True, child_name_separator="_"):
        dclm = DCLM("TEST:{")
    return dclm


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


def _raw_filter_config() -> dict:
    source = resources.files("hextools").joinpath("filters.yml")
    return yaml.safe_load(source.read_text(encoding="utf-8"))["filters"]


def test_load_filters_returns_all_entries_in_order():
    raw = _raw_filter_config()
    filters = load_filters()
    assert [f.name for f in filters] == list(raw.keys())
    assert all(isinstance(f, Filter) for f in filters)


def test_load_filters_parses_named_and_described_positions():
    filters = {f.name: f for f in load_filters()}
    positions = filters["filter1_upstream"].positions
    assert positions[FilterPosition.UPPER_LIMIT] == FilterSetting(68.0, None)
    assert positions[FilterPosition.PASS_THROUGH] == FilterSetting(66.6, None)
    assert positions[FilterPosition.POS_1] == FilterSetting(41.5, "12 mm SiC")
    assert positions[FilterPosition.POS_2] == FilterSetting(6.5, "9 mm SiC")
    assert positions[FilterPosition.POS_3] == FilterSetting(-28.5, "6 mm SiC")
    assert positions[FilterPosition.POS_4] == FilterSetting(-58.0, "3 mm SiC")
    assert positions[FilterPosition.LOWER_LIMIT] == FilterSetting(-63.2, None)


def test_load_filters_supports_undescribed_middle_positions():
    positions = {f.name: f for f in load_filters()}["filter3"].positions
    for slot in (
        FilterPosition.POS_1,
        FilterPosition.POS_2,
        FilterPosition.POS_3,
        FilterPosition.POS_4,
    ):
        assert positions[slot].description is None
    assert positions[FilterPosition.POS_1].position == pytest.approx(40.0)


def test_load_filters_matches_raw_yaml_pvs_and_counts():
    raw = _raw_filter_config()
    filters = {f.name: f for f in load_filters()}
    for name, spec in raw.items():
        f = filters[name]
        assert len(f.positions) == len(spec["positions"])
        assert spec["motor_pv"] in f.filter_motor.user_readback.source
        assert f.in_position.source.endswith(spec["in_position_switch"])


def test_load_filters_maps_every_enum_position():
    for f in load_filters():
        assert set(f.positions) == set(FilterPosition)


def test_load_filters_requires_one_position_per_enum(monkeypatch):
    bad_config = {
        "filters": {
            "filter_bad": {
                "motor_pv": "TEST:MTR",
                "in_position_switch": "TEST:IN_POS",
                "positions": {"upper_limit": 1.0, "pass_through": 2.0},
            }
        }
    }
    monkeypatch.setattr(yaml, "safe_load", lambda *_a, **_k: bad_config)
    with pytest.raises(ValueError, match="exactly 7 positions"):
        load_filters()




@pytest.mark.parametrize("energy", [8.0, 10.0, 12.0])
async def test_change_energy_moves_motors(RE: RunEngine, dclm: DCLM, energy: float):
    bragg_angle = np.arcsin(1.977 / energy)
    expected_angle = 35.2544 - np.rad2deg(bragg_angle)
    expected_z = 25.0 / np.tan(2 * bragg_angle)

    RE(change_energy(dclm, energy))

    assert await dclm.xtal1_pitch.user_readback.get_value() == pytest.approx(
        expected_angle
    )
    assert await dclm.xtal2_pitch.user_readback.get_value() == pytest.approx(
        expected_angle
    )
    assert await dclm.xtal2_z.user_readback.get_value() == pytest.approx(expected_z)
    assert await dclm.flourescence_screen.user_readback.get_value() == pytest.approx(
        dclm.fs_in
    )
    assert await dclm.xtal1_vertical_trans.user_readback.get_value() == pytest.approx(
        dclm.xtal1_in
    )
    assert await dclm.cooled_beam_stop.user_readback.get_value() == pytest.approx(
        dclm.beam_stop_in
    )


async def test_change_energy_raises_if_not_monochromatic(RE: RunEngine, dclm: DCLM):
    set_mock_value(dclm.xtal1_vertical_trans.user_readback, dclm.xtal1_out)
    set_mock_value(dclm.cooled_beam_stop.user_readback, dclm.beam_stop_out)

    with pytest.raises(RuntimeError, match="not in monochromatic mode"):
        RE(change_energy(dclm, 10.0))


class NDStatsWithMeanIO(NDStatsIO):
    mean: A[SignalR[float], PvSuffix("MeanValue_RBV")]


async def test_change_energy_with_auto_tune(RE: RunEngine, dclm: DCLM):
    energy = 10.0
    bragg_angle = np.arcsin(1.977 / energy)
    expected_angle = 35.2544 - np.rad2deg(bragg_angle)
    sigma = 0.05

    with init_devices(mock=True, child_name_separator="_"):
        driver = ADBaseIO("TEST:CAM:cam1:")
        stats1 = NDStatsWithMeanIO("TEST:CAM:Stats1:")
        fs_camera = AreaDetector(
            driver,
            acquire_logic=ADAcquireLogic(driver),
            plugins={"stats1": stats1},
        )
    fs_camera.add_detector_logics(
        PluginSignalDataLogic(driver=driver, signal=stats1.mean)
    )
    set_mock_value(driver.detector_state, ADState.IDLE)

    # Simulate a Gaussian peak: read motor position when acquire fires
    async def on_acquire(value, **_):
        if value:
            pos = await dclm.xtal2_pitch.user_readback.get_value()
            intensity = float(np.exp(-((pos - expected_angle) ** 2) / (2 * sigma**2)))
            set_mock_value(stats1.mean, intensity)

    callback_on_mock_put(driver.acquire, on_acquire)

    RE(change_energy(dclm, energy, fs_camera))

    # Auto-tune should converge on the peak center
    assert await dclm.xtal2_pitch.user_readback.get_value() == pytest.approx(
        expected_angle, abs=0.01
    )
    # Fluorescence screen moved out after tuning
    assert await dclm.flourescence_screen.user_readback.get_value() == pytest.approx(
        dclm.fs_out
    )


async def test_change_energy_no_peak_coarse_scan(RE: RunEngine, dclm: DCLM, mocker):
    with init_devices(mock=True, child_name_separator="_"):
        driver = ADBaseIO("TEST:CAM:cam1:")
        stats1 = NDStatsWithMeanIO("TEST:CAM:Stats1:")
        fs_camera = AreaDetector(
            driver,
            acquire_logic=ADAcquireLogic(driver),
            plugins={"stats1": stats1},
        )
    fs_camera.add_detector_logics(
        PluginSignalDataLogic(driver=driver, signal=stats1.mean)
    )
    set_mock_value(driver.detector_state, ADState.IDLE)

    mock_ps = mocker.patch(
        "hextools.photon_delivery_system.PeakStats", autospec=True
    ).return_value
    mock_ps.com = None

    with pytest.raises(RuntimeError, match="No peak found in coarse scan"):
        RE(change_energy(dclm, 10.0, fs_camera))


async def test_change_energy_no_peak_fine_scan(RE: RunEngine, dclm: DCLM, mocker):
    with init_devices(mock=True, child_name_separator="_"):
        driver = ADBaseIO("TEST:CAM:cam1:")
        stats1 = NDStatsWithMeanIO("TEST:CAM:Stats1:")
        fs_camera = AreaDetector(
            driver,
            acquire_logic=ADAcquireLogic(driver),
            plugins={"stats1": stats1},
        )
    fs_camera.add_detector_logics(
        PluginSignalDataLogic(driver=driver, signal=stats1.mean)
    )
    set_mock_value(driver.detector_state, ADState.IDLE)

    mock_ps = mocker.patch(
        "hextools.photon_delivery_system.PeakStats", autospec=True
    ).return_value
    # Coarse scan succeeds, fine scan fails
    mock_ps.com = 5.0
    mock_ps.reset.side_effect = lambda: setattr(mock_ps, "com", None)

    with pytest.raises(RuntimeError, match="No peak found in fine scan"):
        RE(change_energy(dclm, 10.0, fs_camera))


@pytest.mark.parametrize("energy", [8.0, 10.0, 12.0])
async def test_energy_derived_signal(RE: RunEngine, dclm: DCLM, energy: float):
    bragg_angle = np.arcsin(1.977 / energy)
    pitch_angle = 35.2544 - np.rad2deg(bragg_angle)
    set_mock_value(dclm.xtal1_pitch.user_readback, pitch_angle)

    result = await dclm.energy.get_value()
    assert result == pytest.approx(energy)
