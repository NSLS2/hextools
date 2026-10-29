"""Contracts requested in the operator-fix review."""

from typing import Any, cast

import pytest
from ophyd_async.core import EnableDisable, get_mock_put, set_mock_value

from hextools.photon_delivery_system.dclm import DCLM
from hextools.photon_delivery_system.filters import (
    Filter,
    Filter1UpstreamPosition,
    Filter2Position,
    set_filters,
)


@pytest.mark.parametrize("file_format", ["hdf", "tiff"])
async def test_direct_germ_trigger_enables_stats(germ, file_format):
    if file_format == "tiff":
        germ.save_as_tiff()
    set_mock_value(germ.stats1.enable_callbacks, EnableDisable.DISABLE)
    set_mock_value(germ.stats1.compute_statistics, False)
    await germ.stage()
    try:
        await germ.trigger()
        assert await germ.stats1.enable_callbacks.get_value() == EnableDisable.ENABLE
        assert await germ.stats1.compute_statistics.get_value() is True
    finally:
        await germ.unstage()


async def test_filter_settings_independent_of_display_name(RE):
    from ophyd_async.core import init_devices

    with init_devices(mock=True):
        device = Filter(
            "TEST:M", "TEST:IN", name="custom-label", settings_key="filter2"
        )
    assert device.settings_key == "filter2"
    device.set_name("another-label")
    assert await device.description.get_value() == "pass through"


def test_dclm_config_is_frozen_and_preserves_current_numbers():
    from dataclasses import FrozenInstanceError, asdict

    from hextools.bl_constants import fetch_beamline_constants

    device = DCLM("TEST:")
    assert asdict(device.config) == fetch_beamline_constants("monochromator")
    with pytest.raises(FrozenInstanceError):
        cast(Any, device.config).fs_in = 99


@pytest.mark.parametrize("change", ["unknown", "missing", "nonnumeric", "nonfinite"])
def test_dclm_rejects_invalid_constants(monkeypatch, change):
    from hextools.photon_delivery_system import dclm as module

    constants = module.fetch_beamline_constants("monochromator").copy()
    if change == "unknown":
        constants["typo"] = 1.0
    elif change == "missing":
        del constants["beam_stop_in"]
    else:
        constants["beam_stop_in"] = "wrong" if change == "nonnumeric" else float("nan")
    monkeypatch.setattr(module, "fetch_beamline_constants", lambda _: constants)
    with pytest.raises((TypeError, ValueError)):
        DCLM("TEST:")


@pytest.mark.parametrize("missing", ["pass_through", "pos_4"])
def test_filter_requires_all_seven_positions(monkeypatch, missing):
    from hextools.photon_delivery_system import filters as module

    settings = module.fetch_beamline_constants("filters")
    del settings["filter2"][missing]
    monkeypatch.setattr(module, "fetch_beamline_constants", lambda _: settings)
    with pytest.raises(ValueError, match="seven"):
        Filter("TEST:M", "TEST:IN", name="filter2", settings_key="filter2")


async def test_set_filters_only_resolves_selected_devices(RE):
    from ophyd_async.core import init_devices

    from hextools.photon_delivery_system.filters import FilterPosition
    from hextools.photon_delivery_system.shutter import PPSShutter, ShutterStatus

    with init_devices(mock=True):
        filter2 = Filter("TEST:M", "TEST:IN", name="filter2", settings_key="filter2")
        fe_shutter = PPSShutter("TEST:SH", name="fe_shutter")
    set_mock_value(fe_shutter.status, ShutterStatus.CLOSED)
    RE(
        set_filters(
            filter2_setting=Filter2Position("Pass Through"),
            filter2=filter2,
            fe_shutter=fe_shutter,
        )
    )
    get_mock_put(filter2.filter_motor.user_setpoint).assert_called_once_with(
        filter2.positions[FilterPosition.PASS_THROUGH].position
    )


def test_set_filters_validates_all_selections_before_device_lookup(RE):
    with pytest.raises(ValueError, match="not a valid"):
        RE(
            set_filters(
                filter1_upstream_setting=Filter1UpstreamPosition("Pass Through"),
                filter2_setting=cast(Any, "wrong"),
            )
        )
