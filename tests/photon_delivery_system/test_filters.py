from importlib import resources

import bluesky.plan_stubs as bps
import pytest
import yaml
from bluesky.run_engine import RunEngine
from bluesky.utils import FailedStatus
from ophyd_async.core import callback_on_mock_put, init_devices, set_mock_value

from hextools.photon_delivery_system.filters import (
    Filter,
    FilterPosition,
    FilterSetting,
    load_filters,
)

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


def _raw_filter_config() -> dict:
    source = resources.files("hextools.photon_delivery_system").joinpath("filters.yml")
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
