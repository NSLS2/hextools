from importlib import resources

import bluesky.plan_stubs as bps
import pytest
import yaml
from bluesky.run_engine import RunEngine
from bluesky.utils import FailedStatus
from ophyd_async.core import (
    callback_on_mock_put,
    get_mock_put,
    init_devices,
    set_mock_value,
)

from hextools.photon_delivery_system.filters import (
    SET_FILTERS_NAMES,
    SETTING_POSITIONS,
    Filter,
    FilterPosition,
    FilterSetting,
    load_filters,
    set_filters,
)
from hextools.photon_delivery_system.shutter import PPSShutter, ShutterStatus

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
    assert positions[FilterPosition.POS_1] == FilterSetting(41.5, "6 mm Cu")
    assert positions[FilterPosition.POS_2] == FilterSetting(6.5, "6 mm Al")
    assert positions[FilterPosition.POS_3] == FilterSetting(-28.5, None)
    assert positions[FilterPosition.POS_4] == FilterSetting(-58.0, "Bo-Hori.")
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


def _set_filters_devices() -> dict[str, Filter]:
    """The set_filters filters from filters.yml, mocked under the RE loop."""
    # init_devices connects only devices bound to local names
    with init_devices(mock=True):
        filter1_upstream, filter1_downstream, filter2 = (
            f for f in load_filters() if f.name in SET_FILTERS_NAMES
        )
    devices = {
        "filter1_upstream": filter1_upstream,
        "filter1_downstream": filter1_downstream,
        "filter2": filter2,
    }
    for device in devices.values():
        set_mock_value(device.in_position, True)
        callback_on_mock_put(
            device.filter_motor.user_setpoint,
            lambda value, device=device, **_: set_mock_value(
                device.filter_motor.user_readback, value
            ),
        )
    return devices


def _fe_shutter(status: ShutterStatus = ShutterStatus.CLOSED) -> PPSShutter:
    with init_devices(mock=True):
        fe_shutter = PPSShutter("TEST:FE:", name="fe_shutter")
    set_mock_value(fe_shutter.status, status)
    return fe_shutter


def test_set_filters_choices_are_each_filters_materials():
    raw = _raw_filter_config()
    for name in SET_FILTERS_NAMES:
        slots = [raw[name]["positions"][f"pos_{i}"] for i in range(1, 5)]
        materials = [s["description"] for s in slots if isinstance(s, dict)]
        assert list(SETTING_POSITIONS[name]) == ["pass through", *materials]
    assert list(SETTING_POSITIONS["filter1_upstream"]) == [
        "pass through",
        "6 mm Cu",
        "6 mm Al",
        "Bo-Hori.",
    ]
    assert "6 mm Al" in SETTING_POSITIONS["filter1_downstream"]
    assert "1 mm Cu" in SETTING_POSITIONS["filter2"]


async def test_set_filters_moves_only_the_chosen_filters(RE: RunEngine):
    devices = _set_filters_devices()
    for device in devices.values():
        set_mock_value(device.filter_motor.user_readback, 999.0)

    RE(
        set_filters(
            filter1_upstream_setting="6 mm Cu",
            filter2_setting="1 mm Cu",
            filter1_upstream=devices["filter1_upstream"],
            filter1_downstream=devices["filter1_downstream"],
            filter2=devices["filter2"],
            fe_shutter=_fe_shutter(),
        )
    )

    positions = {
        name: await d.filter_motor.user_readback.get_value()
        for name, d in devices.items()
    }
    assert positions == {
        "filter1_upstream": pytest.approx(41.5),
        "filter1_downstream": 999.0,
        "filter2": pytest.approx(-24.8),
    }


async def test_set_filters_sets_all_three(RE: RunEngine):
    devices = _set_filters_devices()

    RE(
        set_filters(
            filter1_upstream_setting="6 mm Cu",
            filter1_downstream_setting="6 mm Al",
            filter2_setting="1 mm Cu",
            filter1_upstream=devices["filter1_upstream"],
            filter1_downstream=devices["filter1_downstream"],
            filter2=devices["filter2"],
            fe_shutter=_fe_shutter(),
        )
    )

    descriptions = {n: await d.description.get_value() for n, d in devices.items()}
    assert descriptions == {
        "filter1_upstream": "6 mm Cu",
        "filter1_downstream": "6 mm Al",
        "filter2": "1 mm Cu",
    }


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "at least one"),
        ({"filter2_setting": "6 mm Al"}, "not a setting of filter2"),
    ],
)
def test_set_filters_refuses_bad_input(RE: RunEngine, kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        RE(set_filters(**kwargs, **_set_filters_devices()))


def _trip_fe_interlock_on_move(filt: Filter, fe_shutter: PPSShutter) -> None:
    """A filter move closes the FE shutter and drops its interlock acknowledge."""

    def move(value, **_):
        set_mock_value(fe_shutter.status, ShutterStatus.CLOSED)
        set_mock_value(fe_shutter.interlock.acknowledged, False)
        set_mock_value(filt.filter_motor.user_readback, value)

    callback_on_mock_put(filt.filter_motor.user_setpoint, move)


@pytest.mark.parametrize("fe_was_open", [True, False])
async def test_set_filters_acknowledges_the_interlock_and_restores_the_fe_shutter(
    RE: RunEngine, monkeypatch, fe_was_open: bool
):
    monkeypatch.setattr("random.uniform", lambda *_: 0.0)
    devices = _set_filters_devices()
    fe_shutter = _fe_shutter(
        ShutterStatus.OPEN if fe_was_open else ShutterStatus.CLOSED
    )
    _trip_fe_interlock_on_move(devices["filter2"], fe_shutter)

    RE(
        set_filters(
            filter2_setting="1 mm Cu",
            filter2=devices["filter2"],
            fe_shutter=fe_shutter,
        )
    )

    get_mock_put(fe_shutter.interlock.request).assert_called_once()
    assert await fe_shutter.interlock.acknowledged.get_value()
    expected = ShutterStatus.OPEN if fe_was_open else ShutterStatus.CLOSED
    assert await fe_shutter.status.get_value() == expected


async def test_set_filters_leaves_an_acknowledged_interlock_alone(RE: RunEngine):
    devices = _set_filters_devices()
    fe_shutter = _fe_shutter()

    RE(
        set_filters(
            filter2_setting="1 mm Cu",
            filter2=devices["filter2"],
            fe_shutter=fe_shutter,
        )
    )

    get_mock_put(fe_shutter.interlock.request).assert_not_called()


async def test_filter_move_reports_progress(RE: RunEngine):
    filt = _set_filters_devices()["filter2"]
    names: list[str] = []

    status = filt.set(FilterPosition.POS_1)
    status.watch(lambda **kw: names.append(kw["name"]))  # type: ignore
    await status

    assert names
    assert names[-1] == filt.filter_motor.name


async def test_set_filters_fails_and_keeps_the_fe_shutter_closed_if_ack_never_comes(
    RE: RunEngine, monkeypatch
):
    monkeypatch.setattr(
        "hextools.photon_delivery_system.shutter.INTERLOCK_ACK_TIMEOUT", 0.1
    )
    devices = _set_filters_devices()
    fe_shutter = _fe_shutter(ShutterStatus.OPEN)
    _trip_fe_interlock_on_move(devices["filter2"], fe_shutter)
    callback_on_mock_put(fe_shutter.interlock.request, lambda *_, **__: None)

    with pytest.raises(FailedStatus):
        RE(
            set_filters(
                filter2_setting="1 mm Cu",
                filter2=devices["filter2"],
                fe_shutter=fe_shutter,
            )
        )

    assert await fe_shutter.status.get_value() == ShutterStatus.CLOSED
