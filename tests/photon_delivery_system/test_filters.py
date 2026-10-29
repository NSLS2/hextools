from typing import cast

import bluesky.plan_stubs as bps
import pytest
from bluesky.run_engine import RunEngine
from bluesky.utils import FailedStatus
from ophyd_async.core import (
    callback_on_mock_put,
    get_mock_put,
    init_devices,
    set_mock_value,
)

from hextools.bl_constants import fetch_beamline_constants
from hextools.photon_delivery_system.filters import (
    Filter,
    Filter1DownstreamPosition,
    Filter1UpstreamPosition,
    Filter2Position,
    Filter3Position,
    Filter4Position,
    FilterPosition,
    FilterSetting,
    set_filters,
)
from hextools.photon_delivery_system.shutter import PPSShutter, ShutterStatus


@pytest.fixture
def test_filter() -> Filter:
    with init_devices(mock=True):
        f = Filter("TEST:FILTER:MTR", "TEST:FILTER:IN_POS", name="filter1_upstream")
    set_mock_value(f.in_position, True)
    return f


@pytest.mark.parametrize(
    "position, expected_description",
    [
        (FilterPosition.PASS_THROUGH, "pass through"),
        (FilterPosition.POS_1, "6 mm Cu"),
        (FilterPosition.POS_3, "pos 3"),
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
        RE(bps.mv(test_filter, cast(FilterPosition, len(FilterPosition))))
    assert "Invalid filter position" in str(exc_info.value.__cause__)


def test_filter_requires_its_name_in_the_beamline_constants():
    with pytest.raises(KeyError, match="not_a_filter"):
        Filter("TEST:FILTER:MTR", "TEST:FILTER:IN_POS", name="not_a_filter")


def _filters() -> dict[str, Filter]:
    return {
        name: Filter(f"TEST:{name}:MTR", f"TEST:{name}:IN_POS", name=name)
        for name in fetch_beamline_constants("filters")
    }


def test_filter_parses_named_and_described_positions():
    positions = _filters()["filter1_upstream"].positions
    assert positions[FilterPosition.UPPER_LIMIT] == FilterSetting(68.0, None)
    assert positions[FilterPosition.PASS_THROUGH] == FilterSetting(66.6, None)
    assert positions[FilterPosition.POS_1] == FilterSetting(41.5, "6 mm Cu")
    assert positions[FilterPosition.POS_2] == FilterSetting(6.5, "6 mm Al")
    assert positions[FilterPosition.POS_3] == FilterSetting(-28.5, None)
    assert positions[FilterPosition.POS_4] == FilterSetting(-58.0, "Bo-Hori.")
    assert positions[FilterPosition.LOWER_LIMIT] == FilterSetting(-63.2, None)


def test_filter_supports_undescribed_middle_positions():
    positions = _filters()["filter3"].positions
    for slot in (
        FilterPosition.POS_1,
        FilterPosition.POS_2,
        FilterPosition.POS_3,
        FilterPosition.POS_4,
    ):
        assert positions[slot].description is None
    assert positions[FilterPosition.POS_1].position == pytest.approx(40.0)


def test_every_filter_maps_every_enum_position():
    for f in _filters().values():
        assert set(f.positions) == set(FilterPosition)


def _set_filters_devices() -> dict[str, Filter]:
    """The set_filters filters from the beamline constants, mocked under the RE loop."""
    # init_devices connects only devices bound to local names
    with init_devices(mock=True):
        filter1_upstream = Filter("F1U:MTR", "F1U:IN_POS", name="filter1_upstream")
        filter1_downstream = Filter("F1D:MTR", "F1D:IN_POS", name="filter1_downstream")
        filter2 = Filter("F2:MTR", "F2:IN_POS", name="filter2")
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


def test_filter_choices_are_pass_through_and_each_filters_materials():
    assert {p.name: p.value for p in Filter1UpstreamPosition} == {
        "PASS_THROUGH": "Pass Through",
        "POS_1": "6 mm Cu",
        "POS_2": "6 mm Al",
        "POS_4": "Bo-Hori.",
    }
    assert "6 mm Al" in list(Filter1DownstreamPosition)
    assert "1 mm Cu" in list(Filter2Position)
    # Filters 3 and 4 hold no described materials
    assert list(Filter3Position) == list(Filter4Position) == ["Pass Through"]


async def test_set_filters_moves_only_the_chosen_filters(RE: RunEngine):
    devices = _set_filters_devices()
    for device in devices.values():
        set_mock_value(device.filter_motor.user_readback, 999.0)

    RE(
        set_filters(
            filter1_upstream_setting="6 mm Cu",  # type: ignore
            filter2_setting="1 mm Cu",  # type: ignore
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
            filter1_upstream_setting="6 mm Cu",  # type: ignore
            filter1_downstream_setting="6 mm Al",  # type: ignore
            filter2_setting="1 mm Cu",  # type: ignore
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
        ({"filter2_setting": "6 mm Al"}, "is not a valid Filter2Position"),
    ],
)
def test_set_filters_refuses_bad_input(RE: RunEngine, kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        RE(set_filters(**kwargs, **_set_filters_devices()))


def _trip_fe_interlock_on_move(filt: Filter, fe_shutter: PPSShutter) -> None:
    """A filter move closes the FE shutter and drops its interlock acknowledge."""

    def move(value, **_):
        set_mock_value(fe_shutter.status, ShutterStatus.CLOSED)
        set_mock_value(fe_shutter.interlock_ack.acknowledged, False)
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
            filter2_setting="1 mm Cu",  # type: ignore
            filter2=devices["filter2"],
            fe_shutter=fe_shutter,
        )
    )

    get_mock_put(fe_shutter.interlock_ack.request).assert_called_once()
    assert await fe_shutter.interlock_ack.acknowledged.get_value()
    expected = ShutterStatus.OPEN if fe_was_open else ShutterStatus.CLOSED
    assert await fe_shutter.status.get_value() == expected


async def test_set_filters_leaves_an_acknowledged_interlock_alone(RE: RunEngine):
    devices = _set_filters_devices()
    fe_shutter = _fe_shutter()

    RE(
        set_filters(
            filter2_setting="1 mm Cu",  # type: ignore
            filter2=devices["filter2"],
            fe_shutter=fe_shutter,
        )
    )

    get_mock_put(fe_shutter.interlock_ack.request).assert_not_called()


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
    devices = _set_filters_devices()
    fe_shutter = _fe_shutter(ShutterStatus.OPEN)
    _trip_fe_interlock_on_move(devices["filter2"], fe_shutter)
    callback_on_mock_put(fe_shutter.interlock_ack.request, lambda *_, **__: None)

    with pytest.raises(FailedStatus):
        RE(
            set_filters(
                filter2_setting="1 mm Cu",  # type: ignore
                filter2=devices["filter2"],
                fe_shutter=fe_shutter,
            )
        )

    assert await fe_shutter.status.get_value() == ShutterStatus.CLOSED
