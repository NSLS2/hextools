import copy
from collections.abc import Callable
from typing import Any

import pytest
from tiled.structures.core import Spec

from hextools import specs
from hextools.specs import (
    EDXD_CALIBRATION_V1,
    EDXD_SCAN_V1,
    ENERGY_AUTO_TUNE_V1,
    RADIOGRAPH_V1,
    TOMO_ALIGNMENT_SCAN_V1,
    TOMO_DARK_FLAT_V1,
    TOMO_FLYSCAN_V1,
    XRD_CALIBRATION_V1,
    SpecValidationError,
    validate_edxd_calibration,
    validate_edxd_scan,
    validate_energy_auto_tune,
    validate_radiograph,
    validate_run,
    validate_tomo_alignment_scan,
    validate_tomo_dark_flat,
    validate_tomo_flyscan,
    validate_xrd_calibration,
)


class FakeRun:
    """Minimal stand-in for a BlueskyRunV3: a start document and streams of field names."""

    def __init__(self, start: dict[str, Any], streams: dict[str, list[str]]):
        self.start = start
        self.streams = streams

    def __contains__(self, key: str) -> bool:
        return key in self.streams

    def __getitem__(self, key: str) -> list[str]:
        return self.streams[key]

    def __iter__(self):
        return iter(self.streams)


def _tomo_dark_flat() -> FakeRun:
    return FakeRun(
        {
            "plan_name": "tomo_dark_flat",
            "exposure_time": 0.1,
            "num_dark_images": 20,
            "num_flat_images": 50,
            "x_offset": 5.0,
        },
        {"dark": ["kinetix1", "ts_kinetix1"], "flat": ["kinetix1", "ts_kinetix1"]},
    )


def _tomo_flyscan() -> FakeRun:
    return FakeRun(
        {
            "plan_name": "tomo_flyscan",
            "detectors": ["kinetix1"],
            "num_points": 10,
            "exposure_time": 0.1,
            "acquire_period": None,
            "time_based": False,
            "start_position": 0.0,
            "stop_position": 180.0,
            "images_to_average": 1,
            "last_dark_flat_uid": "dark-flat-uid",
            "last_dark_flat_scan_id": 1,
        },
        {"primary": ["kinetix1", "rotation_angle"]},
    )


def _tomo_alignment_scan() -> FakeRun:
    return FakeRun(
        {
            "plan_name": "tomography_alignment_scan",
            "description": "Tomography alignment scan",
            "detectors": ["kinetix1"],
            "motors": ["rot"],
        },
        {"flat": ["kinetix1"], "primary": ["kinetix1", "rot"]},
    )


def _radiograph() -> FakeRun:
    return FakeRun(
        {"plan_name": "take_radiograph", "detectors": ["kinetix1", "kinetix2"]},
        {"primary": ["kinetix1", "kinetix2"]},
    )


def _edxd_scan() -> FakeRun:
    return FakeRun(
        {"plan_name": "edxd_grid_scan", "detectors": ["germ"], "motors": ["x", "y"]},
        {"primary": ["germ", "germ-stats1-total", "x", "y"]},
    )


def _edxd_calibration() -> FakeRun:
    return FakeRun(
        {"plan_name": "edxd_calib_scan"},
        {"primary": ["germ", "germ-stats1-total"]},
    )


def _xrd_calibration() -> FakeRun:
    return FakeRun(
        {
            "plan_name": "xrd_calibration",
            "description": "Energy-geometry calibration",
            "detectors": ["perkin_elmer"],
            "motors": ["det_z"],
        },
        {"primary": ["perkin_elmer", "det_z"]},
    )


def _energy_auto_tune() -> FakeRun:
    return FakeRun(
        {"plan_name": "energy_auto_tune", "detectors": ["screen"], "motors": ["pitch2"]},
        {"primary": ["screen", "screen-stats1-mean_value", "pitch2"]},
    )


CASES: list[tuple[Spec, Callable[..., None], Callable[[], FakeRun]]] = [
    (TOMO_DARK_FLAT_V1, validate_tomo_dark_flat, _tomo_dark_flat),
    (TOMO_FLYSCAN_V1, validate_tomo_flyscan, _tomo_flyscan),
    (TOMO_ALIGNMENT_SCAN_V1, validate_tomo_alignment_scan, _tomo_alignment_scan),
    (RADIOGRAPH_V1, validate_radiograph, _radiograph),
    (EDXD_SCAN_V1, validate_edxd_scan, _edxd_scan),
    (EDXD_CALIBRATION_V1, validate_edxd_calibration, _edxd_calibration),
    (XRD_CALIBRATION_V1, validate_xrd_calibration, _xrd_calibration),
    (ENERGY_AUTO_TUNE_V1, validate_energy_auto_tune, _energy_auto_tune),
]
CASE_IDS = [spec.name for spec, _, _ in CASES]


def _mutated(make: Callable[[], FakeRun], mutate: Callable[[FakeRun], None]) -> FakeRun:
    run = make()
    run.start = copy.deepcopy(run.start)
    run.streams = copy.deepcopy(run.streams)
    mutate(run)
    return run


# --- Helpers -------------------------------------------------------------------------


def test_require_plan_name():
    run = FakeRun({"plan_name": "a"}, {})
    specs._require_plan_name(run, "a", "b")
    with pytest.raises(SpecValidationError, match="plan_name is 'a'"):
        specs._require_plan_name(run, "b")
    with pytest.raises(SpecValidationError, match="plan_name is None"):
        specs._require_plan_name(FakeRun({}, {}), "a")


def test_require_start_keys():
    run = FakeRun({"a": 1, "b": None}, {})
    specs._require_start_keys(run, "a", "b")
    with pytest.raises(SpecValidationError, match=r"missing \['c', 'd'\]"):
        specs._require_start_keys(run, "a", "c", "d")


@pytest.mark.parametrize(
    ("value", "count", "match"),
    [
        (None, None, "must be a list of names"),
        ("det", None, "must be a list of names"),
        (["det", 1], None, "must be a list of names"),
        (["a", "b"], 1, r"Expected 1 name\(s\)"),
        ([], 1, r"Expected 1 name\(s\)"),
    ],
)
def test_names_invalid(value: Any, count: int | None, match: str):
    with pytest.raises(SpecValidationError, match=match):
        specs._names(FakeRun({"detectors": value}, {}), "detectors", count=count)


def test_names_valid():
    run = FakeRun({"detectors": ["a", "b"]}, {})
    assert specs._names(run, "detectors") == ["a", "b"]
    assert specs._names(run, "detectors", count=2) == ["a", "b"]
    assert specs._names(FakeRun({"motors": []}, {}), "motors") == []


def test_require_stream():
    run = FakeRun({}, {"primary": ["a", "b"]})
    specs._require_stream(run, "primary")
    specs._require_stream(run, "primary", "a", "b")
    with pytest.raises(SpecValidationError, match="no 'baseline' stream"):
        specs._require_stream(run, "baseline")
    with pytest.raises(SpecValidationError, match=r"'primary' is missing \['c'\]"):
        specs._require_stream(run, "primary", "a", "c")


def test_require_optional_str():
    specs._require_optional_str(FakeRun({}, {}), "key")
    specs._require_optional_str(FakeRun({"key": "uid"}, {}), "key")
    with pytest.raises(SpecValidationError, match="'key' must be a string"):
        specs._require_optional_str(FakeRun({"key": 1}, {}), "key")


# --- Every validator ------------------------------------------------------------------


@pytest.mark.parametrize(("spec", "validator", "make"), CASES, ids=CASE_IDS)
def test_valid_run_passes(spec: Spec, validator: Callable[..., None], make: Callable[[], FakeRun]):
    validator(make())
    validator(make(), spec.version)
    validate_run(make(), spec)  # type: ignore[arg-type]


@pytest.mark.parametrize(("spec", "validator", "make"), CASES, ids=CASE_IDS)
def test_wrong_plan_name_fails(spec: Spec, validator: Callable[..., None], make: Callable[[], FakeRun]):
    run = _mutated(make, lambda r: r.start.update(plan_name="count"))
    with pytest.raises(SpecValidationError, match="plan_name is 'count'"):
        validator(run)


@pytest.mark.parametrize(
    ("spec", "validator", "make"),
    [case for case in CASES if case[0] is not TOMO_DARK_FLAT_V1],
    ids=[i for i in CASE_IDS if i != TOMO_DARK_FLAT_V1.name],
)
def test_missing_primary_stream_fails(
    spec: Spec, validator: Callable[..., None], make: Callable[[], FakeRun]
):
    def drop_primary(run: FakeRun):
        for name in [n for n in run.streams if n.startswith("primary")]:
            del run.streams[name]

    with pytest.raises(SpecValidationError, match="primary"):
        validator(_mutated(make, drop_primary))


@pytest.mark.parametrize(
    ("spec", "validator", "make"),
    [case for case in CASES if case[0] not in (EDXD_CALIBRATION_V1, TOMO_DARK_FLAT_V1)],
    ids=[i for i in CASE_IDS if i not in (EDXD_CALIBRATION_V1.name, TOMO_DARK_FLAT_V1.name)],
)
def test_missing_detectors_key_fails(
    spec: Spec, validator: Callable[..., None], make: Callable[[], FakeRun]
):
    run = _mutated(make, lambda r: r.start.pop("detectors"))
    with pytest.raises(SpecValidationError, match="'detectors' must be a list"):
        validator(run)


@pytest.mark.parametrize(
    ("spec", "validator", "make"),
    [case for case in CASES if case[0] is not TOMO_DARK_FLAT_V1],
    ids=[i for i in CASE_IDS if i != TOMO_DARK_FLAT_V1.name],
)
def test_missing_detector_field_fails(
    spec: Spec, validator: Callable[..., None], make: Callable[[], FakeRun]
):
    run = make()
    det = run.start.get("detectors", ["germ"])[0]

    def drop_det(r: FakeRun):
        for fields in r.streams.values():
            if det in fields:
                fields.remove(det)

    with pytest.raises(SpecValidationError, match=rf"missing \['{det}'"):
        validator(_mutated(make, drop_det))


# --- Spec-specific rules --------------------------------------------------------------


@pytest.mark.parametrize("key", ["exposure_time", "num_dark_images", "num_flat_images", "x_offset"])
def test_tomo_dark_flat_missing_start_key(key: str):
    run = _mutated(_tomo_dark_flat, lambda r: r.start.pop(key))
    with pytest.raises(SpecValidationError, match=rf"missing \['{key}'\]"):
        validate_tomo_dark_flat(run)


@pytest.mark.parametrize("stream", ["dark", "flat"])
def test_tomo_dark_flat_requires_both_streams(stream: str):
    run = _mutated(_tomo_dark_flat, lambda r: r.streams.pop(stream))
    with pytest.raises(SpecValidationError, match=f"no '{stream}' stream"):
        validate_tomo_dark_flat(run)


def test_tomo_dark_flat_flat_must_cover_dark_detectors():
    run = _mutated(_tomo_dark_flat, lambda r: r.streams["flat"].remove("kinetix1"))
    with pytest.raises(SpecValidationError, match=r"'flat' is missing \['kinetix1'\]"):
        validate_tomo_dark_flat(run)


def test_tomo_dark_flat_needs_a_detector():
    run = _mutated(_tomo_dark_flat, lambda r: r.streams.update(dark=["ts_kinetix1", "seq_num"]))
    with pytest.raises(SpecValidationError, match="no detector fields"):
        validate_tomo_dark_flat(run)


@pytest.mark.parametrize(
    "key",
    [
        "num_points",
        "exposure_time",
        "acquire_period",
        "time_based",
        "start_position",
        "stop_position",
        "images_to_average",
    ],
)
def test_tomo_flyscan_missing_start_key(key: str):
    run = _mutated(_tomo_flyscan, lambda r: r.start.pop(key))
    with pytest.raises(SpecValidationError, match=rf"missing \['{key}'\]"):
        validate_tomo_flyscan(run)


def test_tomo_flyscan_missing_rotation_angle():
    run = _mutated(_tomo_flyscan, lambda r: r.streams["primary"].remove("rotation_angle"))
    with pytest.raises(SpecValidationError, match=r"missing \['rotation_angle'\]"):
        validate_tomo_flyscan(run)


def test_tomo_flyscan_dark_flat_link_is_optional():
    def drop_link(r: FakeRun):
        r.start.pop("last_dark_flat_uid")
        r.start.pop("last_dark_flat_scan_id")

    validate_tomo_flyscan(_mutated(_tomo_flyscan, drop_link))
    run = _mutated(_tomo_flyscan, lambda r: r.start.update(last_dark_flat_uid=3))
    with pytest.raises(SpecValidationError, match="'last_dark_flat_uid' must be a string"):
        validate_tomo_flyscan(run)


def test_tomo_alignment_scan_flat_is_optional():
    validate_tomo_alignment_scan(_mutated(_tomo_alignment_scan, lambda r: r.streams.pop("flat")))
    run = _mutated(_tomo_alignment_scan, lambda r: r.streams["flat"].clear())
    with pytest.raises(SpecValidationError, match=r"'flat' is missing \['kinetix1'\]"):
        validate_tomo_alignment_scan(run)


@pytest.mark.parametrize("key", ["detectors", "motors"])
def test_tomo_alignment_scan_requires_exactly_one(key: str):
    run = _mutated(_tomo_alignment_scan, lambda r: r.start[key].append("extra"))
    with pytest.raises(SpecValidationError, match=rf"Expected 1 name\(s\) in '{key}'"):
        validate_tomo_alignment_scan(run)


def test_tomo_alignment_scan_requires_description_and_motor_field():
    run = _mutated(_tomo_alignment_scan, lambda r: r.start.pop("description"))
    with pytest.raises(SpecValidationError, match=r"missing \['description'\]"):
        validate_tomo_alignment_scan(run)
    run = _mutated(_tomo_alignment_scan, lambda r: r.streams["primary"].remove("rot"))
    with pytest.raises(SpecValidationError, match=r"missing \['rot'\]"):
        validate_tomo_alignment_scan(run)


def test_radiograph_checks_every_detector():
    run = _mutated(_radiograph, lambda r: r.streams["primary"].remove("kinetix2"))
    with pytest.raises(SpecValidationError, match=r"missing \['kinetix2'\]"):
        validate_radiograph(run)


@pytest.mark.parametrize(
    "plan_name", ["edxd_count", "edxd_scan", "edxd_grid_scan", "edxd_custom_pos_list_grid"]
)
def test_edxd_scan_plan_names(plan_name: str):
    validate_edxd_scan(_mutated(_edxd_scan, lambda r: r.start.update(plan_name=plan_name)))


@pytest.mark.parametrize("motors", [None, [], ["x"], ["x", "y"]])
def test_edxd_scan_zero_to_two_motors(motors: list[str] | None):
    def set_motors(r: FakeRun):
        if motors is None:
            r.start.pop("motors")
        else:
            r.start["motors"] = motors

    validate_edxd_scan(_mutated(_edxd_scan, set_motors))


def test_edxd_scan_too_many_motors():
    def three_motors(r: FakeRun):
        r.start["motors"] = ["x", "y", "z"]
        r.streams["primary"].append("z")

    with pytest.raises(SpecValidationError, match="at most 2 motors"):
        validate_edxd_scan(_mutated(_edxd_scan, three_motors))


@pytest.mark.parametrize("field", ["germ-stats1-total", "x"])
def test_edxd_scan_missing_field(field: str):
    run = _mutated(_edxd_scan, lambda r: r.streams["primary"].remove(field))
    with pytest.raises(SpecValidationError, match=rf"missing \['{field}'\]"):
        validate_edxd_scan(run)


def test_edxd_calibration_needs_a_stats_field():
    run = _mutated(_edxd_calibration, lambda r: r.streams["primary"].remove("germ-stats1-total"))
    with pytest.raises(SpecValidationError, match="no '<det>-stats1-total' field"):
        validate_edxd_calibration(run)


def test_edxd_calibration_ignores_detectors_key():
    validate_edxd_calibration(_mutated(_edxd_calibration, lambda r: r.start.update(detectors=7)))


def test_xrd_calibration_rules():
    run = _mutated(_xrd_calibration, lambda r: r.start.pop("description"))
    with pytest.raises(SpecValidationError, match=r"missing \['description'\]"):
        validate_xrd_calibration(run)
    run = _mutated(_xrd_calibration, lambda r: r.start["motors"].append("extra"))
    with pytest.raises(SpecValidationError, match=r"Expected 1 name\(s\) in 'motors'"):
        validate_xrd_calibration(run)
    run = _mutated(_xrd_calibration, lambda r: r.streams["primary"].remove("det_z"))
    with pytest.raises(SpecValidationError, match=r"missing \['det_z'\]"):
        validate_xrd_calibration(run)


@pytest.mark.parametrize("field", ["screen-stats1-mean_value", "pitch2"])
def test_energy_auto_tune_missing_field(field: str):
    run = _mutated(_energy_auto_tune, lambda r: r.streams["primary"].remove(field))
    with pytest.raises(SpecValidationError, match=rf"missing \['{field}'\]"):
        validate_energy_auto_tune(run)


def test_energy_auto_tune_requires_one_motor():
    run = _mutated(_energy_auto_tune, lambda r: r.start["motors"].append("extra"))
    with pytest.raises(SpecValidationError, match=r"Expected 1 name\(s\) in 'motors'"):
        validate_energy_auto_tune(run)


# --- validate_run ---------------------------------------------------------------------


def test_spec_validation_error_is_value_error():
    assert issubclass(SpecValidationError, ValueError)


def test_every_spec_has_a_validator():
    assert set(specs._VALIDATORS) == {spec.name for spec, _, _ in CASES}


def test_validate_run_unknown_spec():
    with pytest.raises(KeyError, match="No validator for spec 'Unknown'"):
        validate_run(FakeRun({}, {}), Spec("Unknown", version="1"))  # type: ignore[arg-type]


def test_validate_run_dispatches_by_name_and_passes_version(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[Any, str | None]] = []
    monkeypatch.setitem(
        specs._VALIDATORS, RADIOGRAPH_V1.name, lambda run, version: calls.append((run, version))
    )
    run = FakeRun({}, {})
    validate_run(run, Spec(RADIOGRAPH_V1.name, version="2"))  # type: ignore[arg-type]
    assert calls == [(run, "2")]


def test_validate_run_propagates_failure():
    run = _mutated(_radiograph, lambda r: r.start.update(plan_name="count"))
    with pytest.raises(SpecValidationError):
        validate_run(run, RADIOGRAPH_V1)  # type: ignore[arg-type]
