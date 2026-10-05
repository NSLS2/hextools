import pytest

from hextools.detectors.utils import calculate_scan_time


def test_neither_velocity_nor_distance_is_allowed():
    result = calculate_scan_time(num_images=10, exposure_time=0.1)
    assert result == pytest.approx(9 * 0.105 + 0.1)


def test_both_velocity_and_distance_is_allowed():
    result = calculate_scan_time(
        num_images=2,
        exposure_time=0.1,
        max_velocity=2.0,
        travel_distance=100.0,
    )
    assert result == pytest.approx(50.0)


@pytest.mark.parametrize(
    "max_velocity, travel_distance",
    [(2.0, None), (None, 100.0)],
)
def test_only_one_of_velocity_or_distance_raises(max_velocity, travel_distance):
    with pytest.raises(ValueError, match="together or not at all"):
        calculate_scan_time(
            num_images=10,
            exposure_time=0.1,
            max_velocity=max_velocity,
            travel_distance=travel_distance,
        )
