"""Averaging PandA captures down to one value per projection for the .nxs export."""

import numpy as np
import pytest

from hextools.exporters.utils import average_per_projection

OFFSETS = np.array([-4, -2, 0, 2, 4])


def test_averages_each_projections_captures():
    # 19 projections x 5 captures, as Tiled shapes a PandA dataset with
    # collections_per_event=5 (events first).
    centres = np.linspace(0, 180, 19)
    captures = centres[:, None] + OFFSETS
    np.testing.assert_allclose(average_per_projection(captures, 19, 5), centres)


def test_square_capture_averages_over_images():
    centres = np.linspace(0, 180, 5)
    captures = centres[:, None] + OFFSETS
    np.testing.assert_allclose(average_per_projection(captures, 5, 5), centres)


def test_refuses_captures_first():
    captures = (np.linspace(0, 180, 19)[:, None] + OFFSETS).T
    with pytest.raises(ValueError, match="images_to_average"):
        average_per_projection(captures, 19, 5)


def test_leaves_one_capture_per_projection_alone():
    angles = np.linspace(0, 180, 19)
    np.testing.assert_array_equal(average_per_projection(angles, 19, 1), angles)


def test_refuses_a_shape_that_does_not_match():
    with pytest.raises(ValueError, match="images_to_average"):
        average_per_projection(np.zeros((19, 4)), 19, 5)


def test_refuses_a_flattened_capture_when_averaging():
    with pytest.raises(ValueError, match="images_to_average"):
        average_per_projection(np.zeros(95), 19, 5)
