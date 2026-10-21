"""Averaging PandA captures down to one value per projection for the .nxs export."""

import numpy as np
import pytest

from hextools.exporters.utils import average_per_projection


def test_averages_each_projections_captures():
    # 19 projections x 5 captures, as Tiled shapes a PandA dataset with
    # collections_per_event=5.
    centres = np.linspace(0, 180, 19)
    captures = centres[:, None] + np.array([-4, -2, 0, 2, 4])
    np.testing.assert_allclose(average_per_projection(captures, 5), centres)


def test_accepts_captures_first():
    centres = np.linspace(0, 180, 19)
    captures = (centres[:, None] + np.array([-4, -2, 0, 2, 4])).T
    np.testing.assert_allclose(average_per_projection(captures, 5), centres)


def test_leaves_one_capture_per_projection_alone():
    angles = np.linspace(0, 180, 19)
    np.testing.assert_array_equal(average_per_projection(angles, 1), angles)


def test_refuses_a_shape_that_does_not_match():
    with pytest.raises(ValueError, match="images_to_average"):
        average_per_projection(np.zeros((19, 4)), 5)
