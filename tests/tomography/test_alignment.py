import numpy as np
import pytest

from hextools.tomography.alignment import identify_sign_tilt_angle


@pytest.mark.parametrize(
    "x, y, expected_sign",
    [
        # Horizontal arc curving below -> positive sign
        (
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([0.0, -1.0, -2.0, -1.0, 0.0]),
            1,
        ),
        # Horizontal arc curving above -> negative sign
        (
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([0.0, 1.0, 2.0, 1.0, 0.0]),
            -1,
        ),
        # Tilted line with points sagging below -> positive sign
        (
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([0.0, 0.0, 0.0, 1.5, 2.0]),
            1,
        ),
        # Tilted line with points bulging above -> negative sign
        (
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([0.0, 1.0, 2.0, 2.5, 2.0]),
            -1,
        ),
        # Upper semicircle -> negative sign
        (
            np.cos(np.linspace(0, np.pi, 7)),
            np.sin(np.linspace(0, np.pi, 7)),
            -1,
        ),
        # Lower semicircle -> positive sign
        (
            np.cos(np.linspace(0, np.pi, 7)),
            -np.sin(np.linspace(0, np.pi, 7)),
            1,
        ),
    ],
)
def test_identify_sign_tilt_angle(x, y, expected_sign):
    assert identify_sign_tilt_angle(x, y) == expected_sign
