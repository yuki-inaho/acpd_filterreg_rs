from __future__ import annotations

import numpy as np
import pytest

from acpd_filterreg.preprocessing import voxel_downsample


def test_voxel_downsample_keeps_geometry_and_attributes_paired() -> None:
    points = np.array(
        [[0.001, 0.001, 0.001], [0.004, 0.002, 0.001], [0.006, 0.001, 0.001]],
        dtype=np.float64,
    )
    colors = np.array([[10, 20, 30], [20, 30, 40], [100, 110, 120]], dtype=np.uint8)
    uv = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)

    result = voxel_downsample(points, 0.005, attributes=(colors, uv))

    assert result.points.shape == (2, 3)
    assert result.voxel_counts.tolist() == [2, 1]
    np.testing.assert_allclose(result.points[0], [0.0025, 0.0015, 0.001])
    np.testing.assert_array_equal(result.attributes[0][0], [15, 25, 35])
    np.testing.assert_allclose(result.attributes[1][0], [2.0, 3.0])
    assert result.attributes[0].dtype == np.uint8


def test_voxel_downsample_is_deterministic_for_negative_coordinates() -> None:
    points = np.array([[-0.001, 0.0], [-0.004, 0.0], [0.006, 0.0]], dtype=np.float64)
    first = voxel_downsample(points, 0.005)
    second = voxel_downsample(points, 0.005)

    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(first.representative_indices, second.representative_indices)


@pytest.mark.parametrize("bad_size", [0.0, -0.005, np.nan, np.inf])
def test_voxel_downsample_rejects_invalid_size(bad_size: float) -> None:
    with pytest.raises(ValueError, match="voxel_size"):
        voxel_downsample(np.zeros((1, 3)), bad_size)
