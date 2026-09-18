"""Small, dependency-free preprocessing utilities for point registration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["VoxelDownsampleResult", "voxel_downsample"]


@dataclass(frozen=True)
class VoxelDownsampleResult:
    """A point cloud reduced to one centroid per voxel.

    ``attributes`` are reduced with the same voxel membership as ``points``.
    This is important for colored registration: a point's color must remain
    paired with its spatial representative after downsampling.
    """

    points: np.ndarray
    attributes: tuple[np.ndarray, ...]
    representative_indices: np.ndarray
    voxel_counts: np.ndarray


def _reduce_mean(values: np.ndarray, order: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    sorted_values = np.asarray(values)[order]
    sums = np.add.reduceat(sorted_values.astype(np.float64, copy=False), starts, axis=0)
    shape = (len(counts),) + (1,) * (sorted_values.ndim - 1)
    means = sums / counts.reshape(shape)
    if np.issubdtype(sorted_values.dtype, np.integer):
        info = np.iinfo(sorted_values.dtype)
        means = np.rint(np.clip(means, info.min, info.max)).astype(sorted_values.dtype)
    return means


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
    *,
    attributes: Sequence[np.ndarray] = (),
) -> VoxelDownsampleResult:
    """Downsample points by averaging each non-empty voxel.

    Voxel keys are computed in the input coordinate frame using
    ``floor(points / voxel_size)``.  Groups are sorted lexicographically, so
    the result is deterministic and independent of hash-table iteration order.
    Every attribute must have the same first dimension as ``points``; numeric
    attributes are averaged and integer attributes are rounded back to their
    original dtype.
    """

    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] == 0:
        raise ValueError("points must have shape (N, D) with D >= 1")
    if not np.isfinite(point_array).all():
        raise ValueError("points must contain only finite values")
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and positive")

    attribute_arrays = tuple(np.asarray(attribute) for attribute in attributes)
    for attribute in attribute_arrays:
        if attribute.ndim == 0 or attribute.shape[0] != len(point_array):
            raise ValueError("every attribute must have the same first dimension as points")
        if not np.issubdtype(attribute.dtype, np.number) or np.iscomplexobj(attribute):
            raise ValueError("attributes must be real numbers")
        if not np.isfinite(attribute).all():
            raise ValueError("attributes must contain only finite values")

    if len(point_array) == 0:
        empty_attributes = tuple(attribute[:0].copy() for attribute in attribute_arrays)
        return VoxelDownsampleResult(
            point_array.copy(), empty_attributes, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        )

    extent = float(max(-point_array.min(), point_array.max())) / voxel_size
    if not (extent < 2.0 ** 63):
        raise ValueError("points / voxel_size exceeds the representable voxel index range")
    voxel_keys = np.floor(point_array / voxel_size).astype(np.int64)
    sort_keys = tuple(
        voxel_keys[:, axis] for axis in range(voxel_keys.shape[1] - 1, -1, -1)
    )
    order = np.lexsort(sort_keys)
    sorted_keys = voxel_keys[order]
    starts_mask = np.empty(len(order), dtype=bool)
    starts_mask[0] = True
    starts_mask[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    starts = np.flatnonzero(starts_mask)
    ends = np.r_[starts[1:], len(order)]
    counts = (ends - starts).astype(np.int64)

    reduced_points = _reduce_mean(point_array, order, starts, counts)
    reduced_attributes = tuple(
        _reduce_mean(attribute, order, starts, counts) for attribute in attribute_arrays
    )
    return VoxelDownsampleResult(
        points=reduced_points,
        attributes=reduced_attributes,
        representative_indices=order[starts].astype(np.int64, copy=False),
        voxel_counts=counts,
    )
