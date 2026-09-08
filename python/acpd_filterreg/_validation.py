"""Boundary validation, shared by both independent native engines."""
from __future__ import annotations
from numbers import Integral, Real
from typing import Any
import numpy as np


def boolean(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a Python bool")
    return value


def integer(name: str, value: Any, lower: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a bool")
    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return result


def real(name: str, value: Any, *, positive: bool = True) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, not a bool")
    result = float(value)
    if not np.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def probability(value: Any) -> float:
    result = real("w", value, positive=False)
    if result >= 1:
        raise ValueError("w must be in [0, 1)")
    return result


def array(name: str, value: Any, ndim: int, *, copy: bool) -> np.ndarray:
    boolean("copy", copy)
    if copy:
        if np.iscomplexobj(value):
            raise TypeError(f"{name} must not be complex")
        result = np.array(value, dtype=np.float64, order="C", copy=True)
    else:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array with copy=False; use copy=True for conversion")
        result = value
        if result.dtype != np.float64:
            raise TypeError(f"{name} must have dtype float64 with copy=False")
        if not result.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous with copy=False")
    if not result.flags.aligned:
        raise ValueError(f"{name} must be aligned for float64; use copy=True")
    if result.ndim != ndim or any(size == 0 for size in result.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def points(name: str, value: Any, *, copy: bool) -> np.ndarray:
    result = array(name, value, 2, copy=copy)
    if result.shape[1] not in (2, 3):
        raise ValueError(f"{name} must have shape (n, 2) or (n, 3)")
    return result


def backend(value: str) -> str:
    if value not in ("direct", "permutohedral", "permutohedral_noblur", "probreg", "fgt"):
        raise ValueError("unknown Gaussian backend; no automatic fallback is provided")
    return value
