"""Evaluation of saved maps, not a registration algorithm or native fallback."""
from __future__ import annotations
from functools import lru_cache
from math import factorial
import numpy as np


@lru_cache(maxsize=32)
def exponents(dimension: int, degree: int) -> tuple[tuple[int, ...], ...]:
    rows = []
    for order in range(degree + 1):
        for a in range(order, -1, -1):
            if dimension == 2:
                rows.append((a, order - a))
            else:
                rows.extend((a, b, order - a - b) for b in range(order - a, -1, -1))
    return tuple(rows)


def basis(points: np.ndarray, degree: int) -> np.ndarray:
    output = np.ones((len(points), len(exponents(points.shape[1], degree))))
    with np.errstate(over="raise", invalid="raise"):
        for k, alpha in enumerate(exponents(points.shape[1], degree)):
            for axis, power in enumerate(alpha):
                if power:
                    output[:, k] *= points[:, axis] ** power / factorial(power)
    return output


def basis_derivative(points: np.ndarray, degree: int) -> np.ndarray:
    powers = exponents(points.shape[1], degree)
    output = np.zeros((len(points), len(powers), points.shape[1]))
    with np.errstate(over="raise", invalid="raise"):
        for k, alpha in enumerate(powers):
            for axis in range(points.shape[1]):
                if not alpha[axis]:
                    continue
                value = np.ones(len(points))
                for other, exponent in enumerate(alpha):
                    power = exponent - int(other == axis)
                    if power:
                        value *= points[:, other] ** power / factorial(power)
                output[:, k, axis] = value
    return output
