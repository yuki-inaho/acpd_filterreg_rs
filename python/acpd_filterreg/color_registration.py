"""Colored 3D FilterReg + Analytic-CPD with CPU-direct and CUDA FilterReg kernels.

This module is a *colored extension*. The rigid fit and result assembly are in
NumPy; ``cuda_color`` delegates the FilterReg Gaussian-sum E-step to the native
CUDA engine without forming a correspondence matrix. FilterReg/CPD can in
general incorporate feature attributes; the specific
combination implemented here -- a CIELAB (or generic continuous-attribute) Gaussian
gate added to the kernel together with the analytic Taylor mapping -- is a custom
extension of this repository, not a claim about the native engine's capabilities.
Color is treated as an *observed attribute* carried by both clouds and is NOT
deformed by the geometric transform; it only gates correspondences in the kernel.

Algorithm (faithful to the repo theory / native reference, color added only in
the kernel):

* Position-only normalization: fixed centroid + fixed RMS common scale
  (``docs/SOURCE_DIFFERENCES.md`` Analytic-CPD 8). Colors are NOT normalized.
* FilterReg row-direction posterior ``Q`` -> consistent-responsibility Kabsch
  rigid fit -> freeze pose (arXiv:1811.10136).
* CPD column-direction posterior ``P`` -> weighted factorial Taylor absolute
  mapping solved by SVD (minimum-norm, rank-truncated, no ridge), stored as the
  difference from the identity map, composed across degrees
  (arXiv:2605.00934v1 sec 3.2-3.6; ``rust/crates/acpd-core/src/analytic.rs``).

The two posterior normalizations (row vs column) and the outlier / variance
handling are documented in ``design_registration.md`` sec 5-7. ``color_weight=0``
reproduces the geometry-only computation exactly (see tests).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import factorial, log, pi
from typing import Any, Literal

import numpy as np

__all__ = [
    "AnalyticStep",
    "ColorNumericalError",
    "ColorRegistrationError",
    "ColorRegistrationResult",
    "Iteration",
    "PosteriorStats",
    "StageResult",
    "color_posterior_statistics",
    "register_color",
]

Method = Literal["rigid", "analytic", "nonrigid"]
ColorBackend = Literal["direct_cpu_color", "cuda_color"]
_TWO_PI_LN = log(2.0 * pi)


class ColorRegistrationError(ValueError):
    """Invalid input or configuration for the colored registration."""


class ColorNumericalError(ColorRegistrationError):
    """A numerically ill-posed state was reached; no silent fallback is used."""


# --------------------------------------------------------------------------- #
# Factorial-scaled multivariate Taylor basis (matches acpd_filterreg._mapping).
# Reimplemented here so the color backend is self-contained and never depends on
# the FGT-modified package __init__; a test cross-checks equivalence.
# --------------------------------------------------------------------------- #
def exponents(dimension: int, degree: int) -> tuple[tuple[int, ...], ...]:
    if dimension not in (2, 3) or not 0 <= degree <= 10:
        raise ColorRegistrationError("basis requires dimension 2/3 and degree 0..10")
    rows: list[tuple[int, ...]] = []
    for order in range(degree + 1):
        for a in range(order, -1, -1):
            if dimension == 2:
                rows.append((a, order - a))
            else:
                rows.extend((a, b, order - a - b) for b in range(order - a, -1, -1))
    return tuple(rows)


def basis(points: np.ndarray, degree: int) -> np.ndarray:
    exps = exponents(points.shape[1], degree)
    output = np.ones((len(points), len(exps)))
    with np.errstate(over="raise", invalid="raise"):
        for k, alpha in enumerate(exps):
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


# --------------------------------------------------------------------------- #
# Result dataclasses (field names mirror acpd_filterreg._result.RegistrationResult
# so downstream consumers/visualization can treat them uniformly).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Iteration:
    iteration: int
    degree: int
    active: int
    rank: int
    sigma2: float
    nll_before: float | None
    step_rms: float
    fit_rms: float


@dataclass(frozen=True)
class StageResult:
    history: tuple[Iteration, ...] = ()
    initial_sigma2: float = 0.0
    final_sigma2: float = 0.0
    converged: bool = False
    stop_reason: str = "not_run"
    best_iteration: int = 0

    @property
    def iterations(self) -> int:
        return len(self.history)


@dataclass(frozen=True)
class AnalyticStep:
    degree: int
    coefficients: np.ndarray


@dataclass(frozen=True)
class ColorRegistrationResult:
    rotation: np.ndarray
    translation: np.ndarray
    center: np.ndarray
    normalization_scale: float
    transformed: np.ndarray
    rigid_transformed: np.ndarray
    steps: tuple[AnalyticStep, ...]
    rigid_stage: StageResult
    analytic_stage: StageResult
    sigma2: float
    method: str
    color_sigma: float
    color_weight: float
    color_dim: int
    backend: str = "direct_cpu_color"
    engine: str = "python"

    # -- reproducible map re-application (same math as _result.RegistrationResult) --
    def transform(self, points: Any) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != len(self.translation):
            raise ColorRegistrationError("point dimension differs from the saved map")
        try:
            with np.errstate(over="raise", invalid="raise"):
                value = (p @ self.rotation.T + self.translation - self.center) / self.normalization_scale
                for step in self.steps:
                    value = value + basis(value, step.degree) @ step.coefficients
                result = value * self.normalization_scale + self.center
        except FloatingPointError as error:
            raise ColorNumericalError("map evaluation overflow; extrapolation is not bounded") from error
        if not np.isfinite(result).all():
            raise ColorNumericalError("map evaluation overflow; extrapolation is not bounded")
        return result

    def residual_displacement(self, points: Any) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        return self.transform(p) - (p @ self.rotation.T + self.translation)

    def displacement(self, points: Any) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        return self.transform(p) - p

    @property
    def converged(self) -> bool:
        stages = [s for s in (self.rigid_stage, self.analytic_stage) if s.iterations]
        return bool(stages) and all(s.converged for s in stages)

    @property
    def status(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "converged": self.converged,
            "rigid_stop_reason": self.rigid_stage.stop_reason,
            "analytic_stop_reason": self.analytic_stage.stop_reason,
            "rigid_iterations": self.rigid_stage.iterations,
            "analytic_iterations": self.analytic_stage.iterations,
            "final_degree": self.steps[-1].degree if self.steps else 0,
            # rank of the retained best iteration (history keeps every attempt; steps is
            # truncated to the best, and history[len(steps)-1] is that best iteration's row)
            "final_rank": self.analytic_stage.history[len(self.steps) - 1].rank if self.steps else 0,
            "sigma2": self.sigma2,
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly snapshot (arrays -> lists) for coordinator/V consumption."""
        def stage(s: StageResult) -> dict[str, Any]:
            return {
                "initial_sigma2": s.initial_sigma2,
                "final_sigma2": s.final_sigma2,
                "converged": s.converged,
                "stop_reason": s.stop_reason,
                "best_iteration": s.best_iteration,
                "history": [
                    {
                        "iteration": h.iteration, "degree": h.degree, "active": h.active,
                        "rank": h.rank, "sigma2": h.sigma2, "nll_before": h.nll_before,
                        "step_rms": h.step_rms, "fit_rms": h.fit_rms,
                    }
                    for h in s.history
                ],
            }

        return {
            "backend": self.backend, "engine": self.engine, "method": self.method,
            "rotation": self.rotation.tolist(), "translation": self.translation.tolist(),
            "center": self.center.tolist(), "normalization_scale": self.normalization_scale,
            "transformed": self.transformed.tolist(), "rigid_transformed": self.rigid_transformed.tolist(),
            "sigma2": self.sigma2,
            "color_sigma": self.color_sigma, "color_weight": self.color_weight, "color_dim": self.color_dim,
            "steps": [{"degree": s.degree, "coefficients": s.coefficients.tolist()} for s in self.steps],
            "rigid_stage": stage(self.rigid_stage), "analytic_stage": stage(self.analytic_stage),
            "status": self.status,
        }


# --------------------------------------------------------------------------- #
# Posterior statistics (colored kernel, both normalization directions).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PosteriorStats:
    rho: np.ndarray
    px: np.ndarray
    x2: np.ndarray
    mass: float
    nll: float


def color_posterior_statistics(
    y: np.ndarray, x: np.ndarray, sigma2: float, w: float, *,
    inverse: bool, color_logk: np.ndarray | None = None,
) -> PosteriorStats:
    """Direct, dense posterior moments for the colored Gaussian kernel.

    ``inverse=True`` is the FilterReg row direction (normalize per moving row over
    fixed columns); ``inverse=False`` is the CPD column direction (normalize per
    fixed column over moving rows). ``color_logk[i,j]`` is the additive color
    log-kernel ``-0.5*(color_weight/color_sigma^2)*||a_i-b_j||^2`` (moving x fixed);
    ``None`` means geometry only. Moments ``rho/px/x2`` are indexed by moving row.
    """
    if sigma2 <= 0 or not np.isfinite(sigma2):
        raise ColorRegistrationError("sigma2 must be finite and positive")
    if not (0.0 <= w < 1.0) or not np.isfinite(w):
        raise ColorRegistrationError("w must be in [0, 1)")
    m, d = y.shape
    n = x.shape[0]
    # geometric squared-distance matrix (moving x fixed)
    dist = (
        np.sum(y * y, axis=1)[:, None]
        - 2.0 * (y @ x.T)
        + np.sum(x * x, axis=1)[None, :]
    )
    if not np.isfinite(dist).all():
        raise ColorNumericalError("squared distance overflow")
    logk = -0.5 * dist / sigma2  # (m, n)
    if color_logk is not None:
        logk = logk + color_logk

    centers, queries = (n, m) if inverse else (m, n)
    normalizer = 0.5 * d * (_TWO_PI_LN + log(sigma2))
    logc = -np.inf if w == 0.0 else (
        normalizer + log(w) - np.log1p(-w) + log(centers / queries)
    )
    logfactor = np.log1p(-w) - log(centers) - normalizer

    if inverse:  # FilterReg: normalize each moving row over fixed columns
        maxlog = np.maximum(logk.max(axis=1), logc)  # (m,)
        if not np.isfinite(maxlog).all():
            raise ColorNumericalError("posterior has no representable support")
        ex = np.exp(logk - maxlog[:, None])
        denom = np.exp(logc - maxlog) + ex.sum(axis=1)  # (m,)
        prob = ex / denom[:, None]  # p(j|i), (m,n)
        nll = -float(np.sum(maxlog + np.log(denom) + logfactor))
    else:  # CPD: normalize each fixed column over moving rows
        maxlog = np.maximum(logk.max(axis=0), logc)  # (n,)
        if not np.isfinite(maxlog).all():
            raise ColorNumericalError("posterior has no representable support")
        ex = np.exp(logk - maxlog[None, :])
        denom = np.exp(logc - maxlog) + ex.sum(axis=0)  # (n,)
        prob = ex / denom[None, :]  # p(i|j), (m,n)
        nll = -float(np.sum(maxlog + np.log(denom) + logfactor))

    rho = prob.sum(axis=1)  # (m,)
    px = prob @ x  # (m,d)
    x2 = prob @ np.sum(x * x, axis=1)  # (m,)
    mass = float(rho.sum())
    if not (np.isfinite(mass) and np.isfinite(px).all() and np.isfinite(x2).all()):
        raise ColorNumericalError("non-finite posterior moments")
    return PosteriorStats(rho, px, x2, mass, nll)


def _cuda_color_posterior_statistics(
    y: np.ndarray,
    x: np.ndarray,
    sigma2: float,
    w: float,
    *,
    moving_colors: np.ndarray,
    fixed_colors: np.ndarray,
    color_precision: float,
    single_precision: bool,
) -> PosteriorStats:
    """FilterReg row posterior using one CUDA Gaussian-sum over XYZ+color.

    Geometry is whitened by ``sigma2`` and color by ``color_precision``. The
    returned moments still contain XYZ only, so color affects correspondence
    probabilities but is never rotated or translated as geometry.
    """
    from ._api import gaussian_sum
    from ._options import CudaOptions

    if sigma2 <= 0 or not np.isfinite(sigma2):
        raise ColorRegistrationError("sigma2 must be finite and positive")
    if not (0.0 <= w < 1.0) or not np.isfinite(w):
        raise ColorRegistrationError("w must be in [0, 1)")
    if color_precision > 0.0 and x.shape[1] + fixed_colors.shape[1] > 16:
        raise ColorRegistrationError("cuda_color supports at most 16 combined geometry/color features")

    geometry_scale = 1.0 / np.sqrt(sigma2)
    if color_precision == 0.0:
        sources = np.ascontiguousarray(x * geometry_scale, dtype=np.float64)
        queries = np.ascontiguousarray(y * geometry_scale, dtype=np.float64)
    else:
        color_scale = np.sqrt(color_precision)
        sources = np.ascontiguousarray(
            np.concatenate((x * geometry_scale, fixed_colors * color_scale), axis=1),
            dtype=np.float64,
        )
        queries = np.ascontiguousarray(
            np.concatenate((y * geometry_scale, moving_colors * color_scale), axis=1),
            dtype=np.float64,
        )
    values = np.ascontiguousarray(
        np.column_stack((np.ones(len(x)), x, np.sum(x * x, axis=1))),
        dtype=np.float64,
    )
    moments = gaussian_sum(
        sources,
        queries,
        values,
        sigma2=1.0,
        backend="cuda",
        cuda=CudaOptions(single_precision=single_precision),
    )

    m, d = y.shape
    n = len(x)
    normalizer = 0.5 * d * (_TWO_PI_LN + log(sigma2))
    outlier = 0.0 if w == 0.0 else np.exp(
        normalizer + log(w) - np.log1p(-w) + log(n / m)
    )
    denominator = moments[:, 0] + outlier
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise ColorNumericalError("CUDA posterior has no representable support")

    rho = moments[:, 0] / denominator
    px = moments[:, 1 : d + 1] / denominator[:, None]
    x2 = moments[:, d + 1] / denominator
    mass = float(rho.sum())
    logfactor = np.log1p(-w) - log(n) - normalizer
    nll = -float(np.sum(np.log(denominator) + logfactor))
    if not (np.isfinite(mass) and np.isfinite(px).all() and np.isfinite(x2).all()):
        raise ColorNumericalError("non-finite CUDA posterior moments")
    return PosteriorStats(rho, px, x2, mass, nll)


def _initial_variance(x: np.ndarray, y: np.ndarray) -> float:
    d = x.shape[1]
    sx = float(np.sum((x - x.mean(axis=0)) ** 2) / len(x))
    sy = float(np.sum((y - y.mean(axis=0)) ** 2) / len(y))
    result = (sx + sy + float(np.sum((x.mean(axis=0) - y.mean(axis=0)) ** 2))) / d
    if not np.isfinite(result):
        raise ColorNumericalError("initial variance overflow")
    return result


def _variance_from_statistics(y: np.ndarray, stats: PosteriorStats, floor: float) -> float:
    d = y.shape[1]
    if not np.isfinite(stats.mass) or stats.mass <= 1e-14:
        raise ColorNumericalError("invalid or insufficient posterior mass")
    a = stats.x2
    b = 2.0 * np.sum(y * stats.px, axis=1)
    c = stats.rho * np.sum(y * y, axis=1)
    residual = float(np.sum(a - b + c))
    magnitude = float(np.sum(np.abs(a) + np.abs(b) + np.abs(c)))
    if not np.isfinite(residual) or residual < -1e-10 * max(magnitude, 1.0):
        raise ColorNumericalError("invalid second-moment residual")
    return max(floor, max(residual, 0.0) / (d * stats.mass))


# --------------------------------------------------------------------------- #
# Rigid fit (consistent-responsibility Kabsch) and analytic fit (SVD abs. map).
# --------------------------------------------------------------------------- #
def _fit_rigid_kabsch(y: np.ndarray, stats: PosteriorStats) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
    d = y.shape[1]
    active = np.where(stats.rho > 1e-12)[0]
    if len(active) < d:
        raise ColorNumericalError("insufficient effective FilterReg correspondences")
    rho_a = stats.rho[active]
    mass = float(rho_a.sum())
    ya = y[active]
    pxa = stats.px[active]
    cy = (rho_a[:, None] * ya).sum(axis=0) / mass          # responsibility-weighted centroid of y
    cz = pxa.sum(axis=0) / mass                            # weighted centroid of soft targets
    h = (ya - cy).T @ (pxa - rho_a[:, None] * cz)          # H = sum_i rho_i (y_i-cy)(z_i-cz)^T
    if not np.isfinite(h).all():
        raise ColorNumericalError("non-finite rigid cross-covariance")
    u, s, vt = np.linalg.svd(h)
    rank = int((s > 1e-12 * s.max()).sum()) if s.size and s.max() > 0 else 0
    if rank < d - 1:
        raise ColorNumericalError("rigid pose is not identifiable")
    sign = np.eye(d)
    if np.linalg.det(vt.T @ u.T) < 0:
        sign[d - 1, d - 1] = -1.0
    rotation = vt.T @ sign @ u.T
    translation = cz - rotation @ cy
    nxt = y @ rotation.T + translation
    z_active = stats.px[active] / rho_a[:, None]
    err = float(np.sum(rho_a * np.sum((nxt[active] - z_active) ** 2, axis=1)))
    fit_rms = float(np.sqrt(err / mass))
    return rotation, translation, nxt, rank, len(active), fit_rms


def _fit_analytic(
    y: np.ndarray, stats: PosteriorStats, degree: int, *,
    min_degree: int, max_degree: int, rank_tolerance: float, min_mass: float,
) -> tuple[AnalyticStep, np.ndarray, int, int, float]:
    d = y.shape[1]
    if degree < min_degree or degree > max_degree:
        raise ColorRegistrationError("requested degree outside options")
    if np.any(stats.rho < 0.0):
        raise ColorRegistrationError("negative analytic fitting weight")
    active = np.where(stats.rho > min_mass)[0]
    n = len(active)
    while degree > min_degree and len(exponents(d, degree)) > n:
        degree -= 1
    phi = basis(y, degree)
    k = phi.shape[1]
    if n < k:
        raise ColorNumericalError("insufficient active rows for the minimum analytic degree")
    sqrt_rho = np.sqrt(stats.rho[active])
    design = sqrt_rho[:, None] * phi[active]              # (n,k)
    targets = stats.px[active] / sqrt_rho[:, None]        # (n,d)
    if not (np.isfinite(design).all() and np.isfinite(targets).all()):
        raise ColorNumericalError("non-finite weighted Taylor design/targets")
    # Minimum-norm ABSOLUTE mapping via rank-truncated SVD (no ridge/projection/cap).
    u, sv, vt = np.linalg.svd(design, full_matrices=False)
    smax = sv.max() if sv.size else 0.0
    threshold = rank_tolerance * smax
    rank = int((sv > threshold).sum())
    sinv = np.where(sv > threshold, 1.0 / np.where(sv > 0, sv, 1.0), 0.0)
    absolute = (vt.T * sinv) @ (u.T @ targets)            # (k,d)
    nxt = phi @ absolute
    if not np.isfinite(nxt).all():
        raise ColorNumericalError("analytic M-step overflow")
    coefficients = absolute.copy()
    for a in range(d):
        coefficients[1 + a, a] -= 1.0                     # store as difference from identity
    mass = float(stats.rho[active].sum())
    z = stats.px[active] / stats.rho[active][:, None]
    sse = float(np.sum(stats.rho[active] * np.sum((z - nxt[active]) ** 2, axis=1)))
    fit_rms = float(np.sqrt(sse / mass))
    return AnalyticStep(degree, coefficients), nxt, n, rank, fit_rms


def _degree_schedule(iterations: int, low: int, high: int) -> list[int]:
    if iterations == 0 or low < 1 or high < low or high > 10:
        raise ColorRegistrationError("invalid degree schedule")
    count = high - low + 1
    unit = count * (count + 1) // 2
    lengths = [(count - k) * (iterations // unit) for k in range(count)]
    remaining = iterations % unit
    for prefix in range(count, 0, -1):
        take = min(prefix, remaining)
        for idx in range(take):
            lengths[idx] += 1
        remaining -= take
        if remaining == 0:
            break
    out: list[int] = []
    for k, length in enumerate(lengths):
        out.extend([low + k] * length)
    return out


def _advance_degree(schedule: list[int], at: int) -> int:
    degree = schedule[at]
    nxt = at
    while nxt < len(schedule) and schedule[nxt] <= degree:
        nxt += 1
    return nxt


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #
def register_color(
    fixed_xyz: Any, moving_xyz: Any, fixed_colors: Any, moving_colors: Any, *,
    color_sigma: float, color_weight: float = 1.0, method: Method = "nonrigid",
    w: float = 0.1, sigma2: float | None = None, max_iterations: int = 60,
    tolerance: float = 1e-7, min_sigma2: float = 1e-8, update_sigma2: bool = True,
    analytic_max_iterations: int = 220, min_degree: int = 1, max_degree: int = 2,
    analytic_tolerance: float = 1e-7, analytic_min_sigma2: float = 1e-12,
    rank_tolerance: float = 1e-12, min_mass: float = 1e-12,
    stable_patience: int = 5, no_improve_patience: int = 8, min_iterations: int = 6,
    improvement_relative: float = 1e-6, rebound_relative: float = 1e-3,
    divergence_radius: float = 100.0, initial_rotation: Any = None,
    initial_translation: Any = None, backend: ColorBackend = "direct_cpu_color",
    cuda_single_precision: bool = False,
) -> ColorRegistrationResult:
    """Register ``moving -> fixed`` in 2D/3D using color-augmented FilterReg + Analytic-CPD.

    See ``design_registration.md`` for the full contract. ``fixed_xyz``/``moving_xyz``
    are metric point clouds ``(n, d)``; ``fixed_colors``/``moving_colors`` are
    per-point continuous attributes ``(n, C)`` (standardized CIELAB recommended),
    treated as fixed observation attributes. ``color_weight=0`` -> geometry only.
    """
    if method not in ("rigid", "analytic", "nonrigid"):
        raise ColorRegistrationError("method must be 'rigid', 'analytic' or 'nonrigid'")
    if backend not in ("direct_cpu_color", "cuda_color"):
        raise ColorRegistrationError("backend must be 'direct_cpu_color' or 'cuda_color'")
    if backend == "cuda_color" and method != "rigid":
        raise ColorRegistrationError("cuda_color currently accelerates FilterReg rigid mode only")
    if type(cuda_single_precision) is not bool:
        raise ColorRegistrationError("cuda_single_precision must be a Python bool")
    fixed = _as_points("fixed_xyz", fixed_xyz)
    moving = _as_points("moving_xyz", moving_xyz)
    d = fixed.shape[1]
    if moving.shape[1] != d:
        raise ColorRegistrationError("fixed and moving dimensions must match")
    if min(len(fixed), len(moving)) < d + 1:
        raise ColorRegistrationError("registration requires at least d+1 points in each cloud")
    fc = _as_colors("fixed_colors", fixed_colors, len(fixed))
    mc = _as_colors("moving_colors", moving_colors, len(moving))
    if fc.shape[1] != mc.shape[1]:
        raise ColorRegistrationError("fixed and moving colors must have the same dimension")
    color_dim = fc.shape[1]
    if not np.isfinite(color_sigma) or color_sigma <= 0:
        raise ColorRegistrationError("color_sigma must be finite and positive")
    if not np.isfinite(color_weight) or color_weight < 0:
        raise ColorRegistrationError("color_weight must be finite and nonnegative")
    if not (0.0 <= w < 1.0):
        raise ColorRegistrationError("w must be in [0, 1)")
    for name, value in (("max_iterations", max_iterations), ("analytic_max_iterations", analytic_max_iterations)):
        if int(value) != value or value < 1:
            raise ColorRegistrationError(f"{name} must be a positive integer")
    if not 1 <= min_degree <= max_degree <= 10:
        raise ColorRegistrationError("degrees must satisfy 1 <= min_degree <= max_degree <= 10")
    if sigma2 is not None and (not np.isfinite(sigma2) or sigma2 <= 0):
        raise ColorRegistrationError("sigma2 must be finite and positive")
    for name, value in (("tolerance", tolerance), ("min_sigma2", min_sigma2),
                        ("analytic_tolerance", analytic_tolerance),
                        ("analytic_min_sigma2", analytic_min_sigma2),
                        ("rank_tolerance", rank_tolerance), ("min_mass", min_mass)):
        if not np.isfinite(value) or value <= 0:
            raise ColorRegistrationError(f"{name} must be finite and positive")
    if rank_tolerance >= 1:
        raise ColorRegistrationError("rank_tolerance must be < 1")
    for name, value in (("improvement_relative", improvement_relative),
                        ("rebound_relative", rebound_relative)):
        if not np.isfinite(value) or value < 0:
            raise ColorRegistrationError(f"{name} must be finite and nonnegative")
    for name, value in (("stable_patience", stable_patience),
                        ("no_improve_patience", no_improve_patience),
                        ("min_iterations", min_iterations)):
        if int(value) != value or value < 1:
            raise ColorRegistrationError(f"{name} must be a positive integer")
    if not np.isfinite(divergence_radius) or divergence_radius <= 1:
        raise ColorRegistrationError("divergence_radius must be greater than one")

    rotation = np.eye(d) if initial_rotation is None else np.asarray(initial_rotation, dtype=np.float64)
    translation0 = np.zeros(d) if initial_translation is None else np.asarray(initial_translation, dtype=np.float64)
    if rotation.shape != (d, d) or translation0.shape != (d,):
        raise ColorRegistrationError("invalid initial pose dimensions")
    if not np.allclose(rotation.T @ rotation, np.eye(d), atol=1e-8, rtol=0) or abs(np.linalg.det(rotation) - 1) > 1e-8:
        raise ColorRegistrationError("initial_rotation must belong to SO(d)")

    # Position-only normalization (fixed centroid + fixed RMS common scale).
    center = fixed.mean(axis=0)
    x = fixed - center
    y0 = moving - center
    scale = float(np.linalg.norm(x) / np.sqrt(len(x)))
    if not np.isfinite(scale) or scale <= 1e-150:
        raise ColorNumericalError("fixed cloud has zero or unrepresentable extent")
    x = x / scale
    y0 = y0 / scale

    # Color kernel is independent of pose (color is not deformed): precompute once
    # in RAW color units; the geometric scale is never applied to color.
    color_precision = color_weight / (color_sigma * color_sigma)
    if backend == "cuda_color":
        color_logk = None
    elif color_precision == 0.0:
        color_logk = None  # exact geometry-only path (weight-0 parity)
    else:
        dc = (
            np.sum(mc * mc, axis=1)[:, None]
            - 2.0 * (mc @ fc.T)
            + np.sum(fc * fc, axis=1)[None, :]
        )
        np.maximum(dc, 0.0, out=dc)  # guard tiny negative from round-off
        color_logk = -0.5 * color_precision * dc

    rotation = rotation.copy()
    translation = (rotation @ center + translation0 - center) / scale
    y = y0 @ rotation.T + translation

    rigid_stage = StageResult()
    analytic_stage = StageResult()
    steps: list[AnalyticStep] = []
    sig = 0.0

    if method != "analytic":
        sig = (sigma2 / scale / scale) if (sigma2 is not None and sigma2 > 0) else _initial_variance(x, y)
        sig = max(sig, min_sigma2)
        initial_sig = sig
        history: list[Iteration] = []
        converged = False
        stop = "iteration_limit"
        best_iter = 0
        for it in range(int(max_iterations)):
            if backend == "cuda_color":
                stats = _cuda_color_posterior_statistics(
                    y,
                    x,
                    sig,
                    w,
                    moving_colors=mc,
                    fixed_colors=fc,
                    color_precision=color_precision,
                    single_precision=cuda_single_precision,
                )
            else:
                stats = color_posterior_statistics(y, x, sig, w, inverse=True, color_logk=color_logk)
            rot_step, t_step, nxt, rank, active, fit_rms = _fit_rigid_kabsch(y, stats)
            next_sigma = _variance_from_statistics(nxt, stats, min_sigma2) if update_sigma2 else sig
            motion = float(np.linalg.norm(y - nxt) / np.sqrt(len(y)))
            change = abs(next_sigma - sig) / (sig + 1e-12)
            rotation = rot_step @ rotation
            translation = rot_step @ translation + t_step
            y = nxt
            history.append(Iteration(it + 1, 0, active, rank, next_sigma, stats.nll, motion, fit_rms))
            sig = next_sigma
            best_iter = it + 1
            if motion <= tolerance and change <= tolerance:
                converged = True
                stop = "tolerance"
                break
        rigid_stage = StageResult(tuple(history), initial_sig, sig, converged, stop, best_iter)

    world_translation = scale * translation + center - rotation @ center
    rigid_transformed = moving @ rotation.T + world_translation

    if method != "rigid":
        resolved_init = "filterreg" if method == "nonrigid" else "cpd"
        if sigma2 is not None and sigma2 > 0:
            sig = sigma2 / scale / scale
        elif resolved_init == "cpd":
            sig = _initial_variance(x, y)
        sig = max(sig, analytic_min_sigma2)
        initial_sig = sig
        history = []
        converged = False
        stop = "iteration_limit"
        best_iter = 0
        best_y = y.copy()
        best_sigma = sig
        best_score = np.sqrt(d * sig)
        previous_score = best_score
        best_steps = 0
        stable = 0
        no_improve = 0
        previous_degree: int | None = None
        schedule = _degree_schedule(int(analytic_max_iterations), min_degree, max_degree)
        divergence_limit = divergence_radius * float(np.max(np.linalg.norm(x, axis=1)))
        cursor = 0
        it = 0
        min_exp = len(exponents(d, min_degree))
        while it < int(analytic_max_iterations) and cursor < len(schedule):
            raw_degree = schedule[cursor]
            try:
                stats = color_posterior_statistics(y, x, sig, w, inverse=False, color_logk=color_logk)
                active = int((stats.rho > min_mass).sum())
                if stats.mass <= min_mass or active < min_exp:
                    stop = "insufficient_posterior_mass"
                    break
                step, nxt, active_n, rank, fit_rms = _fit_analytic(
                    y, stats, raw_degree, min_degree=min_degree, max_degree=max_degree,
                    rank_tolerance=rank_tolerance, min_mass=min_mass)
                next_sigma = _variance_from_statistics(nxt, stats, analytic_min_sigma2)
            except ColorNumericalError:
                if history:
                    stop = "numerical_divergence"
                    break
                raise
            if float(np.max(np.linalg.norm(nxt, axis=1))) > divergence_limit:
                stop = "numerical_divergence"
                break
            if previous_degree is not None and previous_degree != step.degree:
                stable = 0
                no_improve = 0
            previous_degree = step.degree
            score = float(np.sqrt(d * next_sigma))
            delta_y = float(np.linalg.norm(nxt - y) / (np.linalg.norm(y) + 1e-12))
            delta_sigma = abs(next_sigma - sig) / (abs(sig) + 1e-12)
            delta_score = abs(score - previous_score) / (abs(previous_score) + 1e-12)
            motion = float(np.linalg.norm(y - nxt) / np.sqrt(len(y)))
            significant = best_score - score > max(1e-12, improvement_relative * abs(best_score))
            no_improve = 0 if significant else no_improve + 1
            steps.append(step)
            y = nxt
            sig = next_sigma
            history.append(Iteration(it + 1, step.degree, active_n, rank, sig, stats.nll, motion, fit_rms))
            if score < best_score:
                best_score = score
                best_y = y.copy()
                best_sigma = sig
                best_steps = len(steps)
                best_iter = it + 1
            stable = stable + 1 if (delta_y < analytic_tolerance and delta_sigma < analytic_tolerance
                                    and delta_score < analytic_tolerance) else 0
            previous_score = score
            if score < analytic_tolerance:
                converged = True
                stop = "residual_tolerance"
                break
            if it + 1 >= min_iterations:
                is_stable = stable >= stable_patience
                is_rebound = no_improve >= stable_patience and score > best_score * (1.0 + rebound_relative) + 1e-12
                is_stalled = no_improve >= no_improve_patience
                if is_stable or is_rebound or is_stalled:
                    nxt_cursor = _advance_degree(schedule, cursor)
                    if nxt_cursor < len(schedule):
                        cursor = nxt_cursor
                        stable = 0
                        no_improve = 0
                        previous_degree = None
                        it += 1
                        continue
                    if is_stable:
                        converged = True
                        stop = "stable_tolerance"
                    else:
                        stop = "internal_rebound" if is_rebound else "no_improvement"
                    break
            it += 1
            cursor += 1
        y = best_y
        sig = best_sigma
        steps = steps[:best_steps]
        analytic_stage = StageResult(tuple(history), initial_sig, sig, converged, stop, best_iter)

    transformed = y * scale + center
    if not np.isfinite(transformed).all():
        raise ColorNumericalError("result overflow")

    rigid_stage = _scale_stage(rigid_stage, scale)
    analytic_stage = _scale_stage(analytic_stage, scale)
    return ColorRegistrationResult(
        rotation=rotation, translation=world_translation, center=center,
        normalization_scale=scale, transformed=transformed, rigid_transformed=rigid_transformed,
        steps=tuple(steps), rigid_stage=rigid_stage, analytic_stage=analytic_stage,
        sigma2=sig * scale * scale, method=method, color_sigma=float(color_sigma),
        color_weight=float(color_weight), color_dim=color_dim, backend=backend,
    )


def _scale_stage(stage: StageResult, scale: float) -> StageResult:
    if not stage.history:
        return stage
    s2 = scale * scale
    hist = tuple(
        Iteration(h.iteration, h.degree, h.active, h.rank, h.sigma2 * s2, h.nll_before,
                  h.step_rms * scale, h.fit_rms * scale)
        for h in stage.history
    )
    return StageResult(hist, stage.initial_sigma2 * s2, stage.final_sigma2 * s2,
                       stage.converged, stage.stop_reason, stage.best_iteration)


def _as_points(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3) or arr.shape[0] == 0:
        raise ColorRegistrationError(f"{name} must have shape (n, 2) or (n, 3) with n>=1")
    if not np.isfinite(arr).all():
        raise ColorRegistrationError(f"{name} must contain only finite values")
    return arr


def _as_colors(name: str, value: Any, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] == 0:
        raise ColorRegistrationError(f"{name} must have shape ({n}, C) with C>=1")
    if not np.isfinite(arr).all():
        raise ColorRegistrationError(f"{name} must contain only finite values")
    return arr
