"""Numerical tests for the colored ``direct_cpu_color`` FilterReg + Analytic-CPD.

Run isolated from the native-engine conftest:
    PYTHONPATH=python .venv-registration/bin/python -m pytest tests/test_color_registration.py --noconftest

The tests use independent brute-force oracles for the two posterior directions and
the variance update, known rigid / smooth-nonrigid ground truth, ``color_weight=0``
geometry parity, colored disambiguation, support collapse, and input guards.
"""
from __future__ import annotations

import pathlib
import sys
from math import log, pi

import numpy as np
import pytest

# self-contained import of the copied package (no editable install, no external .pth)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from acpd_filterreg.color_registration import (
    ColorNumericalError,
    ColorRegistrationError,
    basis,
    basis_derivative,
    color_posterior_statistics,
    exponents,
    register_color,
)

TWO_PI_LN = log(2.0 * pi)


# --------------------------------------------------------------------------- #
# Independent oracles.
# --------------------------------------------------------------------------- #
def _naive_posterior(y, x, sigma2, w, inverse, color_precision, mc, fc):
    """Brute-force posterior moments, computed differently from the module."""
    m, d = y.shape
    n = x.shape[0]
    logL = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            g = -0.5 * float(np.sum((y[i] - x[j]) ** 2)) / sigma2
            c = -0.5 * color_precision * float(np.sum((mc[i] - fc[j]) ** 2))
            logL[i, j] = g + c
    normalizer = 0.5 * d * (TWO_PI_LN + log(sigma2))
    centers, queries = (n, m) if inverse else (m, n)
    logc = -np.inf if w == 0.0 else normalizer + log(w) - log(1 - w) + log(centers / queries)
    logfactor = log(1 - w) - log(centers) - normalizer
    rho = np.zeros(m)
    px = np.zeros((m, d))
    x2 = np.zeros(m)
    nll = 0.0
    if inverse:
        for i in range(m):
            terms = logL[i, :]
            mx = max(float(terms.max()), logc)
            den = (np.exp(logc - mx) if np.isfinite(logc) else 0.0) + float(np.sum(np.exp(terms - mx)))
            nll -= mx + log(den) + logfactor
            for j in range(n):
                p = float(np.exp(logL[i, j] - mx)) / den
                rho[i] += p
                px[i] += p * x[j]
                x2[i] += p * float(np.sum(x[j] ** 2))
    else:
        for j in range(n):
            terms = logL[:, j]
            mx = max(float(terms.max()), logc)
            den = (np.exp(logc - mx) if np.isfinite(logc) else 0.0) + float(np.sum(np.exp(terms - mx)))
            nll -= mx + log(den) + logfactor
            for i in range(m):
                p = float(np.exp(logL[i, j] - mx)) / den
                rho[i] += p
                px[i] += p * x[j]
                x2[i] += p * float(np.sum(x[j] ** 2))
    return rho, px, x2, float(rho.sum()), nll


def _color_logk(mc, fc, color_precision):
    dc = np.sum((mc[:, None, :] - fc[None, :, :]) ** 2, axis=2)
    return -0.5 * color_precision * dc


# --------------------------------------------------------------------------- #
# Basis / exponents cross-check against the pristine _mapping module.
# --------------------------------------------------------------------------- #
def test_basis_matches_mapping_module():
    from acpd_filterreg import _mapping

    rng = np.random.default_rng(1)
    pts = rng.normal(size=(7, 3))
    for degree in (1, 2, 3):
        assert exponents(3, degree) == _mapping.exponents(3, degree)
        np.testing.assert_allclose(basis(pts, degree), _mapping.basis(pts, degree), rtol=0, atol=1e-14)
        np.testing.assert_allclose(
            basis_derivative(pts, degree), _mapping.basis_derivative(pts, degree), rtol=0, atol=1e-14
        )


def test_cuda_six_dimensional_gaussian_matches_direct():
    import acpd_filterreg as reg

    if not reg.cuda_available():
        pytest.skip("no CUDA-enabled C++ extension/device")
    rng = np.random.default_rng(91)
    sources = np.ascontiguousarray(rng.normal(size=(37, 6)), dtype=np.float64)
    queries = np.ascontiguousarray(rng.normal(size=(29, 6)), dtype=np.float64)
    values = np.ascontiguousarray(rng.normal(size=(37, 5)), dtype=np.float64)
    direct = reg.gaussian_sum(sources, queries, values, sigma2=0.7, backend="direct")
    device = reg.gaussian_sum(sources, queries, values, sigma2=0.7, backend="cuda")
    np.testing.assert_allclose(device, direct, rtol=2e-13, atol=2e-13)


def test_cuda_color_rigid_matches_direct_color():
    import acpd_filterreg as reg

    if not reg.cuda_available():
        pytest.skip("no CUDA-enabled C++ extension/device")
    rng = np.random.default_rng(92)
    fixed = rng.normal(scale=0.07, size=(48, 3))
    colors = rng.normal(size=(48, 3))
    angle = 0.045
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    moving = (fixed - np.array([0.008, -0.004, 0.002])) @ rotation
    kwargs = dict(
        color_sigma=0.8,
        color_weight=0.6,
        method="rigid",
        max_iterations=20,
        tolerance=1e-9,
    )
    direct = register_color(fixed, moving, colors, colors, backend="direct_cpu_color", **kwargs)
    device = register_color(fixed, moving, colors, colors, backend="cuda_color", **kwargs)
    np.testing.assert_allclose(device.rotation, direct.rotation, rtol=0, atol=2e-10)
    np.testing.assert_allclose(device.translation, direct.translation, rtol=0, atol=2e-10)
    np.testing.assert_allclose(device.transformed, direct.transformed, rtol=0, atol=2e-10)


def test_cuda_geometry_weight_zero_matches_direct_geometry():
    import acpd_filterreg as reg

    if not reg.cuda_available():
        pytest.skip("no CUDA-enabled C++ extension/device")
    rng = np.random.default_rng(93)
    fixed = rng.normal(scale=0.05, size=(32, 3))
    moving = fixed + np.array([0.01, -0.006, 0.003])
    colors = rng.uniform(0, 100, size=(32, 3))
    kwargs = dict(
        color_sigma=20.0,
        color_weight=0.0,
        method="rigid",
        max_iterations=12,
        tolerance=1e-9,
    )
    direct = register_color(fixed, moving, colors, colors, backend="direct_cpu_color", **kwargs)
    device = register_color(fixed, moving, colors, colors, backend="cuda_color", **kwargs)
    np.testing.assert_allclose(device.rotation, direct.rotation, rtol=0, atol=2e-10)
    np.testing.assert_allclose(device.translation, direct.translation, rtol=0, atol=2e-10)
    np.testing.assert_allclose(device.transformed, direct.transformed, rtol=0, atol=2e-10)


# --------------------------------------------------------------------------- #
# Posterior oracle, both directions, with and without color.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("inverse", [True, False])
@pytest.mark.parametrize("color_precision", [0.0, 0.05])
@pytest.mark.parametrize("w", [0.0, 0.1, 0.4])
def test_posterior_matches_oracle_both_directions(inverse, color_precision, w):
    rng = np.random.default_rng(7)
    y = rng.normal(size=(5, 3))
    x = rng.normal(size=(6, 3))
    mc = rng.uniform(0, 100, size=(5, 2))
    fc = rng.uniform(0, 100, size=(6, 2))
    sigma2 = 0.7
    clk = None if color_precision == 0.0 else _color_logk(mc, fc, color_precision)
    stats = color_posterior_statistics(y, x, sigma2, w, inverse=inverse, color_logk=clk)
    rho, px, x2, mass, nll = _naive_posterior(y, x, sigma2, w, inverse, color_precision, mc, fc)
    np.testing.assert_allclose(stats.rho, rho, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(stats.px, px, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(stats.x2, x2, rtol=1e-11, atol=1e-12)
    assert stats.mass == pytest.approx(mass, rel=1e-11)
    assert stats.nll == pytest.approx(nll, rel=1e-10)


def test_variance_update_matches_oracle():
    from acpd_filterreg.color_registration import _variance_from_statistics

    rng = np.random.default_rng(3)
    y = rng.normal(size=(6, 3))
    x = rng.normal(size=(6, 3))
    stats = color_posterior_statistics(y, x, 0.5, 0.1, inverse=False, color_logk=None)
    got = _variance_from_statistics(y, stats, 1e-12)
    a = stats.x2
    b = 2.0 * np.sum(y * stats.px, axis=1)
    c = stats.rho * np.sum(y * y, axis=1)
    residual = float(np.sum(a - b + c))
    oracle = max(1e-12, max(residual, 0.0) / (3 * stats.mass))
    assert got == pytest.approx(oracle, rel=1e-12)


# --------------------------------------------------------------------------- #
# Known rigid ground truth + freeze / reapply consistency.
# --------------------------------------------------------------------------- #
def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_known_rigid_recovery():
    rng = np.random.default_rng(11)
    fixed = rng.normal(size=(12, 3)) * 0.03 + np.array([0.4, 0.1, 0.5])
    colors = rng.uniform(20, 80, size=(12, 3))
    R = _rot_z(0.2)
    t = np.array([0.02, -0.015, 0.008])
    # moving is the rigidly displaced fixed cloud; registering moving->fixed must invert it
    moving = fixed @ R.T + t
    res = register_color(fixed, moving, colors, colors, color_sigma=15.0, method="rigid")
    assert res.rigid_stage.stop_reason == "tolerance"
    # recovered pose maps moving back onto fixed
    np.testing.assert_allclose(res.transformed, fixed, atol=1e-9)
    np.testing.assert_allclose(res.rotation, R.T, atol=1e-7)
    np.testing.assert_allclose(res.rotation @ R, np.eye(3), atol=1e-7)


def test_rigid_freeze_and_reapply_consistency():
    rng = np.random.default_rng(5)
    fixed = rng.normal(size=(10, 3)) * 0.05
    colors = rng.uniform(0, 100, size=(10, 3))
    moving = fixed @ _rot_z(0.1).T + np.array([0.01, 0.02, -0.01])
    res = register_color(fixed, moving, colors, colors, color_sigma=20.0, method="nonrigid")
    # rigid_transformed is exactly the frozen pose applied to moving
    np.testing.assert_allclose(res.rigid_transformed, moving @ res.rotation.T + res.translation, atol=1e-12)
    # transform() re-applies frozen pose + all residual maps and reproduces transformed
    np.testing.assert_allclose(res.transform(moving), res.transformed, atol=1e-9)
    # displacement decomposition is self-consistent
    np.testing.assert_allclose(
        res.residual_displacement(moving), res.transformed - res.rigid_transformed, atol=1e-9
    )


# --------------------------------------------------------------------------- #
# Known smooth non-rigid ground truth.
# --------------------------------------------------------------------------- #
def test_known_smooth_nonrigid_warp():
    rng = np.random.default_rng(21)
    fixed = rng.uniform(-0.05, 0.05, size=(12, 3)) + np.array([0.3, 0.0, 0.4])
    colors = rng.uniform(20, 80, size=(12, 3))
    # a gentle quadratic bend applied to the centered cloud; moving = warped fixed
    c = fixed.mean(0)
    q = fixed - c
    warped = q.copy()
    warped[:, 0] += 0.6 * q[:, 1] ** 2 / 0.05 * 0.02  # small smooth bend
    moving = warped + c
    res = register_color(fixed, moving, colors, colors, color_sigma=15.0, method="nonrigid",
                         min_degree=1, max_degree=2)
    # moving mapped through the recovered map should land close to fixed
    residual = np.linalg.norm(res.transformed - fixed, axis=1)
    assert float(residual.mean()) < 5e-3
    assert res.analytic_stage.iterations > 0
    assert res.steps  # a non-trivial residual map was fit


# --------------------------------------------------------------------------- #
# color_weight = 0 geometry parity (same implementation, geometry-only path).
# --------------------------------------------------------------------------- #
def test_weight_zero_geometry_parity():
    rng = np.random.default_rng(31)
    fixed = rng.normal(size=(11, 3)) * 0.04
    moving = fixed @ _rot_z(0.12).T + np.array([0.01, -0.02, 0.005])
    colors_a = rng.uniform(0, 100, size=(11, 3))
    colors_b = rng.uniform(0, 100, size=(11, 3))
    # weight 0 must be independent of the color VALUES ...
    ra = register_color(fixed, moving, colors_a, colors_a, color_sigma=5.0, color_weight=0.0, method="nonrigid")
    rb = register_color(fixed, moving, colors_b, colors_b, color_sigma=5.0, color_weight=0.0, method="nonrigid")
    np.testing.assert_allclose(ra.transformed, rb.transformed, atol=1e-12)
    np.testing.assert_allclose(ra.rotation, rb.rotation, atol=1e-12)
    assert ra.sigma2 == pytest.approx(rb.sigma2, rel=0, abs=1e-18)
    # ... and independent of color_sigma when weight is 0 (precision 0 either way)
    rc = register_color(fixed, moving, colors_a, colors_a, color_sigma=999.0, color_weight=0.0, method="nonrigid")
    np.testing.assert_allclose(ra.transformed, rc.transformed, atol=1e-12)


def test_posterior_weight_zero_ignores_color():
    rng = np.random.default_rng(32)
    y = rng.normal(size=(5, 3))
    x = rng.normal(size=(5, 3))
    mc = rng.uniform(0, 100, size=(5, 2))
    fc = rng.uniform(0, 100, size=(5, 2))
    geom = color_posterior_statistics(y, x, 0.6, 0.1, inverse=True, color_logk=None)
    zero = color_posterior_statistics(y, x, 0.6, 0.1, inverse=True, color_logk=_color_logk(mc, fc, 0.0))
    np.testing.assert_allclose(geom.rho, zero.rho, atol=1e-15)
    np.testing.assert_allclose(geom.px, zero.px, atol=1e-15)


# --------------------------------------------------------------------------- #
# Color disambiguates a geometrically symmetric correspondence.
# --------------------------------------------------------------------------- #
def test_color_disambiguates_symmetric_posterior():
    # moving point at the origin is geometrically equidistant from two fixed
    # candidates; only color distinguishes the correct one.
    y = np.array([[0.0, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, -0.6, 0.0], [0.3, 0.0, 0.0]])
    x = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, -0.6, 0.0]])
    mc = np.array([[50.0], [50.0], [50.0], [50.0]])
    fc = np.array([[50.0], [90.0], [50.0], [50.0]])  # +x candidate matches the moving color
    sigma2 = 0.5
    # geometry only: the origin point is pulled symmetrically -> ~0 net x
    geom = color_posterior_statistics(y, x, sigma2, 0.1, inverse=True, color_logk=None)
    # with color: the +x (color-matching) candidate dominates -> net +x pull
    clk = _color_logk(mc, fc, color_precision=0.05)
    col = color_posterior_statistics(y, x, sigma2, 0.1, inverse=True, color_logk=clk)
    assert abs(geom.px[0, 0]) < 1e-6
    assert col.px[0, 0] > 0.1  # decisively pulled toward the same-color target
    assert col.px[0, 0] > geom.px[0, 0] + 0.1


# --------------------------------------------------------------------------- #
# Large color difference removes inlier support.
# --------------------------------------------------------------------------- #
def test_large_color_difference_removes_support():
    rng = np.random.default_rng(41)
    fixed = rng.normal(size=(10, 3)) * 0.05
    moving = fixed.copy()
    matched = np.full((10, 1), 50.0)
    far = np.full((10, 1), 5000.0)
    m_match = color_posterior_statistics(moving, fixed, 0.01, 0.1, inverse=False,
                                         color_logk=_color_logk(matched, matched, 1.0)).mass
    m_far = color_posterior_statistics(moving, fixed, 0.01, 0.1, inverse=False,
                                       color_logk=_color_logk(matched, far, 1.0)).mass
    assert m_far < 1e-6 * m_match  # color mismatch collapses the posterior support
    # end-to-end: analytic stage refuses to run without support (explicit stop reason)
    res = register_color(fixed, moving, matched, far, color_sigma=1.0, color_weight=1.0, method="analytic")
    assert res.analytic_stage.stop_reason == "insufficient_posterior_mass"
    assert not res.steps


# --------------------------------------------------------------------------- #
# Rank / degree reporting on a degenerate (near-collinear) configuration.
# --------------------------------------------------------------------------- #
def test_rank_and_degree_reported_on_degenerate_cloud():
    # collinear points: the Taylor design cannot be full rank; the analytic fit must
    # record the reduced rank via the minimum-norm SVD solve and never silently
    # substitute a different computation. (The rigid stage legitimately rejects a
    # collinear cloud as non-identifiable, so this exercises the analytic path.)
    t = np.linspace(-0.05, 0.05, 8)
    fixed = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    moving = fixed + np.array([0.005, 0.0, 0.0])
    colors = np.linspace(10, 90, 8)[:, None]
    res = register_color(fixed, moving, colors, colors, color_sigma=20.0, method="analytic",
                         min_degree=1, max_degree=2, sigma2=0.001)
    assert res.analytic_stage.iterations > 0
    # every recorded iteration exposes an explicit rank; at least one is rank deficient
    assert all(row.rank >= 0 for row in res.analytic_stage.history)
    assert any(row.rank < len(exponents(3, row.degree)) for row in res.analytic_stage.history)
    assert res.analytic_stage.stop_reason in (
        "residual_tolerance", "stable_tolerance", "no_improvement", "internal_rebound",
        "iteration_limit", "insufficient_posterior_mass", "numerical_divergence",
    )


# --------------------------------------------------------------------------- #
# Input guards (empty / shape / NaN / bad sigma / bad color).
# --------------------------------------------------------------------------- #
def _ok_inputs():
    rng = np.random.default_rng(99)
    fixed = rng.normal(size=(6, 3)) * 0.05
    moving = fixed + 0.001
    colors = rng.uniform(0, 100, size=(6, 3))
    return fixed, moving, colors


def test_input_guards():
    fixed, moving, colors = _ok_inputs()
    with pytest.raises(ColorRegistrationError):
        register_color(np.empty((0, 3)), moving, colors, colors, color_sigma=10.0)
    with pytest.raises(ColorRegistrationError):
        register_color(fixed, moving[:, :2], colors, colors, color_sigma=10.0)  # dim mismatch
    with pytest.raises(ColorRegistrationError):
        register_color(fixed, moving, colors[:3], colors, color_sigma=10.0)  # color rows mismatch
    with pytest.raises(ColorRegistrationError):
        register_color(fixed, moving, colors, colors[:, :2], color_sigma=10.0)  # color dim mismatch
    bad = fixed.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ColorRegistrationError):
        register_color(bad, moving, colors, colors, color_sigma=10.0)
    for cs in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ColorRegistrationError):
            register_color(fixed, moving, colors, colors, color_sigma=cs)
    with pytest.raises(ColorRegistrationError):
        register_color(fixed, moving, colors, colors, color_sigma=10.0, color_weight=-1.0)
    for bad_w in (1.0, 1.5, -0.1):
        with pytest.raises(ColorRegistrationError):
            register_color(fixed, moving, colors, colors, color_sigma=10.0, w=bad_w)
    with pytest.raises(ColorRegistrationError):
        register_color(fixed, moving, colors, colors, color_sigma=10.0, method="bogus")
    with pytest.raises(ColorRegistrationError):
        register_color(fixed, moving, colors, colors, color_sigma=10.0,
                       initial_rotation=np.diag([1.0, 1.0, -1.0]))  # reflection not in SO(3)
    # too few points
    with pytest.raises(ColorRegistrationError):
        register_color(fixed[:2], moving[:2], colors[:2], colors[:2], color_sigma=10.0)


def test_posterior_rejects_bad_sigma_and_w():
    y = np.zeros((3, 3))
    x = np.ones((3, 3))
    with pytest.raises(ColorRegistrationError):
        color_posterior_statistics(y, x, 0.0, 0.1, inverse=True)
    with pytest.raises(ColorRegistrationError):
        color_posterior_statistics(y, x, 1.0, 1.0, inverse=True)


def test_2d_supported():
    rng = np.random.default_rng(77)
    fixed = rng.normal(size=(8, 2)) * 0.05
    theta = 0.1
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    moving = fixed @ R.T + np.array([0.01, -0.02])
    colors = rng.uniform(0, 100, size=(8, 1))
    res = register_color(fixed, moving, colors, colors, color_sigma=10.0, method="rigid")
    np.testing.assert_allclose(res.transformed, fixed, atol=1e-8)


def _small_colored_pair(seed: int = 3, n: int = 80):
    rng = np.random.default_rng(seed)
    moving = rng.uniform(-1, 1, (n, 3))
    fixed = moving.copy()
    fixed[:, 0] += 0.05 * moving[:, 1] ** 2
    colors = rng.uniform(0, 1, (n, 3))
    return fixed, colors, moving, colors.copy()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"sigma2": -1.0}, "sigma2 must be finite and positive"),
        ({"sigma2": 0.0}, "sigma2 must be finite and positive"),
        ({"sigma2": float("nan")}, "sigma2 must be finite and positive"),
        ({"divergence_radius": 0.5}, "divergence_radius must be greater than one"),
        ({"rank_tolerance": 2.0}, "rank_tolerance must be < 1"),
        ({"rank_tolerance": 0.0}, "rank_tolerance must be finite and positive"),
        ({"min_mass": -1.0}, "min_mass must be finite and positive"),
        ({"stable_patience": 0}, "stable_patience must be a positive integer"),
        ({"improvement_relative": -1.0}, "improvement_relative must be finite and nonnegative"),
    ],
)
def test_invalid_numeric_controls_are_rejected(kwargs, message) -> None:
    """These used to be consumed as truth tests only: a bad sigma2 silently selected
    the automatic initialization, and a bad divergence_radius or rank_tolerance
    produced an analytic stage that ran zero iterations and reported a stop reason
    as if it had converged. AnalyticOptions/FilterRegOptions reject all of them."""
    fixed, fc, moving, mc = _small_colored_pair()
    with pytest.raises(ColorRegistrationError, match=message):
        register_color(fixed_xyz=fixed, fixed_colors=fc, moving_xyz=moving, moving_colors=mc,
                       color_sigma=0.3, max_iterations=6, analytic_max_iterations=6, **kwargs)


def test_transform_overflow_raises_the_module_error_not_FloatingPointError() -> None:
    """basis() runs under errstate(over="raise") itself, so the usual extrapolation
    overflow raised FloatingPointError from inside transform's own errstate block and
    never reached its isfinite check. FloatingPointError is not a
    ColorRegistrationError, so callers catching the documented base class missed it."""
    fixed, fc, moving, mc = _small_colored_pair()
    result = register_color(fixed_xyz=fixed, fixed_colors=fc, moving_xyz=moving, moving_colors=mc,
                            color_sigma=0.3, max_iterations=8, analytic_max_iterations=10,
                            min_degree=2, max_degree=3)
    assert any(step.degree >= 2 for step in result.steps)
    with pytest.raises(ColorNumericalError, match="extrapolation is not bounded"):
        result.transform(np.array([[1e170, 0.0, 0.0], [0.0, 0.0, 0.0]]))
