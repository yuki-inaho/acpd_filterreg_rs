"""Frozen expectations from ORIGINAL sources, not another release implementation."""
import json
from pathlib import Path
import numpy as np
import pytest
import acpd_filterreg as reg


def _backends():
    """Backends shared by the C++ and Rust native-engine test matrices.

    CUDA is tested separately with the ``engine`` fixture.  Keeping it out of
    module-level parametrization is important because collection happens before
    the fixture can select the native engine, and the Rust environment does not
    install the C++ extension merely to answer a CUDA capability query.
    """
    return tuple(b for b in reg.backend_names() if b != 'cuda')
from acpd_filterreg import _api

CASES = json.loads((Path(__file__).parent/'fixtures/upstream.json').read_text())['cases']


@pytest.mark.parametrize('case', CASES['lattice'])
def test_upstream_lattice(case, engine):
    result = reg.permutohedral_filter(np.array(case['features']), np.array(case['values']),
        with_blur=case['blur'], reverse=case['reverse'], start=case['start'], engine=engine)
    # f32 scalar upstream versus independent f64 implementation, before any
    # registration weights or Gaussian-density normalization are applied.
    np.testing.assert_allclose(result.values, case['expected']['values'], atol=2e-6, rtol=3e-6)
    assert result.vertices == case['expected']['vertices']


@pytest.mark.parametrize('case', CASES['stats'])
def test_upstream_exact_posterior(case, engine):
    result = reg.posterior_stats(np.array(case['x']), np.array(case['y']),
                                sigma2=case['sigma2'], w=case['w'], engine=engine)
    for name in ('rho', 'px', 'x2', 'mass'):
        np.testing.assert_allclose(getattr(result, name), case['expected'][name], atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize('case', CASES['basis'])
def test_upstream_taylor_basis(case, engine):
    actual = _api._get_engine(engine).basis(np.array(case['y']), case['degree'])
    np.testing.assert_allclose(actual, case['expected'], atol=1e-14, rtol=2e-13)








@pytest.mark.parametrize('d', [2, 3])
@pytest.mark.parametrize('backend', _backends())
def test_filterreg_fused_moments(d, backend, engine):
    rng = np.random.default_rng(5+d)
    x, y = rng.normal(size=(71,d)), rng.normal(size=(52,d))
    variance, w = .8, .15
    channels = np.column_stack([np.ones(len(x)), x, (x*x).sum(axis=1)])
    moments = reg.gaussian_sum(x, y, channels, sigma2=variance, backend=backend, engine=engine)
    c = (2*np.pi*variance)**(d/2)*w/(1-w)*len(x)/len(y)
    den = moments[:,0]+c
    out = reg.posterior_stats(x, y, sigma2=variance, w=w, kind='filterreg', backend=backend, engine=engine)
    np.testing.assert_allclose(out.rho, moments[:,0]/den, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(out.px, moments[:,1:d+1]/den[:,None], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(out.x2, moments[:,-1]/den, atol=1e-12, rtol=1e-12)
    if backend == 'probreg':
        assert out.lattice_mode == 'probreg_noblur'


@pytest.mark.parametrize('d', [2, 3])
def test_filterreg_fused_moments_cuda(d, engine):
    """Exercise the CUDA fused-moment path only in the C++ environment."""
    if engine != 'cpp' or not reg.cuda_available(engine):
        pytest.skip('the cuda backend needs a device and the C++ engine')
    rng = np.random.default_rng(105+d)
    x, y = rng.normal(size=(71, d)), rng.normal(size=(52, d))
    variance, w = .8, .15
    channels = np.column_stack([np.ones(len(x)), x, (x*x).sum(axis=1)])
    moments = reg.gaussian_sum(x, y, channels, sigma2=variance, backend='cuda', engine=engine)
    c = (2*np.pi*variance)**(d/2)*w/(1-w)*len(x)/len(y)
    den = moments[:, 0] + c
    out = reg.posterior_stats(x, y, sigma2=variance, w=w, kind='filterreg', backend='cuda', engine=engine)
    np.testing.assert_allclose(out.rho, moments[:, 0]/den, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(out.px, moments[:, 1:d+1]/den[:, None], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(out.x2, moments[:, -1]/den, atol=1e-12, rtol=1e-12)








@pytest.mark.parametrize('d', [2, 3])
@pytest.mark.parametrize('variance', [0.5, 0.05, 0.005])
def test_fgt_approximates_the_exact_gaussian_sum(d, variance, engine):
    """The IFGT is an approximation with a controllable, measured error.

    Its truncation error is bounded by the covering radius in units of
    h = sqrt(2*sigma2), which the grid clustering keeps at cluster_radius, so the
    accuracy must not collapse as sigma shrinks.
    """
    rng = np.random.default_rng(17+d)
    x, y = rng.normal(size=(300,d)), rng.normal(size=(260,d))+0.1
    values = np.column_stack([np.ones(len(x)), x])
    exact = reg.gaussian_sum(x, y, values, sigma2=variance, backend='direct', engine=engine)
    approximate = reg.gaussian_sum(x, y, values, sigma2=variance, backend='fgt', engine=engine)
    scale = max(float(np.abs(exact).max()), 1e-300)
    assert float(np.abs(approximate-exact).max())/scale < 1e-4


def test_fgt_error_shrinks_with_a_smaller_covering_radius(engine):
    """Accuracy must be a knob, not luck: halving cluster_radius must help."""
    rng = np.random.default_rng(23)
    x, y = rng.normal(size=(250,3)), rng.normal(size=(220,3))+0.1
    values = np.ones((len(x),1))
    exact = reg.gaussian_sum(x, y, values, sigma2=0.2, backend='direct', engine=engine)
    errors = []
    for radius in (1.0, 0.25):
        got = reg.gaussian_sum(x, y, values, sigma2=0.2, backend='fgt', engine=engine,
                               fgt=reg.FgtOptions(cluster_radius=radius, order=3))
        errors.append(float(np.abs(got-exact).max()))
    assert errors[1] < errors[0]
