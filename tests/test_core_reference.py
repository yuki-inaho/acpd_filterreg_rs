"""C++ core probes, deliberately separate from Python-extension acceptance tests."""
import json
from pathlib import Path
import numpy as np
import pytest
CASES = json.loads((Path(__file__).parent/'fixtures/upstream.json').read_text())['cases']

@pytest.mark.parametrize('case', CASES['fit'])
def test_upstream_analytic_mstep(case, driver):
    actual = driver.fit(np.array(case['y']), np.array(case['z']), np.array(case['weights']), case['degree'])
    np.testing.assert_allclose(actual['next'], case['expected'], atol=2e-10, rtol=2e-10)
    assert actual['degree'] == case['degree']


@pytest.mark.parametrize('case', CASES['schedule'])
def test_upstream_degree_schedule(case, driver):
    assert driver.schedule(case['budget'], 1, case['degree']) == case['expected']


@pytest.mark.parametrize('case', CASES['simplex'])
def test_original_filterreg_simplex(case, driver):
    actual = driver._run(['simplex', case['d'], len(case['features']), False], np.array(case['features']))
    for a, b in zip(actual, case['expected'], strict=True):
        assert a['keys'] == b['keys']
        np.testing.assert_allclose(a['weights'], b['weights'], atol=4e-7, rtol=3e-6)


@pytest.mark.parametrize('d', [2,3])
def test_barycentric_identity_and_variance(d, driver):
    rng = np.random.default_rng(d)
    x, y, z = rng.normal(size=(13,d)), rng.normal(size=(17,d)), rng.normal(size=(17,d))
    k = np.exp(-((y[:,None]-x[None])**2).sum(axis=2)/1.8)
    p = k/(k.sum(axis=0)+.2)
    rho, px = p.sum(axis=1), p@x
    bary = px/rho[:,None]
    lhs = np.sum(p*((z[:,None]-x[None])**2).sum(axis=2))
    rhs = np.sum(rho*((z-bary)**2).sum(axis=1)) + np.sum(p*((bary[:,None]-x[None])**2).sum(axis=2))
    assert lhs == pytest.approx(rhs, rel=1e-13)
    x2 = p@(x*x).sum(axis=1)
    variance = driver._run(['variance', d, len(z), 1e-12], z, rho, px, x2)
    assert variance == pytest.approx(lhs/(d*p.sum()), rel=2e-13)


@pytest.mark.parametrize('d', [2,3])
def test_twist_derivative_and_mstep(d, driver):
    rng = np.random.default_rng(21+d)
    y = rng.normal(size=(81,d)); weights = rng.uniform(.1,1,len(y))
    k = 3 if d==2 else 6
    zero = np.zeros(k)
    base = driver.twist(y[0], zero)
    for axis in range(k):
        delta = np.eye(k)[axis]*1e-7
        a, b = driver.twist(y[0], delta), driver.twist(y[0], -delta)
        ya = np.array(a['rotation'])@y[0]+a['translation']
        yb = np.array(b['rotation'])@y[0]+b['translation']
        np.testing.assert_allclose((ya-yb)/2e-7, np.array(base['jacobian'])[:,axis], atol=3e-9)
    pose = driver.twist(y[0], np.full(k,.02))
    z = y@np.array(pose['rotation']).T + pose['translation']
    normals = rng.normal(size=y.shape); normals /= np.linalg.norm(normals,axis=1)[:,None]
    for solver, n in [('twist',None),('kabsch',None),('twist',normals)]:
        fit = driver.rigid_fit(y,z,weights,inner=5,solver=solver,normals=n)
        np.testing.assert_allclose(fit['next'],z,atol=2e-10)
        assert fit['fit_rms'] < 2e-10


@pytest.mark.parametrize('d', [2,3])
def test_feasibility_and_no_hidden_ridge(d, driver):
    rng = np.random.default_rng(70+d)
    y = rng.normal(size=(d+2,d)); z = 2*y+3; weights = np.ones(len(y))
    result = driver.fit(y,z,weights,10)
    assert result['degree'] == 1
    np.testing.assert_allclose(result['next'],z,atol=2e-12)
    np.testing.assert_allclose(result['coefficients'][0],np.full(d,3.0),atol=2e-12)

