"""C++/Rust compiled-extension comparison; never substitute the test driver."""
import importlib
import numpy as np
import pytest
import acpd_filterreg as reg


def _backends():
    """Backends both engines implement.

    cuda is excluded because a parity case needs the same backend on both sides and
    only the C++ engine has a device path. The Rust engine raises for it rather than
    computing on the CPU, which tests/test_api.py asserts directly.
    """
    return tuple(b for b in reg.backend_names() if b != 'cuda')

@pytest.mark.parametrize('d',[2,3])
@pytest.mark.parametrize('backend',_backends())
def test_cpp_rust_parity(d,backend):
    importlib.import_module('acpd_filterreg_cpp._native')
    importlib.import_module('acpd_filterreg_rs._native')
    rng=np.random.default_rng(444+d);y=rng.normal(size=(64,d));x=y+.03
    rigid=reg.FilterRegOptions(max_iterations=8,sigma2=.08)
    analytic=reg.AnalyticOptions(max_iterations=8,max_degree=2)
    outputs=[reg.registration(x,y,engine=e,backend=backend,rigid=rigid,analytic=analytic) for e in ('cpp','rust')]
    a,b=outputs
    np.testing.assert_allclose(a.rotation,b.rotation,atol=2e-8,rtol=2e-8)
    np.testing.assert_allclose(a.translation,b.translation,atol=2e-8,rtol=2e-8)
    np.testing.assert_allclose(a.transformed,b.transformed,atol=3e-7,rtol=3e-7)
    assert a.analytic_stage.best_iteration==b.analytic_stage.best_iteration
