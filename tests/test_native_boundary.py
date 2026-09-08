"""Real compiled-extension acceptance only. A driver/monkeypatch is not permitted.

Missing extensions cause a collection/test ERROR, never a silent skip or fallback.
These tests were supplied but not executed in the source-release environment.
"""
from __future__ import annotations
import gc
from importlib import import_module
import threading
import time
import numpy as np
import pytest
from acpd_filterreg import _api, FilterRegOptions, AnalyticOptions


@pytest.fixture
def native(request):
    engine=request.config.getoption('--native-engine')
    if engine=='driver':
        pytest.fail('Native-boundary acceptance cannot use the test driver')
    name='acpd_filterreg_cpp._native' if engine=='cpp' else 'acpd_filterreg_rs._native'
    module=import_module(name)
    assert module.__file__.endswith(('.so','.pyd','.dylib'))
    return module


def inputs(d=2):
    rng=np.random.default_rng(187+d)
    y=rng.normal(size=(45,d));x=y+.01
    o=_api._options('analytic','direct',FilterRegOptions(),AnalyticOptions(max_iterations=2,max_degree=1))
    return x,y,o,np.eye(d),np.zeros(d),np.empty((0,d))


@pytest.mark.parametrize('kind',['float32','fortran','strided','negative','list','unaligned','nan','empty'])
def test_native_rejects_bad_arrays(native,kind):
    x,*_=inputs()
    if kind=='float32':a=x.astype(np.float32)
    elif kind=='fortran':a=np.asfortranarray(x)
    elif kind=='strided':a=x[::2]
    elif kind=='negative':a=x[::-1]
    elif kind=='list':a=x.tolist()
    elif kind=='nan':a=x.copy();a[0,0]=np.nan
    elif kind=='empty':a=np.empty((0,2))
    else:
        a=np.ndarray(x.shape,dtype=np.float64,buffer=bytearray(x.nbytes+1),offset=1);a[:]=x
    with pytest.raises((TypeError,ValueError,RuntimeError)):
        native.basis(a,2)


@pytest.mark.parametrize('d',[2,3])
def test_native_readonly_ownership_and_roundtrip(native,d):
    args=inputs(d);x,y=args[:2];x.flags.writeable=False;y.flags.writeable=False
    original=y.copy();out=native.registration(*args)
    saved=np.array(out['transformed'],copy=True)
    np.testing.assert_array_equal(y,original)
    assert out['transformed'].dtype==np.float64
    assert out['transformed'].flags.c_contiguous
    del args,x,y;gc.collect()
    np.testing.assert_array_equal(out['transformed'],saved)
    for _ in range(30):native.basis(np.ones((20,d)),10)
    np.testing.assert_array_equal(out['transformed'],saved)
    engine='rust' if native.__name__.startswith('acpd_filterreg_rs.') else 'cpp'
    result=_api.RegistrationResult.from_native(out,engine)
    assert result.analytic_stage.iterations>0


def test_native_missing_options_and_numerical_error(native):
    args=list(inputs());args[2]={}
    with pytest.raises((ValueError,TypeError)):native.registration(*args)
    args=list(inputs());args[0][:]=1
    with pytest.raises(RuntimeError):native.registration(*args)


def test_native_zero_size_normal_sentinel_and_lattice_schema(native):
    native.registration(*inputs())
    f=np.array([[0.,0.],[1.,0.],[0.,1.]])
    out=native.permutohedral_filter(f,np.ones((3,2)),True,0,False)
    assert set(out)=={'values','vertices'}
    assert out['values'].shape==(3,2) and out['vertices']>0


def test_native_releases_gil_during_gaussian_work(native):
    rng=np.random.default_rng(182)
    sources=rng.normal(size=(6000,3));queries=rng.normal(size=(5000,3));values=np.ones((6000,2))
    stamp=[];ready=threading.Event()
    def worker():
        ready.wait();time.sleep(.015);stamp.append(time.perf_counter())
    thread=threading.Thread(target=worker);thread.start()
    start=time.perf_counter();ready.set()
    out=native.gaussian_sum(sources,queries,values,.3,'direct',_api._fgt(None))
    end=time.perf_counter();thread.join(timeout=5)
    assert out.shape==(5000,2) and np.isfinite(out).all()
    assert end-start>.025,'Timing probe too short: enlarge the problem before accepting this test'
    assert stamp and start<stamp[0]<end-.001,'Background Python did not run during native work'
