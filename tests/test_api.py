import json
import gc
import numpy as np
import pytest
import acpd_filterreg as reg
from acpd_filterreg import _api


def pair(d):
    rng = np.random.default_rng(309+d)
    y = rng.uniform(-1,1,(96,d))*np.arange(1,d+1)
    x = y.copy(); x[:,0] += .04+.015*y[:,1]**2
    return x,y


@pytest.mark.parametrize('d', [2,3])
@pytest.mark.parametrize('backend', reg.backend_names())
def test_frozen_pose_composition_and_storage(d, backend, engine, tmp_path):
    x,y=pair(d)
    rigid=reg.FilterRegOptions(sigma2=.04,max_iterations=18)
    analytic=reg.AnalyticOptions(max_iterations=20,max_degree=3)
    r=reg.registration_rigid(x,y,backend=backend,engine=engine,rigid=rigid)
    h=reg.registration_nonrigid(x,y,backend=backend,engine=engine,rigid=rigid,analytic=analytic)
    np.testing.assert_allclose(h.rotation,r.rotation,atol=1e-13,rtol=0)
    np.testing.assert_allclose(h.translation,r.translation,atol=1e-13,rtol=0)
    np.testing.assert_allclose(h.transform(y),h.transformed,atol=2e-11)
    assert len(h.steps)==h.analytic_stage.best_iteration
    assert all(item.lattice_mode=='direct' for item in h.analytic_stage.history)
    expected_initial=((x[:,None]-h.rigid_transformed[None])**2).sum()/(d*len(x)*len(y))
    assert h.analytic_stage.initial_sigma2==pytest.approx(expected_initial,rel=2e-13)
    path=tmp_path/'transform.npz';h.save(path);loaded=reg.load_result(path)
    q=np.ascontiguousarray(y[:7]+.013)
    np.testing.assert_allclose(loaded.transform(q),h.transform(q),atol=1e-13)
    np.testing.assert_allclose(loaded.jacobian(q),h.jacobian(q),atol=1e-13)
    jac=h.jacobian(q);eps=1e-6
    for a in range(d):
        plus=q.copy();minus=q.copy();plus[:,a]+=eps;minus[:,a]-=eps
        np.testing.assert_allclose((h.transform(plus)-h.transform(minus))/(2*eps),jac[:,:,a],atol=3e-6,rtol=3e-6)
    assert not h.transformed.flags.writeable
    assert h.rigid_stage.index_builds==(0 if backend=='direct' else h.rigid_stage.index_builds)


@pytest.mark.parametrize('d', [2,3])
def test_explicit_handoff_and_best_rollback(d, engine, tmp_path):
    x,y=pair(d)
    # A deliberately small externally supplied variance makes the initial
    # state the best under the source Fig.1 initialization. History must not be
    # confused with the returned map after an internal rebound.
    r=reg.registration(x,y,engine=engine,
        rigid=reg.FilterRegOptions(max_iterations=8,sigma2=.03),
        analytic=reg.AnalyticOptions(initialization='filterreg',max_iterations=20,max_degree=2))
    assert r.analytic_stage.initial_sigma2==r.rigid_stage.final_sigma2
    scores=[r.analytic_stage.initial_sigma2]+[h.sigma2 for h in r.analytic_stage.history]
    assert r.sigma2==pytest.approx(min(scores),rel=1e-12)
    assert len(r.steps)==r.analytic_stage.best_iteration
    np.testing.assert_allclose(r.transform(y),r.transformed,atol=2e-11)
    path=tmp_path/'rollback.npz';r.save(path)
    np.testing.assert_allclose(reg.load_result(path).transform(y),r.transformed,atol=2e-11)


@pytest.mark.parametrize('d', [2,3])
def test_noblur_index_reused_with_fixed_variance(d, engine):
    x,y=pair(d)
    result=reg.registration_rigid(x,y,engine=engine,backend='permutohedral_noblur',
        rigid=reg.FilterRegOptions(sigma2=.6,update_sigma2=False,max_iterations=5))
    assert result.rigid_stage.index_builds==1
    assert result.rigid_stage.initial_sigma2==result.rigid_stage.final_sigma2
    assert result.rigid_stage.iterations>1


@pytest.mark.parametrize('d', [2,3])
def test_unordered_unequal_counts(d, engine):
    x,y=pair(d)
    a=reg.registration_analytic(x,y,engine=engine,analytic=reg.AnalyticOptions(max_iterations=8,max_degree=2))
    b=reg.registration_analytic(x[::-1].copy(),y,engine=engine,analytic=reg.AnalyticOptions(max_iterations=8,max_degree=2))
    np.testing.assert_allclose(a.transformed,b.transformed,atol=2e-9)
    c=reg.registration_analytic(x[:-3].copy(),y,engine=engine,analytic=reg.AnalyticOptions(max_iterations=8,max_degree=2))
    assert c.transformed.shape==y.shape


@pytest.mark.parametrize('d', [2,3])
def test_initial_pose_coordinate_conjugation(d, engine):
    x,y=pair(d)
    theta=.2;c,s=np.cos(theta),np.sin(theta);r=np.eye(d);r[:2,:2]=[[c,-s],[s,c]];t=np.arange(d)*.2+.1
    shift=np.arange(d)*900.;scale=37.
    result=reg.registration_analytic(x*scale+shift,y*scale+shift,initial_rotation=r,
        initial_translation=t,engine=engine,analytic=reg.AnalyticOptions(max_iterations=2,max_degree=1))
    np.testing.assert_allclose(result.rotation,r,atol=1e-14)
    np.testing.assert_allclose(result.translation,t,atol=1e-11)
    np.testing.assert_allclose(result.rigid_transformed,(y*scale+shift)@r.T+t,atol=1e-11)
    np.testing.assert_allclose(result.transform(y*scale+shift),result.transformed,atol=1e-10)


@pytest.mark.parametrize('d',[2,3])
def test_normals_registration(d,engine):
    rng=np.random.default_rng(808+d)
    y=rng.normal(size=(120,d));x=y+0.003
    normals=rng.normal(size=x.shape);normals/=np.linalg.norm(normals,axis=1)[:,None]
    result=reg.registration_rigid(x,y,target_normals=normals,engine=engine,backend='direct',
        rigid=reg.FilterRegOptions(objective='point_to_plane',sigma2=1e-4,max_iterations=5))
    assert np.sqrt(np.mean((result.transformed-x)**2))<1e-5


@pytest.mark.parametrize('kind',['cpd','filterreg'])
def test_w_zero_no_denominator_floor_distortion(kind,engine):
    x=np.array([[900.,0.],[901.,0.],[903.,0.]])
    y=np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.]])
    stats=reg.posterior_stats(x,y,sigma2=1.,w=0,kind=kind,engine=engine)
    assert stats.mass==pytest.approx(len(x) if kind=='cpd' else len(y))


def test_zero_lattice_support_is_exposed(engine):
    x=np.array([[900.,0.],[901.,1.],[903.,2.]])
    y=np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.]])
    stats=reg.posterior_stats(x,y,sigma2=1.,w=0,kind='filterreg',backend='permutohedral',engine=engine)
    assert stats.mass==0 and stats.unsupported==len(y) and stats.nll is None
    with pytest.raises(RuntimeError):
        reg.registration_rigid(x,y,engine=engine,rigid=reg.FilterRegOptions(sigma2=.01))


@pytest.mark.parametrize('config',[
    {'max_degree':11},{'rank_tolerance':1.0},{'w':1.0},{'w':-.1},{'max_iterations':False},
    {'min_degree':0},{'stable_patience':0},{'initialization':'unknown'},
    {'regularization':1e-6},{'max_step':.25},
])
def test_analytic_configuration_rejected(config):
    with pytest.raises((ValueError,TypeError)):
        reg.AnalyticOptions(**config)


@pytest.mark.parametrize('bad',[
    np.zeros((3,4)),np.empty((0,2)),np.full((5,2),np.nan),np.zeros((5,2),dtype='float32'),
    np.zeros((10,2))[::2],np.array([[1.,2.]],order='C'),[[1,2],[3,4]],
])
def test_input_rejected_before_native_lookup(bad):
    with pytest.raises((ValueError,TypeError)):
        reg.registration(np.ones((5,2)),bad)


@pytest.mark.parametrize('backend',['grid','radius','ifgt','unknown'])
def test_old_substitution_names_rejected(backend):
    with pytest.raises(ValueError):
        reg.registration(*pair(2),backend=backend)


def test_defaults_and_degree_ten():
    assert reg.AnalyticOptions().max_degree==10
    assert reg.AnalyticOptions().w==.1
    assert reg.FilterRegOptions().solver=='twist'
    assert reg.AnalyticOptions(max_iterations=1).max_degree==10


def test_alignment_and_conversion(engine):
    x,y=pair(2)
    unaligned=np.ndarray(y.shape,dtype='float64',buffer=bytearray(y.nbytes+1),offset=1);unaligned[:]=y
    with pytest.raises(ValueError):reg.registration_analytic(x,unaligned,engine=engine)
    r=reg.registration_analytic(x.astype('float32'),unaligned,copy=True,engine=engine,
                               analytic=reg.AnalyticOptions(max_iterations=2))
    assert r.transformed.dtype==np.float64


def test_no_native_fallback(monkeypatch):
    def missing(_):raise ImportError('deliberate missing native extension')
    monkeypatch.setattr(_api,'import_module',missing)
    with pytest.raises(reg.NativeExtensionUnavailableError,match='No fallback'):
        reg.registration(*pair(2),engine='rust')


def test_readonly_input_and_output_ownership(engine):
    x,y=pair(2);original=y.copy();x.flags.writeable=False;y.flags.writeable=False
    r=reg.registration_analytic(x,y,engine=engine,analytic=reg.AnalyticOptions(max_iterations=2))
    np.testing.assert_array_equal(y,original)
    saved=r.transformed.copy();del x,y;gc.collect()
    np.testing.assert_array_equal(saved,r.transformed)


@pytest.mark.parametrize('mutation', ['format','engine','best','variance','converged','nll','degree'])
def test_malformed_saved_state_is_rejected(mutation, engine, tmp_path):
    x,y=pair(2)
    result=reg.registration_analytic(x,y,engine=engine,analytic=reg.AnalyticOptions(max_iterations=2,max_degree=1))
    original=tmp_path/'original.npz';result.save(original)
    with np.load(original,allow_pickle=False) as archive:
        arrays={name:archive[name].copy() for name in archive.files}
    metadata=json.loads(str(arrays['metadata'].item()))
    if mutation=='format':metadata['format_version']=1
    elif mutation=='engine':metadata['engine']='fallback'
    elif mutation=='best':metadata['analytic_stage']['best_iteration']=1000
    elif mutation=='variance':metadata['sigma2']*=2
    elif mutation=='converged':metadata['analytic_stage']['converged']='true'
    elif mutation=='nll':metadata['analytic_stage']['history'][0]['nll_before']='not a number'
    else:metadata['degrees'][0]=11
    arrays['metadata']=np.array(json.dumps(metadata))
    invalid=tmp_path/'invalid.npz';np.savez(invalid,**arrays)
    with pytest.raises((ValueError,TypeError)):reg.load_result(invalid)
