"""2D/3D registration from NPY/CSV, or a reproducible unordered synthetic example.

Synthetic paired truth is used only for reporting, never for fitting/stopping.
There is no native-engine fallback. Failures propagate with a nonzero exit code.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import acpd_filterreg as reg


def load_points(path: Path) -> np.ndarray:
    if path.suffix.lower()=='.npy':
        value=np.load(path,allow_pickle=False)
    elif path.suffix.lower()=='.csv':
        value=np.loadtxt(path,delimiter=',',ndmin=2)
    else:
        raise ValueError(f'Only .npy and .csv are supported: {path}')
    if np.iscomplexobj(value):raise ValueError('Complex coordinates are not supported')
    return np.ascontiguousarray(value,dtype=np.float64)


def synthetic(d: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    rng=np.random.default_rng(41+d)
    moving=rng.uniform(-1,1,(180,d))*np.arange(1,d+1)
    rotation=np.eye(d);angle=.13
    rotation[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    truth=moving@rotation.T+np.linspace(.15,-.08,d)
    truth[:,0]+=.02*moving[:,1]**2
    fixed=truth[rng.permutation(len(truth))].copy()
    return fixed,moving,truth


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',choices=reg.engine_names(),required=True)
    parser.add_argument('--dimension',type=int,choices=(2,3),default=3,help='Synthetic example dimension')
    parser.add_argument('--fixed',type=Path);parser.add_argument('--moving',type=Path)
    parser.add_argument('--normals',type=Path,help='Unit target normals, NPY/CSV, for line/plane objective')
    parser.add_argument('--method',choices=reg.method_names(),default='nonrigid')
    parser.add_argument('--backend',choices=reg.backend_names(),default='permutohedral')
    parser.add_argument('--sigma2',type=float,default=None,help='FilterReg initial variance in input squared units')
    parser.add_argument('--analytic-sigma2',type=float,default=None)
    parser.add_argument('--inherit-variance',action='store_true')
    parser.add_argument('--max-degree',type=int,default=10)
    parser.add_argument('--iterations',type=int,default=55,help='Analytic iteration budget')
    parser.add_argument('--output',type=Path,default=Path('outputs'))
    args=parser.parse_args()
    if bool(args.fixed)!=bool(args.moving):parser.error('--fixed and --moving must be given together')
    truth=None
    if args.fixed is not None:
        fixed,moving=load_points(args.fixed),load_points(args.moving)
    else:
        fixed,moving,truth=synthetic(args.dimension)
    normals=None if args.normals is None else load_points(args.normals)
    result=reg.registration(fixed,moving,engine=args.engine,method=args.method,backend=args.backend,
        target_normals=normals,
        rigid=reg.FilterRegOptions(sigma2=args.sigma2,objective='point_to_point' if normals is None else 'point_to_plane'),
        analytic=reg.AnalyticOptions(max_iterations=args.iterations,max_degree=args.max_degree,
            sigma2=args.analytic_sigma2,initialization='filterreg' if args.inherit_variance else 'cpd'))
    args.output.mkdir(parents=True,exist_ok=True)
    result.save(args.output/'transform.npz');np.save(args.output/'registered.npy',result.transformed)
    report={'dimension':moving.shape[1],'engine':args.engine,'method':args.method,'backend':args.backend,
        'rigid_stop':result.rigid_stage.stop_reason,'analytic_stop':result.analytic_stage.stop_reason,
        'returned_steps':len(result.steps),'attempted_analytic_iterations':result.analytic_stage.iterations,
        'sigma2_world':result.sigma2,'degree_history':result.degree_history,
        'external_correspondences_used_in_fitting':False}
    if truth is not None:
        def rms(p):return float(np.sqrt(np.mean(np.sum((p-truth)**2,axis=1))))
        report.update(initial_paired_rms=rms(moving),rigid_paired_rms=rms(result.rigid_transformed),final_paired_rms=rms(result.transformed))
    text=json.dumps(report,indent=2,allow_nan=False)
    (args.output/'metrics.json').write_text(text+'\n',encoding='utf-8');print(text)

if __name__=='__main__':main()
