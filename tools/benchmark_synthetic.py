"""Small reproducible accuracy audit, including regressions, not a paper benchmark.

Paired ground truth is available ONLY to this evaluation script. It is never
passed to registration as correspondences or as an optimization/stopping guard.
All cases and unsuccessful outcomes are retained. Parameters are not retuned to
make a test pass. `driver` is explicitly a test-only C++ execution path.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tests')]
import acpd_filterreg as reg
from acpd_filterreg import _api
from driver_adapter import Driver


def rms(x,y):return float(np.sqrt(np.mean(np.sum((x-y)**2,axis=1))))

def symmetric_nn_rms(x,y):
    distance=np.sum((x[:,None,:]-y[None,:,:])**2,axis=2)
    return float(np.sqrt((distance.min(axis=0).mean()+distance.min(axis=1).mean())/2))


def summarize(rows):
    done=[c for c in rows if c['status']=='executed']
    return {'cases':len(rows),'exceptions':sum(c['status']=='error' for c in rows),
      'improved_over_rigid':sum(c.get('improves_over_rigid',False) for c in rows),
      'regressed_over_rigid':sum(c.get('regresses_over_rigid',False) for c in rows),
      'unchanged_within_1e_8':sum(c.get('unchanged_within_tolerance',False) for c in rows),
      'unchanged_map':sum(c.get('steps',-1)==0 for c in rows),
      'median_rigid_paired_rms':float(np.median([c['rigid_paired_rms'] for c in done])) if done else None,
      'median_final_paired_rms':float(np.median([c['final_paired_rms'] for c in done])) if done else None,
      'max_final_paired_rms':float(np.max([c['final_paired_rms'] for c in done])) if done else None}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',choices=['cpp','rust','driver'],default='cpp')
    parser.add_argument('--driver');parser.add_argument('--output',type=Path,default=ROOT/'validation/synthetic_accuracy.json')
    args=parser.parse_args()
    if args.engine=='driver':
        if args.driver is None:parser.error('--driver is required for the test-only driver')
        bridge=Driver(args.driver);_api._get_engine=lambda _:bridge
    engine='cpp' if args.engine=='driver' else args.engine
    results=[]
    for d in (2,3):
        for seed in (88+d,128+d,308+d):
            rng=np.random.default_rng(seed);y=rng.uniform(-1,1,(180,d))*np.arange(1,d+1)
            x=y.copy();x[:,0]+=.03*y[:,1]**2+.06
            for backend in ('permutohedral','permutohedral_noblur','direct'):
                for initialization in ('auto','cpd','filterreg'):
                    case={'dimension':d,'seed':seed,'points':len(y),'backend':backend,'initialization':initialization,
                          'initial_symmetric_nn_rms':symmetric_nn_rms(x,y),'initial_paired_rms':rms(x,y),'rigid_sigma2_world':.08,'analytic_degree':10,'analytic_budget':55}
                    start=time.perf_counter()
                    try:
                        z=reg.registration(x,y,engine=engine,backend=backend,
                          rigid=reg.FilterRegOptions(sigma2=.08),
                          analytic=reg.AnalyticOptions(initialization=initialization))
                        before=rms(z.rigid_transformed,x);after=rms(z.transformed,x)
                        case.update(status='executed',rigid_paired_rms=before,final_paired_rms=after,
                            improves_over_rigid=after<before-1e-8,regresses_over_rigid=after>before+1e-8,
                            unchanged_within_tolerance=abs(after-before)<=1e-8,
                            rigid_symmetric_nn_rms=symmetric_nn_rms(z.rigid_transformed,x),
                            final_symmetric_nn_rms=symmetric_nn_rms(z.transformed,x),
                            steps=len(z.steps),attempted_iterations=z.analytic_stage.iterations,
                            stop_reason=z.analytic_stage.stop_reason,final_sigma2=z.sigma2)
                    except Exception as e:
                        case.update(status='error',exception=type(e).__name__,message=str(e))
                    case['seconds_including_boundary_overhead']=time.perf_counter()-start
                    results.append(case)
    report={'execution_path':args.engine,'case_count':len(results),'benchmark_reproduction':False,
      'evaluation_correspondences_used_in_fitting':False,
      'timing_note':'Includes serialization/subprocess when driver; not a cross-method speed measurement',
      'summary':summarize(results),
      'summary_by_initialization':{k:summarize([c for c in results if c['initialization']==k])
        for k in ('auto','cpd','filterreg')},'cases':results}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))
if __name__=='__main__':main()
