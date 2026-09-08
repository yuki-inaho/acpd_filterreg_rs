"""Measure release C++ core differences against frozen ORIGINAL-source outputs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from driver_adapter import Driver


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--driver',required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'validation/upstream_comparison.json');a=parser.parse_args()
    filename=ROOT/'tests/fixtures/upstream.json';cases=json.loads(filename.read_text())['cases'];driver=Driver(a.driver)
    reports={}
    def add(group,error,ok=True):
        item=reports.setdefault(group,{'cases':0,'max_absolute_difference':0.0,'all_structural_checks_equal':True})
        item['cases']+=1;item['max_absolute_difference']=max(item['max_absolute_difference'],float(error))
        item['all_structural_checks_equal'] &= bool(ok)
    def diff(x,y):return float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
    for c in cases['lattice']:
        z=driver.permutohedral_filter(np.array(c['features']),np.array(c['values']),c['blur'],c['start'],c['reverse'])
        add('permutohedral',diff(z['values'],c['expected']['values']),z['vertices']==c['expected']['vertices'])
    for c in cases['stats']:
        z=driver.posterior_stats(np.array(c['x']),np.array(c['y']),c['sigma2'],c['w'],False,'direct')
        add('cpd_posterior',max(diff(z[k],c['expected'][k]) for k in ('rho','px','x2','mass')))
    for c in cases['basis']:add('taylor_basis',diff(driver.basis(np.array(c['y']),c['degree']),c['expected']))
    for c in cases['fit']:
        z=driver.fit(np.array(c['y']),np.array(c['z']),np.array(c['weights']),c['degree'])
        add('analytic_mstep',diff(z['next'],c['expected']),z['degree']==c['degree'])
    for c in cases['schedule']:
        z=driver.schedule(c['budget'],1,c['degree']);add('degree_schedule',diff(z,c['expected']),z==c['expected'])
    for c in cases['simplex']:
        z=driver._run(['simplex',c['d'],len(c['features']),False],np.array(c['features']))
        add('original_filterreg_simplex',max(diff(x['weights'],y['weights']) for x,y in zip(z,c['expected'],strict=True)),
            all(x['keys']==y['keys'] for x,y in zip(z,c['expected'],strict=True)))
    thresholds={'permutohedral':2e-6,'cpd_posterior':2e-13,'taylor_basis':1e-14,'analytic_mstep':2e-10,
                'degree_schedule':0.0,'original_filterreg_simplex':4e-7}
    for k,v in reports.items():v['pass']=v['all_structural_checks_equal'] and v['max_absolute_difference']<=thresholds[k]
    report={'execution_path':'C++ standalone driver, NOT nanobind/PyO3',
        'fixture_sha256':hashlib.sha256(filename.read_bytes()).hexdigest(),'groups':reports,
        'all_pass':all(v['pass'] for v in reports.values()),
        'upstream_lattice_precision':'float32; release float64',
        'scope':'component equivalence, not full-paper benchmark or accuracy guarantee'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));raise SystemExit(0 if report['all_pass'] else 1)
if __name__=='__main__':main()
