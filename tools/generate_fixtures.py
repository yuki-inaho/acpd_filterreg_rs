"""Generate frozen expectations ONLY from independently compiled upstream code.

The release implementation is never imported. Re-running requires the original
archives and tools/build_reference.py. Normal tests read the committed fixture.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def execute(executable: Path, header: list, *arrays) -> dict | list:
    fields = [str(int(x)) if isinstance(x, bool) else str(x) for x in header]
    fields.extend(repr(float(x)) for a in arrays for x in np.asarray(a).ravel())
    result = subprocess.run([str(executable)], input=' '.join(fields), text=True,
                            capture_output=True, check=True, timeout=180)
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracles', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'tests/fixtures/upstream.json')
    args = parser.parse_args()
    rng = np.random.default_rng(20260908)
    cases = dict(lattice=[], simplex=[], stats=[], basis=[], fit=[], schedule=[])
    ref = args.oracles
    for d in (2,3):
        for blur in (False,True):
            for reverse in (False,True):
                for channels in (1,5):
                    for start in (0,13):
                        f = rng.uniform(-2,2,(47,d)); f[0] = 0; f[1] = f[2]
                        v = rng.normal(size=(47,channels))
                        expected = execute(ref/'lattice_reference',[d,len(f),channels,blur,start,reverse],f,v)
                        cases['lattice'].append(dict(d=d,blur=blur,reverse=reverse,start=start,
                                                     features=f.tolist(),values=v.tolist(),expected=expected))
        f = rng.uniform(-3,3,(49,d)); f[0] = 0
        cases['simplex'].append(dict(d=d,features=f.tolist(),
                                     expected=execute(ref/'filterreg_reference',[d,len(f)],f)))
        for w in (0.0,0.1,0.7):
            x = rng.normal(size=(29,d)); y = rng.normal(size=(37,d)); variance = .73
            expected = execute(ref/'acpd_reference',['stats',d,len(x),len(y),variance,w],x,y)
            cases['stats'].append(dict(d=d,x=x.tolist(),y=y.tolist(),sigma2=variance,w=w,expected=expected))
        for degree in (0,1,2,3,5,10):
            y = rng.uniform(-1,1,(7,d))
            expected = execute(ref/'acpd_reference',['basis',d,len(y),degree],y)
            cases['basis'].append(dict(d=d,degree=degree,y=y.tolist(),expected=expected))
        for degree in (1,2,3,4):
            y = rng.uniform(-1.3,1.3,(83,d)); z = y.copy(); z[:,0] += .14*y[:,1]**2 + .05
            z += rng.normal(scale=.004,size=z.shape)
            weights = rng.uniform(.1,1,(len(y),))
            expected = execute(ref/'acpd_reference',['fit',d,len(y),degree],y,z,weights)
            cases['fit'].append(dict(d=d,degree=degree,y=y.tolist(),z=z.tolist(),weights=weights.tolist(),expected=expected))
    for degree in (1,3,10):
        for budget in (1,3,7,19,54,55,56,100):
            expected = execute(ref/'acpd_reference',['schedule',budget,degree])
            cases['schedule'].append(dict(degree=degree,budget=budget,expected=expected))
    records = json.loads((ref/'source_records.json').read_text())
    out = dict(format_version=1,producer='original supplied sources only; not release numerics',seed=20260908,
               references=records,cases=cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,separators=(',',':')))
    print(json.dumps(dict(counts={k:len(v) for k,v in cases.items()},
                          sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()),indent=2))

if __name__ == '__main__': main()
