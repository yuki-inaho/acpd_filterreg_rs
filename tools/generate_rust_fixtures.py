"""Generate dependency-free Rust assertions from the frozen upstream oracle data.

This does not recompute expected values with either release implementation.
The generated file is included at compile time by `cargo test -p acpd-core`.
"""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def scalar(x):
    text=repr(float(x))
    return text if any(c in text for c in '.eE') else text+'.0'

def vec(x): return 'Vector::from_row_slice(&['+','.join(map(scalar,x))+'])'

def mat(x):
    return f'Matrix::from_row_slice({len(x)},{len(x[0])},&['+','.join(scalar(a) for row in x for a in row)+'])'

def main():
    cases=json.loads((ROOT/'tests/fixtures/upstream.json').read_text())['cases']
    out=['// Generated from tests/fixtures/upstream.json. Do not hand-edit.\n']
    for i,c in enumerate(cases['lattice']):
        out.append(f'''#[test]
fn upstream_lattice_{i:02}() -> RegResult<()> {{
    let f={mat(c['features'])}; let v={mat(c['values'])};
    let lattice=Permutohedral::new(&f,{str(c['blur']).lower()})?;
    let actual=lattice.filter(&v,{c['start']},{str(c['reverse']).lower()})?;
    assert_eq!(lattice.lattice_size(),{c['expected']['vertices']});
    close(&actual,&{mat(c['expected']['values'])},2e-6,3e-6); Ok(())
}}\n''')
    for i,c in enumerate(cases['stats']):
        e=c['expected']
        out.append(f'''#[test]
fn upstream_posterior_{i:02}() -> RegResult<()> {{
    let x={mat(c['x'])};let y={mat(c['y'])};
    let s=posterior_statistics(&x,&y,{scalar(c['sigma2'])},{scalar(c['w'])},false,Backend::Direct,None,None,&FgtOptions::default())?;
    close(&s.px,&{mat(e['px'])},2e-13,2e-13);
    assert!((&s.rho-&{vec(e['rho'])}).amax()<2e-12);
    assert!((&s.x2-&{vec(e['x2'])}).amax()<2e-12);
    assert!((s.mass-{scalar(e['mass'])}).abs()<2e-12);Ok(())
}}\n''')
    for i,c in enumerate(cases['basis']):
        out.append(f'''#[test]
fn upstream_basis_{i:02}() -> RegResult<()> {{
    close(&basis(&{mat(c['y'])},{c['degree']})?,&{mat(c['expected'])},1e-14,2e-13);Ok(())
}}\n''')
    for i,c in enumerate(cases['fit']):
        out.append(f'''#[test]
fn upstream_fit_{i:02}() -> RegResult<()> {{
    let y={mat(c['y'])};let z={mat(c['z'])};let w={vec(c['weights'])};
    let s=targets(&z,&w,None);
    let fit=fit_analytic(&y,&s,{c['degree']},&AnalyticOptions::default())?;
    close(&fit.next,&{mat(c['expected'])},2e-10,2e-10);Ok(())
}}\n''')
    for i,c in enumerate(cases['schedule']):
        out.append(f'''#[test]
fn upstream_schedule_{i:02}() -> RegResult<()> {{
    assert_eq!(degree_schedule({c['budget']},1,{c['degree']})?,vec!{c['expected']});Ok(())
}}\n''')
    for i,c in enumerate(cases['simplex']):
        body=[]
        for f,e in zip(c['features'],c['expected'],strict=True):
            weights='['+','.join(map(scalar,e['weights']))+']'
            keys='vec!['+','.join('vec!['+','.join(map(str,k))+']' for k in e['keys'])+']'
            body.append(f'''let s=enclosing_simplex(&[{','.join(map(scalar,f))}],false)?;
    assert_eq!(s.keys,{keys});
    for (a,b) in s.weights.iter().zip({weights}.iter()) {{assert!((a-b).abs()<4e-7+3e-6*b.abs());}}''')
        out.append(f'''#[test]
fn upstream_simplex_{i:02}() -> RegResult<()> {{
    {' '.join(body)} Ok(())
}}\n''')
    target=ROOT/'rust/crates/acpd-core/tests/generated/upstream.rs'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text('\n'.join(out))
    print(target.relative_to(ROOT),sum(len(c) for c in cases.values()),'upstream cases')
if __name__=='__main__':main()
