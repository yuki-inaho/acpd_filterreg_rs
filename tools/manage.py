"""Portable build/test/document commands. Subprocess failures propagate unchanged."""
from __future__ import annotations
import argparse
import ast
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'cpp/build/core'


def run(*arguments: str, cwd: Path=ROOT, env: dict[str,str]|None=None) -> None:
    print('+',' '.join(map(str,arguments)),flush=True)
    subprocess.run([str(a) for a in arguments],cwd=cwd,env=env,check=True)


def core_cpp() -> None:
    run('cmake','-S',ROOT/'cpp','-B',BUILD,'-G','Ninja',
        '-DACPD_BUILD_PYTHON=OFF','-DACPD_BUILD_TESTS=ON','-DCMAKE_BUILD_TYPE=Release')
    run('cmake','--build',BUILD,'--config','Release','--parallel','2')
    run('ctest','--test-dir',BUILD,'-C','Release','--output-on-failure')


def driver() -> Path:
    p=BUILD/('acpd_test_driver.exe' if os.name=='nt' else 'acpd_test_driver')
    if not p.is_file():raise FileNotFoundError(f'{p}; run core-cpp first')
    return p


def test(engine: str) -> None:
    files=['tests/test_reference.py','tests/test_api.py']
    args=['--native-engine',engine]
    if engine in ('cpp','driver'):
        files.append('tests/test_core_reference.py');args+=['--driver',str(driver())]
    if engine!='driver':files.append('tests/test_native_boundary.py')
    # Colored registration and voxel preprocessing use the C++ extension for
    # their CUDA checks.  Keep them in the C++ environment; the Rust-only
    # environment intentionally does not build/import acpd_filterreg_cpp.
    if engine == 'cpp':
        files += ['tests/test_color_registration.py', 'tests/test_preprocessing.py']
    environment=dict(os.environ)
    environment['PYTHONPATH']=os.pathsep.join([str(ROOT/'python'),str(ROOT/'tests'),environment.get('PYTHONPATH','')])
    run(sys.executable,'-m','pytest',*files,*args,env=environment)


def docs() -> None:
    directory=ROOT/'docs';source='theory_ja.tex'
    if shutil.which('xelatex'):
        for _ in range(3):run('xelatex','-interaction=nonstopmode','-halt-on-error',source,cwd=directory)
    elif shutil.which('tectonic'):
        run('tectonic','--keep-logs',source,cwd=directory)
    else:raise SystemExit('XeLaTeX or Tectonic is required. Use `pixi run -e docs docs`.')


def static_check() -> None:
    """Syntax/configuration inspection only: never labelled native build proof."""
    for p in [*ROOT.glob('**/*.py')]:
        if any(s in p.parts for s in ('build','.pixi','target')):continue
        ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    manifests=[ROOT/'pixi.toml',*ROOT.glob('**/pyproject.toml'),*ROOT.glob('rust/**/Cargo.toml')]
    for p in manifests:tomllib.loads(p.read_text())
    manifest=tomllib.loads((ROOT/'pixi.toml').read_text())
    for env,definition in manifest['environments'].items():
        base={} if definition.get('no-default-feature') else dict(manifest.get('tasks',{}))
        for feature in definition['features']:
            base.update(manifest['feature'][feature].get('tasks',{}))
        for name,task in base.items():
            if isinstance(task,dict):
                for dependency in task.get('depends-on',[]):
                    assert dependency in base,(env,name,dependency)
    expected=(ROOT/'docs/review/WORK_ORDER.sha256').read_text().split()[0]
    # The frozen requirement is a text document.  Normalize checkout/editor
    # line endings so Windows does not report a false DoD change after a CRLF
    # conversion; the expected digest is defined over LF bytes.
    work_order=(ROOT/'docs/review/WORK_ORDER.md').read_bytes().replace(b'\r\n',b'\n').replace(b'\r',b'\n')
    assert hashlib.sha256(work_order).hexdigest()==expected,'Frozen DoD changed'
    prohibited=('todo!(','unimplemented!(','Backend::Grid','analytic.regularization','max_step','allow_degree_reduction')
    for p in (ROOT/'rust').glob('**/*.rs'):
        if 'target' in p.parts:continue
        content=p.read_text()
        for token in prohibited:assert token not in content,(p,token)
    print('Python/TOML syntax, task dependencies, frozen DoD and Rust placeholder scan: PASS')
    print('This is not a Rust, PyO3, nanobind or pixi execution test.')


def main() -> None:
    actions={'core-cpp':core_cpp,'test-core':lambda:test('driver'),'test-cpp':lambda:test('cpp'),
             'test-rust':lambda:test('rust'),'docs':docs,'static-check':static_check}
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=actions)
    actions[parser.parse_args().action]()

if __name__=='__main__':main()
