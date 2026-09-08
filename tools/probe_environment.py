"""Record real dependency/build attempts without relabelling missing tools as PASS."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def attempt(command: list[str], output: Path, timeout: int=45) -> dict:
    begin=time.time()
    record={'command':command,'started_unix':begin,'cwd':str(ROOT)}
    try:
        p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=timeout,check=False)
        text=p.stdout+p.stderr
        record.update(returncode=p.returncode,status='PASS' if p.returncode==0 else 'BLOCKED')
    except FileNotFoundError as e:
        text=str(e);record.update(returncode=None,status='BLOCKED',reason='executable_not_found')
    except subprocess.TimeoutExpired as e:
        text=f'timeout after {timeout}s\n{e.stdout!r}\n{e.stderr!r}'
        record.update(returncode=None,status='BLOCKED',reason='timeout')
    output.write_text('+ '+' '.join(command)+'\n'+text,encoding='utf-8')
    record.update(log=output.name,elapsed_seconds=time.time()-begin)
    return record


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,default=ROOT/'cpp/build/boundary-probe')
    parser.add_argument('--download-probe',action='store_true',help='Attempt only the pinned public nanobind wheel download')
    args=parser.parse_args();out=ROOT/'validation';out.mkdir(exist_ok=True)
    report={'python':sys.version,'platform':platform.platform(),
      'executables':{p:shutil.which(p) for p in ('c++','cmake','ninja','rustc','cargo','pixi','xelatex','tectonic')},
      'python_modules':{p:importlib.util.find_spec(p) is not None for p in ('numpy','pytest','nanobind','maturin','scikit_build_core')},
      'attempts':[]}
    commands=[('nanobind_configure.log',['cmake','-S',str(ROOT/'cpp'),'-B',str(args.build_dir),
        '-DACPD_BUILD_PYTHON=ON','-DACPD_BUILD_TESTS=OFF','-DCMAKE_BUILD_TYPE=Release']),
      ('pixi_attempt.log',['pixi','install','-e','cpp']),
      ('rust_attempt.log',['cargo','--version'])]
    if args.download_probe:
        download_dir=args.build_dir.parent/'dependency-download';download_dir.mkdir(parents=True,exist_ok=True)
        commands.append(('nanobind_download.log',[sys.executable,'-m','pip','download','--no-deps','--only-binary=:all:',
          '--retries','0','--timeout','10','--index-url','https://pypi.org/simple','--dest',str(download_dir),'nanobind==2.12.0']))
    for name,command in commands:
        result=attempt(command,out/name);report['attempts'].append(result)
        print(name,result['status'],result['returncode'])
    (out/'environment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
