"""Build independent comparison executables from the supplied archive versions.

No registration implementation from this release is linked. Extraction records
source line ranges and SHA256. Windows-only unrelated code is not compiled.
Usage: python tools/build_reference.py --attachments /path/to/uploads --build build/oracles
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def member(archive: Path, suffix: str) -> bytes:
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.endswith(suffix)]
        if len(names) != 1:
            raise ValueError(f"expected one {suffix} in {archive}; got {names}")
        return z.read(names[0])


def mask_cpp(text: str) -> str:
    pattern = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    return re.sub(pattern, lambda m: ''.join('\n' if c == '\n' else ' ' for c in m[0]), text)


def extract(text: str, name: str, records: list, source: str, kind: str = "function") -> str:
    clean = mask_cpp(text)
    if kind == "struct":
        pattern = rf'\bstruct\s+{re.escape(name)}\s*\{{'
    elif name == "Sn":
        pattern = r'template<class T>\s*T __declspec\(dllexport\) Sn\s*\('
    else:
        pattern = rf'\binline\s+\w+(?:::\w+)*\s+{re.escape(name)}\s*\('
    match = re.search(pattern, clean)
    if match is None:
        raise ValueError(f"missing original definition {source}:{name}")
    left = clean.index('{', match.start())
    depth = 1
    right = left + 1
    while depth:
        if clean[right] == '{': depth += 1
        elif clean[right] == '}': depth -= 1
        right += 1
    if kind == "struct": right = clean.index(';', right) + 1
    code = text[match.start():right]
    records.append(dict(source=source, symbol=name,
                        first_line=text.count('\n', 0, match.start())+1,
                        last_line=text.count('\n', 0, right)+1,
                        extracted_utf8_sha256=hashlib.sha256(code.encode()).hexdigest()))
    return code


def run(command: list[str], log: Path) -> None:
    done = subprocess.run(command, text=True, capture_output=True)
    log.write_text('$ ' + ' '.join(command) + '\n' + done.stdout + done.stderr)
    if done.returncode:
        raise RuntimeError(f"reference build failed; see {log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--attachments', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve(); build.mkdir(parents=True, exist_ok=True)
    records = []
    source_names = []
    acpd_zip = args.attachments / 'Analytic-CPD-main(1).zip'
    for name in ('Algo.h', 'Fitting.h'):
        data = member(acpd_zip, f'/Analytic_CPD/{name}')
        source_names.append(dict(archive=acpd_zip.name, path=f'Analytic_CPD/{name}', sha256=hashlib.sha256(data).hexdigest()))
        (build/name).write_text(data.decode('gb18030'))
    algo = (build/'Algo.h').read_text(); fitting = (build/'Fitting.h').read_text()
    header = '''// Extracted from supplied Analytic-CPD, MIT, Copyright Wei Feng 2026.
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
using namespace Eigen;
using namespace std;
#define __declspec(x)
#define PI 3.14159265358979323846264338327950288419716939937510
using CoeffSet = std::vector<Eigen::MatrixXd>;
'''
    header += extract(algo,'PosteriorResult',records,'Algo.h','struct')+'\n'
    functions = ['Sn','Factorial','Combination','GetCorrectionWeighted','DegreeScheduleDecreasingStages',
                 'UpdateSigma2FromStats','InitializeSigma2','ComputePosteriorP',
                 'MonomialCount','TotalMonomialCount','TotalParamCount','CreateCoeffSet',
                 'ValidateCoeffSet','SetCoeffSetZero','GetTaylorCoef3D','GetMatB4Analytic3D',
                 'GetL4Analytic3D','GetCorrectionWeightedByPoint3D','GetDeltaVec3D4Analytic',
                 'AnalyticFitting3D','ApplyAnalyticMap3D']
    for name in functions: header += extract(algo,name,records,'Algo.h')+'\n'
    header += '\nnamespace ref2d {\n'
    for name in ['CreateCoeffSet2D','GetMatB4Analytic','GetL4Analytic','GetDeltaVec4Analytic',
                 'AnalyticFitting','ApplyAnalyticMap','BuildWeightMatrix2D','AMVFF2D_MStep']:
        header += extract(fitting,name,records,'Fitting.h')+'\n'
    header += '}\nnamespace ref3d {\n'+extract(fitting,'AMVFF3D_MStep',records,'Fitting.h')+'\n}\n'
    (build/'acpd_reference.hpp').write_text(header)
    (build/'source_records.json').write_text(json.dumps(dict(files=source_names,extracts=records),indent=2))
    shutil.copy(ROOT/'tests/oracles/acpd_reference_driver.cpp', build)
    eigen = ROOT/'cpp/third_party/eigen'
    run(['c++','-std=c++17','-O2',f'-I{eigen}',str(build/'acpd_reference_driver.cpp'),'-o',str(build/'acpd_reference')],build/'acpd_build.log')

    probreg_zip = args.attachments/'probreg-master(1).zip'
    for filename in ('permutohedral.cpp','permutohedral.h'):
        data = member(probreg_zip, '/third_party/permutohedral/'+filename)
        (build/filename).write_bytes(data)
        source_names.append(dict(archive=probreg_zip.name,path='third_party/permutohedral/'+filename,sha256=hashlib.sha256(data).hexdigest()))
    shutil.copy(ROOT/'tests/oracles/lattice_reference_driver.cpp',build)
    # Use the exact scalar implementation rather than the platform-dependent SSE path.
    run(['c++','-std=c++17','-O2','-U__SSE__',f'-I{eigen}',str(build/'lattice_reference_driver.cpp'),
         str(build/'permutohedral.cpp'),'-o',str(build/'lattice_reference')],build/'lattice_build.log')
    # Original FilterReg helper without CUDA/PCL includes. Body is unmodified;
    # the tiny key declaration substitutes only its unrelated CUDA header.
    fr_zip = args.attachments/'FilterReg-master.zip'
    data = member(fr_zip,'/geometry_utils/permutohedral_common.hpp')
    source_names.append(dict(archive=fr_zip.name,path='geometry_utils/permutohedral_common.hpp',sha256=hashlib.sha256(data).hexdigest()))
    text = data.decode(); body = text[text.index('template <int FeatureDim>'):text.index('template <int FeatureDim>\nvoid poser::permutohedral_lattice_withblur')]
    # Keep only no-blur scale and no-blur helper (with-blur scale is harmless).
    stub = '''#include <cmath>
namespace poser {
template<int D> struct LatticeCoordKey {short key[D];};
template<int D> float permutohedral_scale_noblur(int);
template<int D> float permutohedral_scale_withblur(int);
template<int D> void permutohedral_lattice_noblur(const float*,LatticeCoordKey<D>*,float*);
}
'''
    (build/'filterreg_helper.hpp').write_text(stub+body)
    shutil.copy(ROOT/'tests/oracles/filterreg_reference_driver.cpp',build)
    run(['c++','-std=c++17','-O2',str(build/'filterreg_reference_driver.cpp'),'-o',str(build/'filterreg_reference')],build/'filterreg_build.log')
    (build/'source_records.json').write_text(json.dumps(dict(files=source_names,extracts=records,
        modifications=['ACPD: select verbatim functions; wrap 2D/3D in comparison namespaces; define __declspec(x) empty',
                       'probreg: -U__SSE__; prefix values explicitly zeroed because upstream compute ignores start',
                       'FilterReg: unchanged no-blur helper body, minimal key declaration instead of CUDA header']),indent=2))
    print(build)

if __name__ == '__main__': main()
