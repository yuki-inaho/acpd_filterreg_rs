"""Reusable speed ablation and profiling harness for the C++ core.

Wall-clock seconds are recorded but are NOT a reliable comparison on a loaded
machine. Every row therefore also carries load-independent work counters, and
`--ir` re-runs each case under callgrind for a deterministic instruction count,
which is the metric to use when comparing two builds.

    python tools/benchmark_speed.py                       # sweep, wall clock
    python tools/benchmark_speed.py --ir --reps 1         # deterministic Ir
    python tools/benchmark_speed.py --compare <other-acpd_bench>   # A/B a build
    python tools/benchmark_speed.py --stages nonrigid --sizes 100,200,500,1000 \\
        --backends permutohedral_noblur,cuda --analytic-backends direct,cuda --trials 5

`--compare` takes the path of an `acpd_bench` built from another revision, so a
change can be measured against its own baseline without trusting the clock.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT/'cpp/build/bench'
BINARY = BUILD/'acpd_bench'
IR_PATTERN = re.compile(r'refs:\s*([\d,]+)')


def build() -> Path:
    """Configure and build the benchmark target; reuses an existing build tree."""
    subprocess.run(['cmake', '-S', str(ROOT/'cpp'), '-B', str(BUILD), '-G', 'Ninja',
                    '-DACPD_BUILD_PYTHON=OFF', '-DACPD_BUILD_TESTS=OFF', '-DACPD_BUILD_BENCH=ON',
                    '-DCMAKE_BUILD_TYPE=Release'], check=True, cwd=ROOT)
    subprocess.run(['cmake', '--build', str(BUILD), '--parallel', '4'], check=True, cwd=ROOT)
    return BINARY


def run_case(binary: Path, arguments: list[str], instructions: bool) -> dict:
    command = [str(binary), *arguments]
    if instructions:
        if shutil.which('valgrind') is None:
            raise SystemExit('--ir needs valgrind on PATH')
        command = ['valgrind', '--tool=callgrind', '--callgrind-out-file=/dev/null', *command]
    begin = time.perf_counter()
    done = subprocess.run(command, capture_output=True, text=True, check=True, cwd=ROOT)
    elapsed = time.perf_counter()-begin
    row = json.loads(done.stdout.strip().splitlines()[-1])
    if instructions:
        found = IR_PATTERN.search(done.stderr)
        # Deterministic: unaffected by other processes on the machine.
        row['instructions'] = int(found.group(1).replace(',', '')) if found else None
        row['seconds'] = None
        row['wall_seconds_under_callgrind'] = elapsed
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sizes', default='100,200,500,1000')
    parser.add_argument('--dims', default='2,3')
    parser.add_argument('--stages', default='estep,rigid,nonrigid')
    parser.add_argument('--backends', default='direct,permutohedral,permutohedral_noblur,probreg')
    parser.add_argument('--pair', default='deformed', choices=['deformed', 'independent'])
    parser.add_argument('--reps', type=int, default=1)
    parser.add_argument('--ir', action='store_true', help='deterministic instruction counts via callgrind')
    parser.add_argument('--compare', type=Path, help='second acpd_bench binary to A/B against')
    parser.add_argument('--binary', type=Path, help='use this acpd_bench instead of building one')
    parser.add_argument('--analytic-backends', default='direct',
                        help='comma separated ACPD E-step backends. --backends selects the FilterReg '
                             'E-step and is independent of these; the two stages are separate.')
    parser.add_argument('--cuda-single-precision', action='store_true')
    parser.add_argument('--trials', type=int, default=1,
                        help='repeat each case and keep the fastest, the robust statistic when other '
                             'processes are competing for the machine')
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()

    binary = options.binary or (BINARY if BINARY.is_file() else build())
    binaries = {'current': binary}
    if options.compare:
        binaries['baseline'] = options.compare

    rows = []
    for stage in options.stages.split(','):
        for dimension in options.dims.split(','):
            for size in options.sizes.split(','):
                for backend in options.backends.split(','):
                    for analytic in options.analytic_backends.split(','):
                        if stage in ('estep', 'rigid') and analytic != options.analytic_backends.split(',')[0]:
                            continue  # neither stage reaches the ACPD E-step
                        arguments = ['--stage', stage, '--d', dimension, '--n', size,
                                     '--backend', backend, '--reps', str(options.reps)]
                        if stage != 'estep':
                            arguments += ['--pair', options.pair, '--analytic-backend', analytic]
                        if options.cuda_single_precision:
                            arguments += ['--cuda-single-precision', '1']
                        metric = 'instructions' if options.ir else 'seconds'
                        for label, path in binaries.items():
                            best = None
                            for _ in range(max(1, options.trials)):
                                candidate = run_case(path, arguments, options.ir)
                                if best is None or (candidate.get(metric) or 0) < (best.get(metric) or 0):
                                    best = candidate
                            best['build'] = label
                            best['trials'] = max(1, options.trials)
                            rows.append(best)
                            print(json.dumps(best), flush=True)

    metric = 'instructions' if options.ir else 'seconds'
    report = {'metric_note': 'seconds are load-sensitive; instructions are deterministic',
              'primary_metric': metric, 'reps': options.reps, 'pair': options.pair, 'rows': rows}
    if options.compare:
        keys = ('stage', 'backend', 'analytic_backend', 'n', 'd')
        current = {tuple(r[k] for k in keys): r for r in rows if r['build'] == 'current'}
        baseline = {tuple(r[k] for k in keys): r for r in rows if r['build'] == 'baseline'}
        report['speedup'] = {'/'.join(map(str, k)): (baseline[k][metric]/current[k][metric])
                             for k in current if k in baseline
                             and current[k].get(metric) and baseline[k].get(metric)}
        for key, value in report['speedup'].items():
            print(f'{key:>42}: baseline/current = {value:.3f}x', file=sys.stderr)
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
