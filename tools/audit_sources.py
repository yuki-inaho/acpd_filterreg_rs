"""Inspect source completeness; deliberately not a substitute for compilation."""
from __future__ import annotations
import dataclasses
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from acpd_filterreg import AnalyticOptions, FilterRegOptions


def main() -> None:
    cpp = (ROOT / 'cpp/src/bindings.cpp').read_text()
    rust = (ROOT / 'rust/crates/acpd-py/src/lib.rs').read_text()
    expected = {'method', 'backend'}
    for prefix, cls in [('rigid', FilterRegOptions), ('analytic', AnalyticOptions)]:
        expected.update(f'{prefix}_{f.name}' for f in dataclasses.fields(cls))
    checks = []
    for key in sorted(expected):
        both = f'"{key}"' in cpp and f'"{key}"' in rust
        checks.append({'option': key, 'present_in_both_bindings': both})
        if not both:
            raise AssertionError(key)
    symbols = ['registration', 'gaussian_sum', 'posterior_stats', 'basis', 'permutohedral_filter']
    for name in symbols:
        assert f'm.def("{name}"' in cpp, name
        assert re.search(r'fn\s+'+name+r'\b', rust), name
    own_rust = list((ROOT/'rust/crates').glob('*/src/*.rs'))
    test_files = list((ROOT/'rust/crates/acpd-core/tests').rglob('*.rs'))
    rust_count = sum(len(re.findall(r'#\[test\]', p.read_text())) for p in test_files)
    assert rust_count == 90, rust_count
    for p in own_rust:
        text=p.read_text()
        assert not any(t in text for t in ('todo!(', 'unimplemented!(', 'extern "C"', 'unsafe {'))
    work = ROOT/'docs/review/WORK_ORDER.md'
    result = {
        'status': 'SOURCE_ONLY',
        'meaning': 'Source inspection only. Rust/PyO3/nanobind not compiled by this audit.',
        'option_keys': checks, 'native_exports': symbols,
        'rust_test_functions_present': rust_count,
        'rust_core_independent_of_cpp': True,
        'rust_core_forbids_unsafe_code': '#![forbid(unsafe_code)]' in (ROOT/'rust/crates/acpd-core/src/lib.rs').read_text(),
        'work_order_sha256': hashlib.sha256(work.read_bytes()).hexdigest(),
        'own_native_source_sha256': {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(own_rust + list((ROOT/'cpp/src').glob('*.cpp')))
        },
        'dependency_api_review': [
            {'dependency': 'numpy 0.28.0',
             'url': 'https://docs.rs/numpy/0.28.0/numpy/trait.PyUntypedArrayMethods.html',
             'checked': ['is_aligned','is_c_contiguous','shape']},
            {'dependency': 'nanobind', 'url': 'https://nanobind.readthedocs.io/en/latest/ndarray.html',
             'checked': ['capsule owner lifetime','const input arrays','no implicit conversion']},
        ],
    }
    (ROOT/'validation/source_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'Source inspection: {len(checks)} option keys, {len(symbols)} exports, {rust_count} Rust tests present.')
    print('SOURCE_ONLY: these checks do not establish compilation or numerical correctness in Rust.')


if __name__ == '__main__':
    main()
