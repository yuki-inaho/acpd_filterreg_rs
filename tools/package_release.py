"""Create a deterministic, source-only archive with an internal SHA-256 manifest.

The archive intentionally excludes compiler outputs, interpreter caches, font
files, document intermediates, and local environments. It does not claim that
unexecuted acceptance tests have passed.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = 'acpd_filterreg_faithful'
EXCLUDED_DIRS = {'.git', '.pixi', '.pytest_cache', '__pycache__', 'build', 'target', '.venv'}
EXCLUDED_SUFFIXES = {
    '.pyc', '.pyo', '.so', '.pyd', '.dll', '.dylib', '.exe', '.o', '.obj', '.a', '.lib',
    '.ttf', '.otf', '.ttc', '.woff', '.woff2', '.aux', '.toc', '.out', '.synctex', '.gz',
    '.zip', '.whl',
}


def release_files(root: Path) -> list[Path]:
    """Return only regular distributable files; refuse symlinks for safety."""
    result: list[Path] = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f'Symlink cannot be included: {relative}')
        if not path.is_file() or path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        if relative.as_posix() == 'SHA256SUMS':
            continue
        if relative.parts[0] == 'docs' and path.suffix == '.log':
            continue
        result.append(path)
    return result


def package(root: Path, output: Path) -> dict[str, str | int]:
    root, output = root.resolve(), output.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = release_files(root)
    if not files:
        raise ValueError('Refusing to produce an empty release')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        def write(relative: str, data: bytes) -> None:
            entry = zipfile.ZipInfo(f'{ARCHIVE_ROOT}/{relative}', (2026, 9, 8, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data)
        for path in files:
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            manifest.append(f'{hashlib.sha256(data).hexdigest()}  {relative}')
            write(relative, data)
        write('SHA256SUMS', ('\n'.join(manifest) + '\n').encode())
    with zipfile.ZipFile(output) as archive:
        error = archive.testzip()
        if error is not None:
            raise RuntimeError(f'ZIP CRC failure: {error}')
    return {
        'archive': str(output),
        'files_including_manifest': len(files) + 1,
        'bytes': output.stat().st_size,
        'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    import json
    print(json.dumps(package(options.root, options.output), indent=2))


if __name__ == '__main__':
    main()
