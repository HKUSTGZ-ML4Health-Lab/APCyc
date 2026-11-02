#!/usr/bin/env python3
"""Package only checksum-verified text sources; never include model binaries."""
import argparse
import hashlib
import io
from pathlib import Path, PurePosixPath
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = root / 'scripts/source.sha256'
    denied_suffixes = {'.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.pkl', '.pickle',
                       '.h5', '.hdf5', '.npz', '.npy', '.pdb', '.mmap', '.bin'}
    denied_dirs = {'.git', '.venv', 'private', 'validation', 'outputs', 'weights',
                   'datasets', '__pycache__', 'dist'}
    payloads = {}
    for line in manifest.read_text().splitlines():
        expected, name = line.split('  ', 1)
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or name in payloads:
            raise ValueError(f'Invalid or duplicate source path: {name}')
        if (relative.suffix.lower() in denied_suffixes or
                denied_dirs.intersection(relative.parts) or
                (relative.name.startswith('.env') and relative.name != '.env.example')):
            raise ValueError(f'Private or binary asset in source manifest: {name}')
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Source path must stay inside the repository: {name}')
        body = path.read_bytes()
        body.decode('utf-8')
        if len(body) > 5 * 1024 * 1024 or b'\0' in body:
            raise ValueError(f'Unexpected binary or oversized source: {name}')
        if hashlib.sha256(body).hexdigest() != expected:
            raise ValueError(f'Source checksum mismatch: {name}')
        payloads[name] = body
    payloads['scripts/source.sha256'] = manifest.read_bytes()
    if args.out.resolve().is_relative_to(root):
        raise ValueError('Choose an output path outside the source repository.')
    # Create a new archive without overwriting an existing file.
    with args.out.open('xb') as output:
        with tarfile.open(fileobj=output, mode='w:gz') as archive:
            for name, body in sorted(payloads.items()):
                entry = tarfile.TarInfo('APCyc/' + name)
                entry.size = len(body)
                entry.mode = 0o755 if name.endswith('.sh') else 0o644
                entry.mtime = 0
                archive.addfile(entry, io.BytesIO(body))
    print(f'Packaged {len(payloads)} verified source files: {args.out}')


if __name__ == '__main__':
    main()
