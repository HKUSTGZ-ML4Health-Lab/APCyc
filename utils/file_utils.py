"""File identity checks shared by checkpoint and evaluation workflows."""
import hashlib
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def require_distinct_paths(outputs, inputs=()):
    """Reject output aliases, including existing symbolic and hard links."""
    outputs = [Path(path).resolve() for path in outputs if path is not None]
    inputs = [Path(path).resolve() for path in inputs if path is not None]
    def same(first, second):
        return first == second or (first.exists() and second.exists() and first.samefile(second))
    if (any(same(output, source) for output in outputs for source in inputs)
            or any(same(a, b) for i, a in enumerate(outputs) for b in outputs[i+1:])):
        raise ValueError('Output paths must be distinct and must not overwrite inputs.')
