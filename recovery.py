"""Durable recovery records and exclusive ownership for local collection runs."""
from __future__ import annotations

from contextlib import AbstractContextManager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time

from common import read


def durable_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def tree_manifest(root, exclude=()):
    root = Path(root)
    return {str(path.relative_to(root)): digest(path)
            for path in sorted(root.rglob('*')) if path.is_file()
            and not any(part in {'__pycache__', '.cache'} for part in path.relative_to(root).parts)
            and path.suffix != '.pyc' and str(path.relative_to(root)) not in exclude
            and not path.name.endswith('.tmp')}


def verify_manifest(root, manifest, *, exact=False, exclude=()):
    root = Path(root)
    for name, expected in manifest.items():
        path = root / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Recovery input/artifact changed: {path}')
    if exact and set(tree_manifest(root, exclude)) != set(manifest):
        raise ValueError(f'Recovery file set changed: {root}')


def candidate_manifest(root, objects):
    result = {}
    for name in objects:
        folder = Path(root) / 'graspgen' / name
        report = folder / 'REPORT.json'
        if not report.is_file():
            raise FileNotFoundError(report)
        for relative, value in tree_manifest(folder).items():
            result[str(folder / relative)] = value
        cache = (folder / read(report)['source_cache']).resolve()
        result[str(cache)] = digest(cache)
    return result


def verify_candidates(root, objects, expected):
    if candidate_manifest(root, objects) != expected:
        raise ValueError('Candidate files changed since this run was registered')


class RunLock(AbstractContextManager):
    """flock is released on process death; children retain it until they exit."""
    def __init__(self, path):
        self.path = Path(path)
        self.fd = None

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError(f'Run is still owned by another process: {self.path}') from None
        os.ftruncate(self.fd, 0)
        os.write(self.fd, json.dumps(dict(pid=os.getpid(), host=socket.gethostname(),
                                         acquired_unix_s=time.time())).encode())
        os.fsync(self.fd)
        return self

    def __exit__(self, *unused):
        # Do not LOCK_UN: surviving child processes must keep ownership.
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def guarded_command(command):
    import sys
    return [sys.executable, '-B', str(Path(__file__).parent / 'tools/guarded_process.py'),
            str(os.getpid()), *map(str, command)]


def initialization_resource_error(error):
    text = error.lower()
    return any(token in text for token in (
        'out of memory', 'erroroutofdevicememory', 'error_out_of_device_memory',
        'cuda_error_out_of_memory', 'cudaerror_memoryallocation'))
