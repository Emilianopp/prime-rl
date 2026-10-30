"""Where a segment leaves its world for the next one.

A segment ends with the game still running: the policy has written its summary and
the pickled ``crafter.Env`` (about 100 KB) must reach the branches that continue it.
It travels by reference, never inside the task or the trace (both are logged and
held in memory with every pending group): the env ``put``s the world here at the
cut and records the path, the next segment ``get``s it back, and whoever runs the
search (the tree sampler in the orchestrator, or the in-env beam of the eval
driver) ``delete``s what the search can no longer revisit.

Two backends: ``DiskStateStore`` writes ``<dir>/<name>.pkl`` with a sha256 check on
read, for the RL pipeline where the env workers and the orchestrator share the
node; ``MemoryStateStore`` keeps the blobs in a dict, for the eval driver whose
whole tree lives inside one ``run`` and for tests.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


def retry_io(fn, *args, what: str = "write", attempts: int = 6, **kwargs):
    """Run an I/O call again on a transient OSError (network scratch returns EREMOTEIO /
    EIO / ENOENT-on-mkdir under load): up to `attempts` tries, backing off 1 to 16 s."""
    delay = 1.0
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except OSError as exc:
            if attempt == attempts:
                raise
            logging.getLogger(__name__).warning("%s failed (%s); retry %d/%d in %.0fs", what, exc, attempt, attempts - 1, delay)
            time.sleep(delay)
            delay = min(delay * 2, 16.0)


class StateStore(ABC):
    @abstractmethod
    def put(self, name: str, state: Any) -> tuple[str, str, int]:
        """Store ``state`` under ``name``; returns (path, sha256, size in bytes)."""

    @abstractmethod
    def get(self, path: str, sha256: str | None = None) -> Any:
        """Load the state at ``path``; a sha256, if given, must match."""

    @abstractmethod
    def delete(self, path: str) -> bool:
        """Drop the state at ``path`` (missing is fine); returns whether it existed."""

    @abstractmethod
    def sweep(self, keep: set[str], older_than_s: float) -> int:
        """Drop every stored state not in ``keep`` and older than ``older_than_s``
        seconds (states of groups the search never observed); returns the count."""

    @staticmethod
    def dumps(state: Any) -> bytes:
        return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loads(blob: bytes, path: str, sha256: str | None) -> Any:
        if sha256 is not None:
            digest = hashlib.sha256(blob).hexdigest()
            if digest != sha256:
                raise ValueError(f"state {path} is corrupt: sha256 {digest} != {sha256}")
        return pickle.loads(blob)


class DiskStateStore(StateStore):
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str) -> Path:
        return self.root / f"{name}.pkl"

    def put(self, name: str, state: Any) -> tuple[str, str, int]:
        blob = self.dumps(state)
        path = self.path_for(name)
        tmp = path.with_suffix(".pkl.tmp")

        def write():
            tmp.write_bytes(blob)
            os.replace(tmp, path)  # a reader never sees a half-written file

        retry_io(write, what=f"saving {path}")
        return str(path), hashlib.sha256(blob).hexdigest(), len(blob)

    def get(self, path: str, sha256: str | None = None) -> Any:
        return self.loads(retry_io(Path(path).read_bytes, what=f"reading {path}"), path, sha256)

    def delete(self, path: str) -> bool:
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return False

    def sweep(self, keep: set[str], older_than_s: float) -> int:
        cutoff = time.time() - older_than_s
        dropped = 0
        for file in self.root.glob("*.pkl"):
            if str(file) in keep:
                continue
            try:
                if file.stat().st_mtime < cutoff:
                    file.unlink()
                    dropped += 1
            except FileNotFoundError:
                pass
        return dropped


class MemoryStateStore(StateStore):
    def __init__(self) -> None:
        self.blobs: dict[str, tuple[bytes, float]] = {}

    def put(self, name: str, state: Any) -> tuple[str, str, int]:
        blob = self.dumps(state)
        path = f"memory://{name}"
        self.blobs[path] = (blob, time.time())
        return path, hashlib.sha256(blob).hexdigest(), len(blob)

    def get(self, path: str, sha256: str | None = None) -> Any:
        if path not in self.blobs:
            raise FileNotFoundError(path)
        return self.loads(self.blobs[path][0], path, sha256)

    def delete(self, path: str) -> bool:
        return self.blobs.pop(path, None) is not None

    def sweep(self, keep: set[str], older_than_s: float) -> int:
        cutoff = time.time() - older_than_s
        stale = [p for p, (_, at) in self.blobs.items() if p not in keep and at < cutoff]
        for p in stale:
            del self.blobs[p]
        return len(stale)


def open_store(state_dir: str | None) -> StateStore:
    """A disk store under ``state_dir``, or an in-memory one when unset."""
    return DiskStateStore(state_dir) if state_dir else MemoryStateStore()
