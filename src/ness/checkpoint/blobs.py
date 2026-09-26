"""Content-addressed local blob store with restricted (pickle-free) array serialisation."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import numpy as np

from ..contracts import CheckpointError


class LocalBlobStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        return self.root / digest[:2] / digest

    def put(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        p = self._path(digest)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)
        return digest

    def get(self, digest: str) -> bytes:
        p = self._path(digest)
        if not p.exists():
            raise CheckpointError(f"missing blob {digest}")
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise CheckpointError(f"blob {digest} failed checksum verification")
        return data

    def exists(self, digest: str) -> bool:
        return self._path(digest).exists()

    def put_array(self, arr: np.ndarray) -> str:
        buf = io.BytesIO()
        np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
        return self.put(buf.getvalue())

    def get_array(self, digest: str) -> np.ndarray:
        arr = np.load(io.BytesIO(self.get(digest)), allow_pickle=False)
        arr.setflags(write=False)
        return arr
