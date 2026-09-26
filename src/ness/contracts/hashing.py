"""Canonical serialisation and content hashing.

Every identity in NESS (config hash, state hash, composition hash, manifest id) is a
sha256 over a canonical JSON encoding. Arrays are hashed by dtype, shape and bytes.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from typing import Any

import numpy as np


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return {"__float__": repr(obj)}
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, np.generic):
        return _to_jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(obj).tobytes()).hexdigest(),
        }
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_to_jsonable(v) for v in obj)
    if hasattr(obj, "canonical"):
        return _to_jsonable(obj.canonical())
    raise TypeError(f"cannot canonicalise object of type {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, arrays replaced by digests."""
    return json.dumps(_to_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(obj: Any) -> str:
    """sha256 hex digest of the canonical JSON encoding."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def array_hash(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def short_hash(obj: Any, n: int = 12) -> str:
    return content_hash(obj)[:n]
