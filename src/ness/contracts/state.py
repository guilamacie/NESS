"""Plugin state snapshots: restricted data (arrays + JSON), never code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import ContractViolation, IncompatibleVersion
from .hashing import array_hash, content_hash


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    plugin_id: str
    plugin_version: str
    state_schema_id: str
    arrays: dict[str, np.ndarray] = field(default_factory=dict)  # flat key -> array
    data: dict[str, Any] = field(default_factory=dict)          # JSON-able scalars/strings/lists
    parent_hash: str | None = None

    def __post_init__(self) -> None:
        for k, v in self.arrays.items():
            if not isinstance(v, np.ndarray):
                raise ContractViolation(f"state array {k!r} must be a numpy array")
            if v.dtype == object:
                raise ContractViolation(f"state array {k!r} has object dtype; restricted serialisation forbids it")

    def content_hash(self) -> str:
        return content_hash({
            "plugin_id": self.plugin_id,
            "plugin_version": self.plugin_version,
            "state_schema_id": self.state_schema_id,
            "arrays": {k: array_hash(v) for k, v in sorted(self.arrays.items())},
            "data": self.data,
        })

    def require_schema(self, expected: str) -> None:
        if self.state_schema_id != expected:
            raise IncompatibleVersion(f"state schema {self.state_schema_id!r} != expected {expected!r}; fail closed (no guessing)")


def flatten_params(params: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {f"{g}/{n}": np.asarray(a) for g, group in params.items() for n, a in group.items()}


def unflatten_params(flat: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for key, arr in flat.items():
        g, n = key.split("/", 1)
        out.setdefault(g, {})[n] = arr
    return out


@dataclass
class ModuleState:
    """Runtime state of one module instance.

    ``params``: group -> name -> array (numpy at rest). ``buffers``: non-trainable
    arrays that still affect predictions (running stats, fixed readouts, tables).
    ``meta``: JSON-able extras (e.g. fitted scalars). Mutability of each group is
    declared in the descriptor's ``parameter_groups``."""

    params: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    buffers: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "ModuleState":
        return ModuleState(
            params={g: {n: np.array(a, copy=True) for n, a in grp.items()} for g, grp in self.params.items()},
            buffers={n: np.array(a, copy=True) for n, a in self.buffers.items()},
            meta=dict(self.meta),
        )

    def param_count(self, groups: tuple[str, ...] | None = None) -> int:
        return int(sum(a.size for g, grp in self.params.items() if groups is None or g in groups for a in grp.values()))
