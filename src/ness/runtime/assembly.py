"""Input assembly shared by the eager executor and the differentiable runtime.

Given a ``ResolvedInput`` and the producer arrays, apply per-source boundary transforms,
the merge, and post transforms using array namespace ``xp``. Because both execution
paths call this exact function, forward parity between them holds by construction.
"""

from __future__ import annotations

from typing import Any

from ..composition.compiler import ResolvedInput
from ..runtimes import ops


def assemble_dense(ri: ResolvedInput, source_arrays: list[Any], edge_params: dict[str, dict[str, Any]], xp: Any) -> Any:
    transformed = []
    for src, x in zip(ri.sources, source_arrays):
        for b, gid in zip(src.ref.boundary, src.boundary_groups):
            x = ops.apply_boundary(b.kind, b.params, edge_params.get(gid, {}) if gid else {}, x, xp)
        transformed.append(x)
    if len(transformed) > 1 or ri.merge == "select":
        assert ri.merge is not None
        y = ops.apply_merge(ri.merge, ri.merge_params, edge_params.get(ri.merge_group, {}) if ri.merge_group else {}, transformed, xp)
    else:
        y = transformed[0]
    for b, gid in zip(ri.post, ri.post_groups):
        y = ops.apply_boundary(b.kind, b.params, edge_params.get(gid, {}) if gid else {}, y, xp)
    return y
