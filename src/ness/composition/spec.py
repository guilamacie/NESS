"""``CompositionGraphSpec`` (``ness.composition/1``): the DAG of scientifically meaningful
macro modules with named typed ports, source selectors, boundary transforms and merges.

This is NOT a universal tensor IR. Internal blocks stay inside module implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contracts import CompositionError, content_hash, require_schema
from .selectors import SourceSelector

COMPOSITION_SCHEMA = "ness.composition/1"


@dataclass(frozen=True, slots=True)
class BoundaryTransformSpec:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def canonical(self) -> dict:
        return {"kind": self.kind, "params": dict(sorted(self.params.items()))}


@dataclass(frozen=True, slots=True)
class SourceRef:
    selector: SourceSelector
    boundary: tuple[BoundaryTransformSpec, ...] = ()

    def canonical(self) -> dict:
        return {"from": str(self.selector), "boundary": [b.canonical() for b in self.boundary]}


@dataclass(frozen=True, slots=True)
class InputWiring:
    """How one destination input port is fed: one or more sources, an explicit merge
    when there is more than one, and post-merge boundary transforms."""

    port: str
    sources: tuple[SourceRef, ...]
    merge: str | None = None
    merge_params: dict[str, Any] = field(default_factory=dict)
    post: tuple[BoundaryTransformSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.sources:
            raise CompositionError(f"input {self.port!r} has no sources")
        if len(self.sources) > 1 and not self.merge:
            raise CompositionError(f"input {self.port!r} has {len(self.sources)} sources but no explicit merge operator")
        if len(self.sources) == 1 and self.merge not in (None, "select"):
            raise CompositionError(f"input {self.port!r}: merge {self.merge!r} needs more than one source")

    def canonical(self) -> dict:
        return {"port": self.port, "sources": [s.canonical() for s in self.sources], "merge": self.merge,
                "merge_params": dict(sorted(self.merge_params.items())), "post": [b.canonical() for b in self.post]}


@dataclass(frozen=True, slots=True)
class NodeSpec:
    node_id: str
    plugin: str
    config: dict[str, Any] = field(default_factory=dict)
    inputs: tuple[InputWiring, ...] = ()
    memory_queries: tuple[str, ...] = ()
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.node_id or "/" in self.node_id or ":" in self.node_id:
            raise CompositionError(f"invalid node id {self.node_id!r}")
        names = [w.port for w in self.inputs]
        if len(names) != len(set(names)):
            raise CompositionError(f"node {self.node_id}: duplicate input wiring {names}")

    def canonical(self) -> dict:
        return {"node_id": self.node_id, "plugin": self.plugin, "config": self.config, "inputs": [w.canonical() for w in self.inputs],
                "memory_queries": list(self.memory_queries), "enabled": self.enabled}


@dataclass(frozen=True, slots=True)
class OutputSpec:
    task_id: str
    source: SourceSelector

    def canonical(self) -> dict:
        return {"task": self.task_id, "from": str(self.source)}


@dataclass(frozen=True, slots=True)
class InferenceWorkspaceSpec:
    """A declared recurrent region. Cycles are legal only inside one of these, and the
    region carries its own state variables, energy/update rule, solver, iteration budget
    and derivative contract (PDF §16, addendum §3.5)."""

    workspace_id: str
    node_ids: tuple[str, ...]
    state_variables: tuple[str, ...]
    energy: str
    solver: str
    iterations: int
    derivative: str  # bp_unroll | pc_local | ep_centered | ...

    def canonical(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in
                (("workspace_id", self.workspace_id), ("node_ids", self.node_ids), ("state_variables", self.state_variables),
                 ("energy", self.energy), ("solver", self.solver), ("iterations", self.iterations), ("derivative", self.derivative))}


@dataclass(frozen=True, slots=True)
class CompositionGraphSpec:
    nodes: tuple[NodeSpec, ...]
    outputs: tuple[OutputSpec, ...]
    recurrent_regions: tuple[InferenceWorkspaceSpec, ...] = ()
    schema_version: str = COMPOSITION_SCHEMA

    def __post_init__(self) -> None:
        require_schema(self.schema_version, COMPOSITION_SCHEMA)
        ids = [n.node_id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise CompositionError(f"duplicate node ids {ids}")
        if not self.outputs:
            raise CompositionError("a composition needs at least one output")
        for r in self.recurrent_regions:
            unknown = set(r.node_ids) - set(ids)
            if unknown:
                raise CompositionError(f"recurrent region {r.workspace_id} names unknown nodes {sorted(unknown)}")

    def node(self, node_id: str) -> NodeSpec:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        raise CompositionError(f"unknown node {node_id!r}")

    def enabled_nodes(self) -> tuple[NodeSpec, ...]:
        return tuple(n for n in self.nodes if n.enabled)

    def canonical(self) -> dict:
        return {"schema_version": self.schema_version, "nodes": [n.canonical() for n in self.nodes],
                "outputs": [o.canonical() for o in self.outputs], "recurrent_regions": [r.canonical() for r in self.recurrent_regions]}

    @property
    def spec_hash(self) -> str:
        return content_hash(self.canonical())
