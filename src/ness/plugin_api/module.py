"""Execution-level types shared by every macro module.

One execution verb (``forward``) serves all module families. Scientific distinctions
live in descriptors, capabilities and typed evidence, not in Python method names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..contracts import (
    AccessPolicy,
    CostLedger,
    Evidence,
    ExecutionBudget,
    FeatureEvidence,
    Forecast,
    HypothesisEvidence,
    ModuleDescriptor,
    ModuleState,
    ObservationBundle,
    PortSpec,
    PredictionRequest,
    PredictiveEvidence,
    Provenance,
    QueryBudget,
    RetrievalEvidence,
    StateSnapshot,
    ContractViolation,
)


@dataclass(frozen=True, slots=True)
class PortValue:
    """A value travelling along a resolved edge, with its producer spec and lineage."""

    payload: Any
    spec: PortSpec
    provenance: Provenance
    evidence_id: str
    mask: np.ndarray | None = None

    def dense(self) -> np.ndarray:
        """Numeric lowering by declared kind. Typed evidence lowers through its own
        documented rule (feature -> masked value, hypothesis -> weights, forecast -> values)."""
        p = self.payload
        if isinstance(p, np.ndarray):
            return p
        if isinstance(p, Forecast):
            return p.dense()
        if isinstance(p, (FeatureEvidence, HypothesisEvidence, PredictiveEvidence)):
            return p.dense()
        if isinstance(p, Evidence):
            raise ContractViolation(f"{p.kind} evidence on port {self.spec.name!r} has no numeric lowering")
        try:
            return np.asarray(p, dtype=np.float64)
        except Exception as exc:  # pragma: no cover
            raise ContractViolation(f"cannot lower payload of type {type(p).__name__} on port {self.spec.name!r}") from exc

    def knownness(self) -> np.ndarray | None:
        p = self.payload
        if isinstance(p, FeatureEvidence):
            return p.knownness
        return self.mask


@dataclass
class RuntimeContext:
    """Per-request execution context handed to every module."""

    request: PredictionRequest
    permitted_bundle: ObservationBundle
    policy: AccessPolicy
    manifest_id: str
    rng: np.random.Generator
    ledger: CostLedger
    memory_views: dict[str, Any] = field(default_factory=dict)  # store_id -> pinned MemoryView
    mode: str = "predict"  # predict | train
    budget: ExecutionBudget = field(default_factory=ExecutionBudget)
    query_budget: QueryBudget = field(default_factory=QueryBudget)
    node_id: str = ""
    services: dict[str, Any] = field(default_factory=dict)  # named non-graph services (memory stores)


@dataclass
class ModuleOutputs:
    """What a module returns: payload per declared output port, extra evidence
    (traces, retrievals), diagnostics and realised cost."""

    ports: dict[str, Any]
    evidence: tuple[Evidence, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    used_inputs: tuple[str, ...] | None = None  # input ports actually consumed (for provenance)


@runtime_checkable
class MacroModule(Protocol):
    """The contract every composition node implements."""

    def describe(self) -> ModuleDescriptor: ...

    def initialize(self, rng: np.random.Generator) -> ModuleState: ...

    def forward(self, inputs: dict[str, PortValue], state: ModuleState, ctx: RuntimeContext) -> ModuleOutputs: ...

    def snapshot_state(self, state: ModuleState) -> StateSnapshot: ...

    def restore_state(self, snapshot: StateSnapshot) -> ModuleState: ...


@runtime_checkable
class DifferentiableModule(Protocol):
    """Trainable numeric modules additionally expose a pure ``apply`` usable inside
    the reference differentiable runtime. ``dense_inputs`` are already-lowered arrays
    (the runtime performs merges/boundary transforms before calling)."""

    def apply(self, params: dict[str, dict[str, Any]], dense_inputs: dict[str, Any], state: ModuleState, xp: Any) -> dict[str, Any]: ...
