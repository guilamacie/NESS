"""Typed evidence and the immutable evidence graph.

A probability distribution, a Boolean feature, an uncertain hypothesis, an execution
trace and a query failure are distinct types. Padding and absence never become facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import numpy as np

from .availability import Provenance
from .errors import ContractViolation
from .forecasts import Forecast
from .hashing import array_hash, content_hash


class Knownness(str, Enum):
    KNOWN = "known"
    MISSING = "missing"                # genuinely absent in the world/view
    UNAVAILABLE_BUDGET = "unavailable_budget"  # not computed because of budget/failure


class ApproximationStatus(str, Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    TRUNCATED = "truncated"
    DEGRADED = "degraded"


class TruthStatus(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"
    ERROR = "error"


class Completeness(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"


@dataclass(frozen=True, slots=True)
class Evidence:
    """Common envelope. Subclasses add typed payloads."""

    evidence_id: str
    provenance: Provenance
    semantic_type: str
    approximation: ApproximationStatus = ApproximationStatus.EXACT
    cost: float = 0.0
    kind: str = "abstract"

    def canonical(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "semantic_type": self.semantic_type,
            "approximation": self.approximation.value,
            "provenance": self.provenance.canonical(),
            "payload": self._payload_canonical(),
        }

    def _payload_canonical(self) -> Any:
        return None

    def dense(self) -> np.ndarray:
        raise ContractViolation(f"{self.kind} evidence has no numeric lowering")


@dataclass(frozen=True, slots=True)
class FeatureEvidence(Evidence):
    """Scalar/vector/structured value with explicit knownness per element."""

    value: np.ndarray = field(default_factory=lambda: np.zeros(0))
    knownness: np.ndarray | None = None  # same shape as value, dtype=object/str or bool mask (True=known)
    interpretation: str = ""
    kind: str = "feature"

    def __post_init__(self) -> None:
        v = np.asarray(self.value, dtype=np.float64)
        object.__setattr__(self, "value", v)
        if self.knownness is None:
            object.__setattr__(self, "knownness", np.ones(v.shape, dtype=bool))
        else:
            k = np.asarray(self.knownness, dtype=bool)
            if k.shape != v.shape:
                raise ContractViolation("knownness mask must match value shape")
            object.__setattr__(self, "knownness", k)
        if np.any(~np.isfinite(v[self.knownness])):
            raise ContractViolation("known feature values must be finite")

    def dense(self) -> np.ndarray:
        """Unknown entries lower to 0 with the mask reported separately; the packer
        always carries ``knownness`` alongside so 0 is never mistaken for a fact."""
        return np.where(self.knownness, self.value, 0.0)

    def _payload_canonical(self) -> Any:
        return {"value": array_hash(self.value), "knownness": array_hash(self.knownness.astype(np.uint8)), "interpretation": self.interpretation}


@dataclass(frozen=True, slots=True)
class PredictiveEvidence(Evidence):
    forecast: Forecast | None = None
    kind: str = "predictive"

    def dense(self) -> np.ndarray:
        assert self.forecast is not None
        return self.forecast.dense()

    def _payload_canonical(self) -> Any:
        return self.forecast.canonical() if self.forecast is not None else None


@dataclass(frozen=True, slots=True)
class HypothesisEvidence(Evidence):
    """Alternatives plus weights with explicit weighting semantics.

    ``weight_semantics``: ``exact_posterior`` (weights are a posterior over a complete
    finite latent), ``normalized_beam`` (retained beam renormalised; omitted mass
    unknown unless stated), ``deterministic_assignment`` (one-hot decision), ``score``
    (unnormalised compatibility scores)."""

    alternatives: tuple[str, ...] = ()
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    weight_semantics: str = "score"
    omitted_mass: float | None = None
    kind: str = "hypothesis"

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=np.float64)
        if w.shape != (len(self.alternatives),):
            raise ContractViolation("weights must be one per alternative")
        if self.weight_semantics in ("exact_posterior", "approximate_posterior", "normalized_beam", "deterministic_assignment"):
            if np.any(w < -1e-12) or not np.isclose(w.sum(), 1.0, atol=1e-8):
                raise ContractViolation(f"{self.weight_semantics} weights must be a distribution; got sum={w.sum()}")
        if self.weight_semantics == "exact_posterior" and self.omitted_mass not in (None, 0.0):
            raise ContractViolation("an exact posterior cannot declare omitted mass")
        object.__setattr__(self, "weights", w)

    def dense(self) -> np.ndarray:
        return self.weights

    def _payload_canonical(self) -> Any:
        return {"alternatives": list(self.alternatives), "weights": array_hash(self.weights), "weight_semantics": self.weight_semantics, "omitted_mass": self.omitted_mass}


@dataclass(frozen=True, slots=True)
class ConstraintEvidence(Evidence):
    relation: str = ""
    hard: bool = False
    penalty_scale: float = 1.0
    satisfied: TruthStatus = TruthStatus.UNKNOWN
    kind: str = "constraint"

    def _payload_canonical(self) -> Any:
        return {"relation": self.relation, "hard": self.hard, "penalty_scale": self.penalty_scale, "satisfied": self.satisfied.value}


@dataclass(frozen=True, slots=True)
class BindingEvidence(Evidence):
    relation: str = ""
    bindings: tuple[tuple[str, str], ...] = ()
    supporting_records: tuple[str, ...] = ()
    completeness: Completeness = Completeness.PARTIAL
    kind: str = "binding"

    def _payload_canonical(self) -> Any:
        return {"relation": self.relation, "bindings": [list(b) for b in self.bindings], "support": list(self.supporting_records)}


@dataclass(frozen=True, slots=True)
class RetrievalEvidence(Evidence):
    """Returned record references plus a query trace. Non-exhaustive no-match is not false."""

    returned_ids: tuple[str, ...] = ()
    candidate_count: int = 0
    completeness: Completeness = Completeness.PARTIAL
    truncated: bool = False
    view_id: str = ""
    query_plan_hash: str = ""
    kind: str = "retrieval"

    def _payload_canonical(self) -> Any:
        return {"returned": list(self.returned_ids), "candidates": self.candidate_count, "completeness": self.completeness.value, "truncated": self.truncated, "view_id": self.view_id, "plan": self.query_plan_hash}


@dataclass(frozen=True, slots=True)
class ExecutionTrace(Evidence):
    """Computation/completion information. Never implicitly a forecast or a fact."""

    status: str = "ok"
    truth_status: TruthStatus | None = None
    completeness: Completeness | None = None
    details: dict[str, Any] = field(default_factory=dict)
    kind: str = "execution_trace"

    def _payload_canonical(self) -> Any:
        return {"status": self.status, "truth": self.truth_status.value if self.truth_status else None,
                "completeness": self.completeness.value if self.completeness else None, "details": self.details}


@dataclass(frozen=True, slots=True)
class QueryStatus(Evidence):
    status: str = "ok"  # ok | timeout | error | truncated
    message: str = ""
    kind: str = "query_status"

    def _payload_canonical(self) -> Any:
        return {"status": self.status, "message": self.message}


EvidenceFragment = tuple[Evidence, ...]


@dataclass(frozen=True, slots=True)
class EvidenceGraph:
    """Immutable logical view: evidence nodes keyed by id, dependency edges derived
    from provenance, and the memory views that were pinned for this prediction.
    ``with_fragment`` returns a new graph; nothing is mutated in place."""

    graph_id: str
    nodes: tuple[Evidence, ...] = ()
    memory_view_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ids = [n.evidence_id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ContractViolation("duplicate evidence ids in graph")

    def with_fragment(self, fragment: EvidenceFragment) -> "EvidenceGraph":
        if not fragment:
            return self
        known = {n.evidence_id for n in self.nodes}
        for e in fragment:
            for dep in e.provenance.dependencies:
                if dep not in known and not dep.startswith("observation:"):
                    raise ContractViolation(f"evidence {e.evidence_id} depends on unknown {dep}")
            known.add(e.evidence_id)
        return replace(self, nodes=self.nodes + tuple(fragment))

    def get(self, evidence_id: str) -> Evidence:
        for n in self.nodes:
            if n.evidence_id == evidence_id:
                return n
        raise KeyError(evidence_id)

    def ids(self) -> tuple[str, ...]:
        return tuple(n.evidence_id for n in self.nodes)

    def edges(self) -> tuple[tuple[str, str], ...]:
        return tuple((dep, n.evidence_id) for n in self.nodes for dep in n.provenance.dependencies)

    def version_hash(self) -> str:
        return content_hash({"graph_id": self.graph_id, "nodes": [n.canonical() for n in self.nodes], "views": list(self.memory_view_ids)})
