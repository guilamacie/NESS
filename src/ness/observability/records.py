"""Immutable prediction records and experience events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contracts import Forecast, PredictionResult, ScoreResult, TrainingOutcome, content_hash


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """Pre-outcome record. Never mutated; a later correction is another record."""

    record_id: str
    request_id: str
    origin: int
    manifest_id: str
    outputs: dict[str, Forecast]
    statuses: dict[str, str]
    evidence_graph_hash: str
    resolved_sources: dict[str, dict[str, list[str]]]
    node_versions: dict[str, tuple[str, str]]
    node_diagnostics: dict[str, dict[str, Any]]
    costs: dict[str, float]
    memory_view_ids: dict[str, str]
    policy_id: str
    group: dict[str, Any] = field(default_factory=dict)
    intervention: str | None = None

    @classmethod
    def from_result(cls, result: PredictionResult, origin: int, memory_view_ids: dict[str, str], group: dict[str, Any]) -> "PredictionRecord":
        d = result.diagnostics
        rid = content_hash({"request": result.request_id, "manifest": result.manifest_id, "outputs": result.canonical()["outputs"]})[:24]
        return cls(rid, result.request_id, origin, result.manifest_id, dict(result.outputs), dict(result.statuses), result.evidence_graph_hash,
                   d.get("resolved_sources", {}), d.get("node_versions", {}), d.get("node_diagnostics", {}), dict(result.cost), memory_view_ids,
                   next(iter(result.policies.values())).policy_id if result.policies else "", group)

    def canonical(self) -> dict:
        return {"record_id": self.record_id, "request_id": self.request_id, "manifest_id": self.manifest_id,
                "outputs": {k: v.canonical() for k, v in sorted(self.outputs.items())}, "evidence_graph_hash": self.evidence_graph_hash}


@dataclass(frozen=True, slots=True)
class ExperienceEvent:
    """Post-outcome, idempotent (event id derived from request + release sequence)."""

    event_id: str
    record_id: str
    request_id: str
    outcomes: dict[str, TrainingOutcome]
    scores: dict[str, ScoreResult]
    consumed_by_update: int | None = None

    @staticmethod
    def make_id(request_id: str, release_sequence: int) -> str:
        return content_hash({"request": request_id, "release": release_sequence})[:24]
