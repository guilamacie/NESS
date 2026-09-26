"""Requests never contain their withheld targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .coordinates import CoordinateSchema
from .errors import ContractViolation
from .forecasts import Forecast
from .hashing import array_hash
from .observations import AccessPolicy, ObservationBundle


@dataclass(frozen=True, slots=True)
class TaskQuery:
    """What is to be predicted, never its answer."""

    query_id: str
    task_id: str
    target: CoordinateSchema
    metadata: dict[str, Any] = field(default_factory=dict)

    def canonical(self) -> dict:
        return {"query_id": self.query_id, "task_id": self.task_id, "target": self.target.schema_id}


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    request_id: str
    bundle: ObservationBundle
    queries: tuple[TaskQuery, ...]
    origin: int
    group: dict[str, Any] = field(default_factory=dict)  # series/source/pair ids for grouped analysis

    def __post_init__(self) -> None:
        if not self.queries:
            raise ContractViolation("a prediction request needs at least one task query")
        if self.bundle.origin != self.origin:
            raise ContractViolation("request origin must equal bundle origin")

    def canonical(self) -> dict:
        return {"request_id": self.request_id, "origin": self.origin, "bundle": self.bundle.canonical(), "queries": [q.canonical() for q in self.queries]}


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """Revealed targets for one query, with coordinate-wise validity and release info.
    The orchestrator may hold both request and outcome; free inference never sees this."""

    request_id: str
    query_id: str
    values: np.ndarray
    mask: np.ndarray  # True where the coordinate has matured/been released
    release_sequence: int
    available_at: int
    target: CoordinateSchema

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=np.float64)
        m = np.asarray(self.mask, dtype=bool)
        if v.shape != m.shape:
            raise ContractViolation("outcome mask must match values shape")
        object.__setattr__(self, "values", v)
        object.__setattr__(self, "mask", m)

    def canonical(self) -> dict:
        return {"request_id": self.request_id, "query_id": self.query_id, "values": array_hash(self.values), "mask": array_hash(self.mask.astype(np.uint8)), "release": self.release_sequence}


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """Loss numerators/denominators so aggregation can sum-then-divide once."""

    numerator: float
    denominator: float
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.numerator / self.denominator if self.denominator > 0 else float("nan")


@dataclass(frozen=True, slots=True)
class PredictionResult:
    request_id: str
    manifest_id: str
    outputs: dict[str, Forecast]  # query_id -> forecast
    statuses: dict[str, str]
    diagnostics: dict[str, Any]
    evidence_graph_hash: str
    cost: dict[str, float]
    policies: dict[str, AccessPolicy]

    def canonical(self) -> dict:
        return {
            "request_id": self.request_id,
            "manifest_id": self.manifest_id,
            "outputs": {k: v.canonical() for k, v in sorted(self.outputs.items())},
            "statuses": dict(sorted(self.statuses.items())),
            "evidence_graph_hash": self.evidence_graph_hash,
        }
