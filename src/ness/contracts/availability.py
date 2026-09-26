"""Availability, roles and provenance: the information boundary of a task.

Access is decided by *knowledge availability and role*, not event time alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FieldRole(str, Enum):
    OBSERVED = "observed"              # available at origin
    KNOWN_FUTURE = "known_future"      # legitimately known ahead (calendar, planned covariates)
    DERIVED = "derived"                # computed from permitted inputs; inherits their availability
    OUTCOME_ONLY = "outcome_only"      # withheld target; never enters target-free prediction
    EVALUATION_ORACLE = "evaluation_oracle"  # diagnostics only


PREDICTION_TIME_ROLES: frozenset[FieldRole] = frozenset({FieldRole.OBSERVED, FieldRole.KNOWN_FUTURE, FieldRole.DERIVED})


@dataclass(frozen=True, slots=True)
class AvailabilityCut:
    """As-of frontier. ``origin`` is the logical time index; ``source_frontiers`` maps
    asynchronous source streams to their last available sequence number."""

    origin: int
    source_frontiers: tuple[tuple[str, int], ...] = ()

    def permits(self, available_at: int, source: str | None = None, seq: int | None = None) -> bool:
        if available_at > self.origin:
            return False
        if source is not None and seq is not None:
            frontier = dict(self.source_frontiers).get(source)
            if frontier is not None and seq > frontier:
                return False
        return True

    def canonical(self) -> dict:
        return {"origin": self.origin, "source_frontiers": list(self.source_frontiers)}


@dataclass(frozen=True, slots=True)
class ProducerRef:
    """Who produced a value: plugin id and version plus the graph node it ran as."""

    node_id: str
    plugin_id: str
    plugin_version: str
    port: str = ""

    def canonical(self) -> dict:
        return {"node_id": self.node_id, "plugin_id": self.plugin_id, "plugin_version": self.plugin_version, "port": self.port}


@dataclass(frozen=True, slots=True)
class Provenance:
    """Lineage of a value: producer, dependencies (evidence ids), roles of its inputs,
    the maximum availability requirement of its dependency closure, and the resolved
    source selectors that fed it (recorded for T43 composition provenance)."""

    producer: ProducerRef
    dependencies: tuple[str, ...] = ()
    roles: frozenset[FieldRole] = field(default_factory=lambda: frozenset({FieldRole.DERIVED}))
    max_available_at: int | None = None
    resolved_sources: tuple[tuple[str, str], ...] = ()  # (input_port, "scheme://node/port")
    derivation: str = ""

    def canonical(self) -> dict:
        return {
            "producer": self.producer.canonical(),
            "dependencies": list(self.dependencies),
            "roles": sorted(r.value for r in self.roles),
            "max_available_at": self.max_available_at,
            "resolved_sources": [list(x) for x in self.resolved_sources],
            "derivation": self.derivation,
        }
