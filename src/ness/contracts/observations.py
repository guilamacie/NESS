"""Observations: immutable, typed, role-tagged records with availability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .availability import FieldRole, AvailabilityCut, PREDICTION_TIME_ROLES
from .coordinates import CoordinateSchema
from .errors import AccessPolicyViolation, ContractViolation
from .hashing import array_hash


@dataclass(frozen=True, slots=True)
class Observation:
    """One observed record. ``payload`` is a numpy array (or scalar) whose axes are
    named by ``coordinates``. ``available_at`` is when this record became knowable."""

    observation_id: str
    field_name: str
    modality: str
    role: FieldRole
    coordinates: CoordinateSchema
    payload: Any
    available_at: int
    source: str = ""
    event_span: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.payload, np.ndarray):
            if self.payload.ndim != len(self.coordinates.axes):
                raise ContractViolation(
                    f"observation {self.observation_id}: payload rank {self.payload.ndim} "
                    f"does not match coordinate axes {self.coordinates.axes}"
                )
            self.payload.setflags(write=False)

    @property
    def array(self) -> np.ndarray:
        return np.asarray(self.payload)

    def canonical(self) -> dict:
        p = self.payload
        return {
            "observation_id": self.observation_id,
            "field_name": self.field_name,
            "modality": self.modality,
            "role": self.role.value,
            "coordinates": self.coordinates.schema_id,
            "payload": array_hash(np.asarray(p)) if isinstance(p, np.ndarray) else p,
            "available_at": self.available_at,
            "source": self.source,
            "event_span": list(self.event_span) if self.event_span else None,
        }


@dataclass(frozen=True, slots=True)
class ObservationField:
    """Schema-level declaration of a bundle field (used for compile-time access checks)."""

    name: str
    modality: str
    role: FieldRole
    semantic_type: str
    coordinates: CoordinateSchema
    shape: tuple[int | None, ...]


@dataclass(frozen=True, slots=True)
class ObservationBundle:
    """Immutable set of observations for one request. Withheld outcomes are NOT here;
    they live in ``TrainingOutcome``. A bundle may still contain ``outcome_only``
    fields for *training orchestration*; ``select`` removes them for prediction."""

    bundle_id: str
    observations: tuple[Observation, ...]
    origin: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [o.field_name for o in self.observations]
        if len(names) != len(set(names)):
            raise ContractViolation(f"bundle {self.bundle_id}: duplicate field names {names}")

    def field(self, name: str) -> Observation:
        for o in self.observations:
            if o.field_name == name:
                return o
        raise KeyError(name)

    def has(self, name: str) -> bool:
        return any(o.field_name == name for o in self.observations)

    def field_names(self) -> tuple[str, ...]:
        return tuple(o.field_name for o in self.observations)

    def select(self, policy: "AccessPolicy") -> "ObservationBundle":
        """Return the task-permitted view. Removes forbidden roles and anything not
        available at the policy's cut. Raises if a permitted field is not allowed."""
        kept = []
        for o in self.observations:
            if o.role not in policy.allowed_roles:
                continue
            if policy.allowed_fields is not None and o.field_name not in policy.allowed_fields:
                continue
            if not policy.cut.permits(o.available_at, o.source or None):
                continue
            kept.append(o)
        return ObservationBundle(
            bundle_id=f"{self.bundle_id}|{policy.policy_id}",
            observations=tuple(kept),
            origin=self.origin,
            metadata={**self.metadata, "view_of": self.bundle_id, "policy": policy.policy_id},
        )

    def assert_prediction_safe(self) -> None:
        for o in self.observations:
            if o.role not in PREDICTION_TIME_ROLES:
                raise AccessPolicyViolation(
                    f"field {o.field_name!r} with role {o.role.value} reached a prediction-time view"
                )

    def canonical(self) -> dict:
        return {"bundle_id": self.bundle_id, "origin": self.origin, "observations": [o.canonical() for o in self.observations]}


@dataclass(frozen=True, slots=True)
class AccessPolicy:
    """A task's information boundary for one request."""

    policy_id: str
    allowed_roles: frozenset[FieldRole]
    cut: AvailabilityCut
    allowed_fields: frozenset[str] | None = None
    allowed_memory_namespaces: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        forbidden = {FieldRole.OUTCOME_ONLY, FieldRole.EVALUATION_ORACLE} & set(self.allowed_roles)
        if forbidden:
            raise AccessPolicyViolation(f"policy {self.policy_id} may not allow roles {sorted(r.value for r in forbidden)}")

    def permits_role(self, role: FieldRole) -> bool:
        return role in self.allowed_roles

    def canonical(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "allowed_roles": sorted(r.value for r in self.allowed_roles),
            "cut": self.cut.canonical(),
            "allowed_fields": sorted(self.allowed_fields) if self.allowed_fields is not None else None,
            "allowed_memory_namespaces": sorted(self.allowed_memory_namespaces),
        }
