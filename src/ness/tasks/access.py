"""Registered access policies. A task selects one; the core never branches on domain."""

from __future__ import annotations

from ..contracts import AccessPolicy, AvailabilityCut, FieldRole

AS_OF_ORIGIN_ROLES: frozenset[FieldRole] = frozenset({FieldRole.OBSERVED, FieldRole.KNOWN_FUTURE, FieldRole.DERIVED})


def as_of_origin_policy(task_id: str, origin: int, allowed_fields: frozenset[str] | None = None,
                        memory_namespaces: frozenset[str] = frozenset()) -> AccessPolicy:
    """Forecast-origin view: observed history and legitimately known-future covariates
    available at the origin. Outcome-only fields are excluded by construction."""
    return AccessPolicy(f"as_of_origin:{task_id}@{origin}", AS_OF_ORIGIN_ROLES, AvailabilityCut(origin), allowed_fields, memory_namespaces)
