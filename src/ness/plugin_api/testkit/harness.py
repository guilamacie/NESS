"""Deterministic fixture harness for unit-testing plugins outside a full system."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import (
    AccessPolicy,
    AvailabilityCut,
    CostLedger,
    Differentiability,
    FieldRole,
    ModuleDescriptor,
    Observation,
    ObservationBundle,
    PortSchema,
    PortSpec,
    PredictionRequest,
    ProducerRef,
    Provenance,
    TaskQuery,
    horizon_schema,
    series_schema,
)
from ..module import PortValue, RuntimeContext

_DEFAULT_VARIABLE_DIM = 8


def synthetic_request(T: int = 32, D: int = 2, H: int = 4, origin: int | None = None, seed: int = 0,
                      channels: tuple[str, ...] | None = None, with_outcome_field: bool = True) -> PredictionRequest:
    """A self-contained rolling-origin request with role-tagged fields (no dataset plugin needed)."""
    rng = np.random.default_rng(seed)
    channels = channels or tuple(f"y{i}" for i in range(D))
    origin = origin if origin is not None else T
    y = np.cumsum(rng.normal(0, 0.3, size=(T + H, D)), axis=0)
    obs = [Observation("obs:history", "target_history", "timeseries", FieldRole.OBSERVED, series_schema(channels), y[:T], origin, "synthetic",
                       (origin - T, origin), {"semantic_type": "observation_series"})]
    if with_outcome_field:
        obs.append(Observation("obs:future", "target_future", "timeseries", FieldRole.OUTCOME_ONLY, horizon_schema(channels, tuple(range(1, H + 1))), y[T:],
                               origin + H, "synthetic", (origin, origin + H), {"semantic_type": "outcome"}))
    bundle = ObservationBundle(f"bundle:synthetic@{origin}", tuple(obs), origin)
    q = TaskQuery(f"synthetic@{origin}/task", "task", horizon_schema(channels, tuple(range(1, H + 1))))
    return PredictionRequest(f"synthetic@{origin}", bundle, (q,), origin)


def as_of_origin(request: PredictionRequest, memory_namespaces: frozenset[str] = frozenset({"episodes"})) -> AccessPolicy:
    return AccessPolicy("test:as_of_origin", frozenset({FieldRole.OBSERVED, FieldRole.KNOWN_FUTURE, FieldRole.DERIVED}), AvailabilityCut(request.origin), None, memory_namespaces)


def make_context(request: PredictionRequest | None = None, node_id: str = "node", seed: int = 0, memory_views: dict[str, Any] | None = None,
                 services: dict[str, Any] | None = None, mode: str = "predict") -> RuntimeContext:
    request = request or synthetic_request()
    policy = as_of_origin(request)
    permitted = request.bundle.select(policy)
    return RuntimeContext(request, permitted, policy, "test-manifest", np.random.default_rng(seed), CostLedger(), memory_views or {}, mode,
                          node_id=node_id, services=services or {})


def port_value(payload: Any, name: str = "x", semantic_type: str = "dense", kind: str = "dense", role: FieldRole = FieldRole.DERIVED,
               producer: str = "test_source", evidence_id: str | None = None, mask: np.ndarray | None = None) -> PortValue:
    shape = tuple(np.asarray(payload).shape) if isinstance(payload, np.ndarray) else ()
    spec = PortSpec(name, PortSchema(kind, shape), semantic_type, role, Differentiability.STOP)
    prov = Provenance(ProducerRef(producer, "test", "0.0.0", name), (), frozenset({role}), None, (), "test fixture")
    return PortValue(payload, spec, prov, evidence_id or f"{producer}/{name}@test", mask)


def random_inputs_for(descriptor: ModuleDescriptor, rng: np.random.Generator, variable_dim: int = _DEFAULT_VARIABLE_DIM,
                      include_optional: bool = True) -> dict[str, PortValue]:
    """Random dense inputs matching declared input port shapes (None -> ``variable_dim``)."""
    out: dict[str, PortValue] = {}
    for p in descriptor.input_ports:
        if p.optional and not include_optional:
            continue
        if p.schema.kind not in ("dense", "feature"):
            raise ValueError(f"random_inputs_for cannot fabricate a {p.schema.kind} payload for port {p.name}; supply it explicitly")
        shape = tuple(d if d is not None else variable_dim for d in p.schema.shape) or (variable_dim,)
        out[p.name] = port_value(rng.normal(size=shape), p.name, p.semantic_type, "dense", p.availability_role)
    return out
