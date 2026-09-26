"""Capability records. Status is verified / unsupported / experimental, never implied."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .errors import UnsupportedCapability


class CapabilityStatus(str, Enum):
    VERIFIED = "verified"
    UNSUPPORTED = "unsupported"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True, slots=True)
class Capability:
    name: str
    status: CapabilityStatus
    evidence: str = ""  # contract-test artifact id

    def require_verified(self, context: str = "") -> None:
        if self.status != CapabilityStatus.VERIFIED:
            raise UnsupportedCapability(f"capability {self.name!r} is {self.status.value} {context}".strip())

    def canonical(self) -> dict:
        return {"name": self.name, "status": self.status.value, "evidence": self.evidence}


@dataclass(frozen=True, slots=True)
class CapabilitySet:
    items: tuple[Capability, ...] = ()

    def get(self, name: str) -> Capability:
        for c in self.items:
            if c.name == name:
                return c
        return Capability(name, CapabilityStatus.UNSUPPORTED, "not declared")

    def require(self, name: str, context: str = "") -> None:
        self.get(name).require_verified(context)

    def canonical(self) -> list:
        return [c.canonical() for c in self.items]


@dataclass(frozen=True, slots=True)
class SubstrateCapabilities:
    runtime: str
    mode: str  # frozen | frozen_external | trainable
    state_ports: tuple[str, ...]
    forecast_ports: tuple[str, ...]
    gradient_boundary: str  # stop | differentiable
    weights_digest: str = ""
    accepted_observation_types: tuple[str, ...] = ()
    hidden_state_support: str = "verified"  # verified | unverified | unsupported
    batching: str = "single_request"
    license_review: str = "not_required"

    def canonical(self) -> dict:
        return {k: getattr(self, k) if not isinstance(getattr(self, k), tuple) else list(getattr(self, k)) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class ProbabilisticReasonerCapabilities:
    runtime: str
    inference_methods: tuple[str, ...]
    exact_enumeration: bool
    supports_discrete_latents: bool
    supports_continuous_latents: bool
    supports_amortized_guide: bool
    supports_sampling: bool
    supports_log_prob: bool
    supports_differentiable_path: bool
    gradient_boundary: str
    supports_online_state: bool
    approximation: str  # exact | monte_carlo | variational | ...
    latent_schema: str = ""
    output_schema: str = ""
    serialization_version: str = "1"

    def canonical(self) -> dict:
        return {k: getattr(self, k) if not isinstance(getattr(self, k), tuple) else list(getattr(self, k)) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    backend_id: str
    revision: str
    capabilities: CapabilitySet = field(default_factory=CapabilitySet)

    def canonical(self) -> dict:
        return {"backend_id": self.backend_id, "revision": self.revision, "capabilities": self.capabilities.canonical()}


@dataclass(frozen=True, slots=True)
class ScenarioCapabilities:
    domain: str
    splits: tuple[str, ...]
    rolling_origin: bool
    outcome_release: str  # immediate | delayed | coordinate_wise
    observation_fields: tuple[str, ...]
    outcome_fields: tuple[str, ...]

    def canonical(self) -> dict:
        return {"domain": self.domain, "splits": list(self.splits), "rolling_origin": self.rolling_origin, "outcome_release": self.outcome_release, "observation_fields": list(self.observation_fields), "outcome_fields": list(self.outcome_fields)}


@dataclass(frozen=True, slots=True)
class MemoryCapabilities:
    backend: str
    isolation: str  # snapshot_copy | ...
    canonical_export: bool
    durability: str  # in_memory | journal | ...
    query_kinds: tuple[str, ...]
    supports_enumeration: bool = True

    def canonical(self) -> dict:
        return {"backend": self.backend, "isolation": self.isolation, "canonical_export": self.canonical_export, "durability": self.durability, "query_kinds": list(self.query_kinds), "supports_enumeration": self.supports_enumeration}
