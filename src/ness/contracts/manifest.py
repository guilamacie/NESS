"""Whole-system identity: the PredictorManifest names everything that affects predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hashing import content_hash


@dataclass(frozen=True, slots=True)
class ComponentRef:
    node_id: str
    plugin_id: str
    plugin_version: str
    module_kind: str
    config_hash: str
    state_hash: str
    state_schema_id: str
    runtime: str

    def canonical(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class PredictorManifest:
    schema_version: str
    protocol_id: str
    arm_id: str
    composition_hash: str
    components: tuple[ComponentRef, ...]
    tasks: tuple[ComponentRef, ...]
    scenario: ComponentRef | None
    memory_snapshot_id: str | None
    inference_profile: str
    learning_profile: str
    credit_map_hash: str
    rng_state_hash: str
    dependency_lock: dict[str, str]
    parent_manifest_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def manifest_id(self) -> str:
        return content_hash(self.canonical())

    def canonical(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "arm_id": self.arm_id,
            "composition_hash": self.composition_hash,
            "components": [c.canonical() for c in self.components],
            "tasks": [c.canonical() for c in self.tasks],
            "scenario": self.scenario.canonical() if self.scenario else None,
            "memory_snapshot_id": self.memory_snapshot_id,
            "inference_profile": self.inference_profile,
            "learning_profile": self.learning_profile,
            "credit_map_hash": self.credit_map_hash,
            "rng_state_hash": self.rng_state_hash,
            "dependency_lock": dict(sorted(self.dependency_lock.items())),
            "parent_manifest_id": self.parent_manifest_id,
            "extra": self.extra,
        }
