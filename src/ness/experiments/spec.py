"""Experiment specification (``ness.experiment/3``): scenario, tasks, arms, protocol."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..composition import CompositionGraphSpec, parse_composition
from ..contracts import ValidationError, content_hash, require_schema

EXPERIMENT_SCHEMA = "ness.experiment/3"


@dataclass(frozen=True, slots=True)
class PluginSpec:
    plugin: str
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, d: Any, what: str) -> "PluginSpec":
        if isinstance(d, str):
            return cls(d, {})
        if not isinstance(d, dict) or "plugin" not in d:
            raise ValidationError(f"{what}: expected {{plugin, config}}")
        return cls(d["plugin"], dict(d.get("config", {})))

    def canonical(self) -> dict:
        return {"plugin": self.plugin, "config": self.config}


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    plugin: str
    config: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def canonical(self) -> dict:
        return {"task_id": self.task_id, "plugin": self.plugin, "config": self.config, "weight": self.weight}


@dataclass(frozen=True, slots=True)
class ArmSpec:
    arm_id: str
    composition: CompositionGraphSpec
    learning: dict[str, Any]
    inference: dict[str, Any]
    memory: dict[str, Any]
    seed: int
    description: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def canonical(self) -> dict:
        return {"arm_id": self.arm_id, "composition": self.composition.canonical(), "learning": self.learning, "inference": self.inference,
                "memory": self.memory, "seed": self.seed}


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    protocol_id: str
    scenario: PluginSpec
    tasks: tuple[TaskSpec, ...]
    arms: tuple[ArmSpec, ...]
    protocol: dict[str, Any]
    dependency_lock: dict[str, Any] = field(default_factory=dict)
    schema_version: str = EXPERIMENT_SCHEMA
    raw: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)  # {backend: auto|numpy|jax|fabricpc, platform, devices, x64, profile}

    def arm(self, arm_id: str) -> ArmSpec:
        for a in self.arms:
            if a.arm_id == arm_id:
                return a
        raise ValidationError(f"unknown arm {arm_id!r}; arms {[a.arm_id for a in self.arms]}")

    def canonical(self) -> dict:
        return {"schema_version": self.schema_version, "protocol_id": self.protocol_id, "scenario": self.scenario.canonical(),
                "tasks": [t.canonical() for t in self.tasks], "arms": [a.canonical() for a in self.arms], "protocol": self.protocol,
                "runtime": self.runtime}

    @property
    def protocol_hash(self) -> str:
        return content_hash(self.canonical())


def parse_arm(arm_id: str, d: dict[str, Any]) -> ArmSpec:
    if "composition" not in d:
        raise ValidationError(f"arm {arm_id}: missing composition")
    return ArmSpec(arm_id, parse_composition(d["composition"]), dict(d.get("learning", {})), dict(d.get("inference", {"profile": "direct"})),
                   dict(d.get("memory", {})), int(d.get("seed", 0)), d.get("description", ""), copy.deepcopy(d))


def parse_experiment(d: dict[str, Any]) -> ExperimentSpec:
    require_schema(d.get("schema_version", ""), EXPERIMENT_SCHEMA)
    if "protocol_id" not in d or "scenario" not in d or "tasks" not in d or "arms" not in d:
        raise ValidationError("experiment needs protocol_id, scenario, tasks, arms")
    tasks = tuple(TaskSpec(t["id"], t["plugin"], dict(t.get("config", {})), float(t.get("weight", 1.0))) for t in d["tasks"])
    arms_raw = d["arms"]
    if isinstance(arms_raw, dict):
        arms = tuple(parse_arm(k, v) for k, v in arms_raw.items())
    else:
        arms = tuple(parse_arm(a["id"], a) for a in arms_raw)
    return ExperimentSpec(d["protocol_id"], PluginSpec.parse(d["scenario"], "scenario"), tasks, arms, dict(d.get("protocol", {})),
                          dict(d.get("dependency_lock", {})), d.get("schema_version", EXPERIMENT_SCHEMA), copy.deepcopy(d), dict(d.get("runtime", {})))
