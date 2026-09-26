"""Ports, module descriptors and parameter groups: the macro-composition vocabulary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .availability import FieldRole
from .errors import ContractViolation


class Differentiability:
    DIFFERENTIABLE = "differentiable"  # gradient may flow into this port's producer
    STOP = "stop"                      # explicit stop-gradient leaf
    NONE = "none"                      # non-numeric / symbolic payload


PORT_KINDS = ("dense", "point_forecast", "quantile_forecast", "feature", "hypothesis", "retrieval", "execution_trace")


@dataclass(frozen=True, slots=True)
class PortSchema:
    """Payload kind and (possibly partially known) shape. ``None`` dims are variable."""

    kind: str
    shape: tuple[int | None, ...] = ()
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.kind not in PORT_KINDS:
            raise ContractViolation(f"unknown port kind {self.kind!r}; expected one of {PORT_KINDS}")

    def feature_dim(self) -> int | None:
        return self.shape[-1] if self.shape else None

    def compatible_shape(self, other: "PortSchema") -> bool:
        if not self.shape or not other.shape:
            return True  # an empty declared shape means rank-agnostic ("any dense payload")
        if len(self.shape) != len(other.shape):
            return False
        return all(a is None or b is None or a == b for a, b in zip(self.shape, other.shape))

    def canonical(self) -> dict:
        return {"kind": self.kind, "shape": list(self.shape), "dtype": self.dtype}


@dataclass(frozen=True, slots=True)
class PortSpec:
    name: str
    schema: PortSchema
    semantic_type: str
    availability_role: FieldRole = FieldRole.DERIVED
    differentiability: str = Differentiability.STOP
    coordinate_schema_id: str | None = None
    prediction_space_id: str | None = None
    accepts_semantic_types: tuple[str, ...] = ("*",)  # for input ports
    optional: bool = False                            # input port may be left unconnected
    description: str = ""

    def accepts(self, semantic_type: str) -> bool:
        return "*" in self.accepts_semantic_types or semantic_type in self.accepts_semantic_types

    def canonical(self) -> dict:
        return {
            "name": self.name,
            "schema": self.schema.canonical(),
            "semantic_type": self.semantic_type,
            "availability_role": self.availability_role.value,
            "differentiability": self.differentiability,
            "coordinate_schema_id": self.coordinate_schema_id,
            "prediction_space_id": self.prediction_space_id,
            "accepts": list(self.accepts_semantic_types),
            "optional": self.optional,
        }


@dataclass(frozen=True, slots=True)
class ParameterGroupSpec:
    """A named set of parameters with declared mutability and runtime."""

    name: str
    mutability: str  # "frozen" | "trainable"
    runtime: str
    differentiable: bool
    description: str = ""

    def canonical(self) -> dict:
        return {"name": self.name, "mutability": self.mutability, "runtime": self.runtime, "differentiable": self.differentiable}


MODULE_KINDS = (
    "substrate", "upper_module", "semantic_adapter", "program", "reasoner",
    "memory_query", "cap", "consumer", "writer", "task", "scenario", "memory_store", "learning_rule",
)

# Which URI schemes may point at which producer kinds.
SCHEME_FOR_KIND = {
    "substrate": "substrate",
    "upper_module": "module",
    "semantic_adapter": "semantic",
    "program": "program",
    "reasoner": "reasoner",
    "memory_query": "memory",
    "cap": "module",
}


@dataclass(frozen=True, slots=True)
class ModuleDescriptor:
    plugin_id: str
    plugin_version: str
    module_kind: str
    runtime: str  # "numpy" | "jax" | "host" | ...
    input_ports: tuple[PortSpec, ...] = ()
    output_ports: tuple[PortSpec, ...] = ()
    parameter_groups: tuple[ParameterGroupSpec, ...] = ()
    capabilities: Any = None
    state_schema_id: str = ""
    contract_version: str = "ness.contracts/1"
    config_hash: str = ""
    requires: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.module_kind not in MODULE_KINDS:
            raise ContractViolation(f"unknown module kind {self.module_kind!r}")
        names = [p.name for p in self.output_ports]
        if len(names) != len(set(names)):
            raise ContractViolation(f"{self.plugin_id}: duplicate output port names")

    def input_port(self, name: str) -> PortSpec:
        for p in self.input_ports:
            if p.name == name:
                return p
        raise KeyError(name)

    def output_port(self, name: str) -> PortSpec:
        for p in self.output_ports:
            if p.name == name:
                return p
        raise KeyError(name)

    def trainable_groups(self) -> tuple[ParameterGroupSpec, ...]:
        return tuple(g for g in self.parameter_groups if g.mutability == "trainable")

    def canonical(self) -> dict:
        return {
            "plugin_id": self.plugin_id,
            "plugin_version": self.plugin_version,
            "module_kind": self.module_kind,
            "runtime": self.runtime,
            "input_ports": [p.canonical() for p in self.input_ports],
            "output_ports": [p.canonical() for p in self.output_ports],
            "parameter_groups": [g.canonical() for g in self.parameter_groups],
            "capabilities": self.capabilities.canonical() if hasattr(self.capabilities, "canonical") else self.capabilities,
            "state_schema_id": self.state_schema_id,
            "contract_version": self.contract_version,
            "config_hash": self.config_hash,
        }
