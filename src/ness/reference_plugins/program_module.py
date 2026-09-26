"""``TypedProgram``: a composition node that executes one ``ProgramIR`` with the reference
interpreter. Bindings map program inputs to node input ports or declared constants."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import (
    ContractViolation,
    Differentiability,
    ExecutionTrace,
    FeatureEvidence,
    FieldRole,
    Knownness,
    ModuleDescriptor,
    ModuleState,
    PortSchema,
    PortSpec,
    ProducerRef,
    Provenance,
    TruthStatus,
    ApproximationStatus,
)
from ..plugin_api.base import BaseModule
from ..plugin_api.module import ModuleOutputs, PortValue, RuntimeContext
from ..symbolic import ProgramIR, ReferenceInterpreter


class TypedProgram(BaseModule):
    plugin_id = "typed_program"
    plugin_version = "1.0.0"
    module_kind = "program"
    runtime = "host"
    state_schema_id = "ness.program_node/1"

    def validate_config(self) -> None:
        if "program" not in self.config:
            raise ContractViolation("typed_program requires config.program (ness.program/1 IR)")
        self.program = ProgramIR.from_dict(self.config["program"])
        self.port_bindings: dict[str, dict[str, Any]] = dict(self.config.get("port_bindings", {}))  # program input -> {port, dim}
        self.constants: dict[str, Any] = dict(self.config.get("constants", {}))
        missing = set(self.program.inputs) - set(self.port_bindings) - set(self.constants)
        if missing:
            raise ContractViolation(f"typed_program: program inputs {sorted(missing)} have neither a port binding nor a constant")
        self.output_dim = int(self.config.get("output_dim", 1))
        self.interpreter = ReferenceInterpreter()

    def describe(self) -> ModuleDescriptor:
        inputs = tuple(
            PortSpec(b["port"], PortSchema("dense", tuple(b.get("shape", (None, int(b.get("dim", 1)))))), b.get("semantic_type", "observation_series"),
                     FieldRole.OBSERVED, Differentiability.STOP, accepts_semantic_types=tuple(b.get("accepts", ("*",))))
            for b in self.port_bindings.values()
        )
        return self._descriptor(
            input_ports=inputs,
            output_ports=(PortSpec("value", PortSchema("feature", (self.output_dim,)), "program_feature", FieldRole.DERIVED, Differentiability.STOP),),
            capabilities={"program_hash": self.program.program_hash, "semantics": self.program.semantics, "interpreter": self.interpreter.interpreter_id},
        )

    def initialize(self, rng: np.random.Generator) -> ModuleState:
        return ModuleState(meta={"program_hash": self.program.program_hash, "interpreter_version": self.interpreter.interpreter_version})

    def forward(self, inputs: dict[str, PortValue], state: ModuleState, ctx: RuntimeContext) -> ModuleOutputs:
        bindings: dict[str, Any] = dict(self.constants)
        for pin, b in self.port_bindings.items():
            bindings[pin] = np.asarray(inputs[b["port"]].dense(), dtype=np.float64)
        res = self.interpreter.execute(self.program, bindings, None, ctx.budget)
        producer = ProducerRef(ctx.node_id, self.plugin_id, self.plugin_version, "value")
        deps = tuple(pv.evidence_id for pv in inputs.values())
        prov = Provenance(producer, deps, frozenset({FieldRole.DERIVED}), None, tuple((b["port"], inputs[b["port"]].evidence_id) for b in self.port_bindings.values()),
                          f"program:{self.program.name}")
        if res.status == "ok" and res.knownness == Knownness.KNOWN and res.value is not None:
            val = np.asarray(res.value, dtype=np.float64).reshape(-1)
            if val.shape != (self.output_dim,):
                raise ContractViolation(f"typed_program {self.program.name}: value shape {val.shape} != declared ({self.output_dim},)")
            known = np.ones(self.output_dim, dtype=bool)
        else:
            val, known = np.zeros(self.output_dim), np.zeros(self.output_dim, dtype=bool)
        approx = ApproximationStatus.EXACT if res.status == "ok" else ApproximationStatus.DEGRADED
        fe = FeatureEvidence(f"{ctx.node_id}/value@{ctx.request.request_id}", prov, "program_feature", approx, value=val, knownness=known,
                             interpretation=f"{self.program.name} ({self.program.output})")
        trace = ExecutionTrace(f"{ctx.node_id}/trace@{ctx.request.request_id}", prov, "program_trace", approx, status=res.status,
                               truth_status=res.truth_status, completeness=res.completeness,
                               details={"fuel_used": res.fuel_used, "knownness": res.knownness.value, "trace": list(res.trace)[-4:], "error": res.error})
        return ModuleOutputs(ports={"value": fe}, evidence=(trace,), diagnostics={"status": res.status, "fuel_used": res.fuel_used}, cost=float(res.fuel_used))
