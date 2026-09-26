"""A trivial deterministic reasoner: one-hot regime by thresholding one feature. It
declares ``deterministic_assignment`` semantics - it is NOT a posterior and says so."""

from __future__ import annotations

import numpy as np

from ness.contracts import (
    ApproximationStatus,
    Differentiability,
    ExecutionTrace,
    FeatureEvidence,
    FieldRole,
    HypothesisEvidence,
    ModuleDescriptor,
    ModuleState,
    PortSchema,
    PortSpec,
    ProbabilisticReasonerCapabilities,
    ProducerRef,
    Provenance,
)
from ness.plugin_api import BaseModule, ModuleOutputs, PortValue, RuntimeContext


class DeterministicThresholdReasoner(BaseModule):
    plugin_id = "deterministic_threshold_reasoner"
    plugin_version = "0.1.0"
    module_kind = "reasoner"
    runtime = "numpy"
    state_schema_id = "example.reasoner.threshold/1"

    def validate_config(self) -> None:
        self.states = tuple(self.config.get("states", ("calm", "volatile")))
        self.feature_ports = {k: int(v) for k, v in self.config.get("feature_ports", {"features": 1}).items()}
        self.index = int(self.config.get("feature_index", 0))
        self.threshold = float(self.config.get("threshold", 0.0))

    def describe(self) -> ModuleDescriptor:
        caps = ProbabilisticReasonerCapabilities("numpy", ("custom",), False, True, False, False, False, False, False, "stop", False, "deterministic",
                                                 f"categorical[{len(self.states)}]", "hypothesis", "1")
        return self._descriptor(
            input_ports=tuple(PortSpec(n, PortSchema("dense", (d,)), "features", FieldRole.DERIVED, Differentiability.STOP) for n, d in self.feature_ports.items()),
            output_ports=(PortSpec("posterior", PortSchema("hypothesis", (len(self.states),)), "regime_posterior", FieldRole.DERIVED, Differentiability.STOP),
                          PortSpec("moments", PortSchema("feature", (len(self.states),)), "posterior_moments", FieldRole.DERIVED, Differentiability.STOP)),
            capabilities=caps)

    def initialize(self, rng: np.random.Generator) -> ModuleState:
        return ModuleState()

    def forward(self, inputs: dict[str, PortValue], state: ModuleState, ctx: RuntimeContext) -> ModuleOutputs:
        feats = np.concatenate([np.asarray(inputs[n].dense(), dtype=np.float64).reshape(-1) for n in self.feature_ports])
        k = 1 if feats[self.index] > self.threshold else 0
        w = np.zeros(len(self.states))
        w[min(k, len(self.states) - 1)] = 1.0
        prov = Provenance(ProducerRef(ctx.node_id, self.plugin_id, self.plugin_version), tuple(pv.evidence_id for pv in inputs.values()),
                          frozenset({FieldRole.DERIVED}), None, tuple((n, pv.evidence_id) for n, pv in inputs.items()), "threshold")
        post = HypothesisEvidence(f"{ctx.node_id}/posterior@{ctx.request.request_id}", prov, "regime_posterior", ApproximationStatus.EXACT,
                                  alternatives=self.states, weights=w, weight_semantics="deterministic_assignment")
        mom = FeatureEvidence(f"{ctx.node_id}/moments@{ctx.request.request_id}", prov, "posterior_moments", value=w, interpretation="one-hot assignment")
        tr = ExecutionTrace(f"{ctx.node_id}/trace@{ctx.request.request_id}", prov, "reasoner_trace", status="ok", details={"inference_method": "threshold"})
        return ModuleOutputs(ports={"posterior": post, "moments": mom}, evidence=(tr,), diagnostics={"inference_method": "threshold"}, cost=1.0)
