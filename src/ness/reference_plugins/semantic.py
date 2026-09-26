"""``SimpleTemporalSemantics``: reads the raw series and (optionally) any neural state
port, and returns typed trend/volatility/regime-cue feature evidence with provenance
naming exactly the ports it consumed (T34)."""

from __future__ import annotations

import numpy as np

from ..contracts import (
    ContractViolation,
    Differentiability,
    FeatureEvidence,
    FieldRole,
    ModuleDescriptor,
    ModuleState,
    PortSchema,
    PortSpec,
    ProducerRef,
    Provenance,
)
from ..plugin_api.base import BaseModule
from ..plugin_api.module import ModuleOutputs, PortValue, RuntimeContext
from ..symbolic.operators import default_registry


class SimpleTemporalSemantics(BaseModule):
    plugin_id = "simple_temporal_semantics"
    plugin_version = "1.0.0"
    module_kind = "semantic_adapter"
    runtime = "numpy"
    state_schema_id = "ness.semantic.simple_temporal/1"

    def validate_config(self) -> None:
        self.window = int(self.config.get("window", 8))
        self.channels = int(self.config.get("channels", 2))
        self.use_neural = bool(self.config.get("use_neural", True))
        self.use_base_context = bool(self.config.get("use_base_context", False))
        self.cross_lags = tuple(int(k) for k in self.config.get("cross_lag_corr", ()))  # corr(channel0_t, channel1_{t-k})
        if self.cross_lags and self.channels < 2:
            raise ContractViolation("cross_lag_corr needs at least two channels (target, auxiliary)")
        self.use_events = bool(self.config.get("use_events", False))
        self.feature_dim = 2 * self.channels + len(self.cross_lags) + int(self.use_neural) + int(self.use_base_context) + int(self.use_events)
        self._prims = default_registry()

    def describe(self) -> ModuleDescriptor:
        inputs = [PortSpec("raw", PortSchema("dense", (None, self.channels)), "observation_series", FieldRole.OBSERVED, Differentiability.STOP,
                           accepts_semantic_types=("observation_series",))]
        if self.use_neural:
            inputs.append(PortSpec("neural", PortSchema("dense", ()), "neural_state", FieldRole.DERIVED, Differentiability.STOP, accepts_semantic_types=("*",)))
        if self.use_base_context:
            inputs.append(PortSpec("base_context", PortSchema("dense", ()), "neural_state", FieldRole.DERIVED, Differentiability.STOP, accepts_semantic_types=("*",)))
        if self.use_events:
            inputs.append(PortSpec("events", PortSchema("dense", ()), "event_indicator", FieldRole.KNOWN_FUTURE, Differentiability.STOP, accepts_semantic_types=("event_indicator",)))
        return self._descriptor(
            input_ports=tuple(inputs),
            output_ports=(PortSpec("features", PortSchema("feature", (self.feature_dim,)), "semantic_features", FieldRole.DERIVED, Differentiability.STOP),),
            capabilities={"features": self.feature_names()},
        )

    def feature_names(self) -> tuple[str, ...]:
        names = [f"slope_{d}" for d in range(self.channels)] + [f"volatility_{d}" for d in range(self.channels)]
        names += [f"xcorr_lag{k}" for k in self.cross_lags]
        if self.use_neural:
            names.append("neural_cue")
        if self.use_base_context:
            names.append("base_context_cue")
        if self.use_events:
            names.append("event_cue")
        return tuple(names)

    def initialize(self, rng: np.random.Generator) -> ModuleState:
        return ModuleState()

    def forward(self, inputs: dict[str, PortValue], state: ModuleState, ctx: RuntimeContext) -> ModuleOutputs:
        raw = np.asarray(inputs["raw"].dense(), dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != self.channels:
            raise ContractViolation(f"{self.plugin_id}: raw must be [T, {self.channels}]")
        slope = self._prims.get("window_slope").fn(raw, self.window)
        known = [np.full(self.channels, slope.knownness.value == "known")]
        vals = [slope.value]
        vol = np.std(np.diff(raw, axis=0), axis=0) if raw.shape[0] > 2 else np.zeros(self.channels)
        vals.append(vol)
        known.append(np.full(self.channels, raw.shape[0] > 2))
        for k in self.cross_lags:  # Pearson correlation between channel 0 at t and channel 1 at t-k over the window
            if raw.shape[0] > k + 3:
                a, b = raw[k:, 0], raw[: raw.shape[0] - k, 1]
                sa, sb = a.std(), b.std()
                c = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb)) if sa > 1e-12 and sb > 1e-12 else 0.0
                vals.append(np.array([c]))
                known.append(np.array([True]))
            else:
                vals.append(np.zeros(1))
                known.append(np.array([False]))
        used = ["raw"]
        for name, flag in (("neural", self.use_neural), ("base_context", self.use_base_context)):
            if flag:
                arr = np.asarray(inputs[name].dense(), dtype=np.float64)
                vals.append(np.array([float(np.tanh(arr.mean()))]))
                known.append(np.array([True]))
                used.append(name)
        if self.use_events:
            ev = np.asarray(inputs["events"].dense(), dtype=np.float64)
            vals.append(np.array([float(ev.mean())]))  # fraction of delivered steps with an active event (future window when wired to event_future)
            known.append(np.array([ev.size > 0]))
            used.append("events")
        value = np.concatenate(vals)
        kn = np.concatenate(known)
        deps = tuple(inputs[u].evidence_id for u in used)
        prov = Provenance(ProducerRef(ctx.node_id, self.plugin_id, self.plugin_version, "features"), deps, frozenset({FieldRole.DERIVED}),
                          None, tuple((u, inputs[u].evidence_id) for u in used), f"window={self.window}")
        fe = FeatureEvidence(f"{ctx.node_id}/features@{ctx.request.request_id}", prov, "semantic_features", value=value, knownness=kn,
                             interpretation=",".join(self.feature_names()))
        return ModuleOutputs(ports={"features": fe}, diagnostics={"used_inputs": used, "window": self.window}, cost=float(raw.size), used_inputs=tuple(used))
