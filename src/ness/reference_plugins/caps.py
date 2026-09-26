"""Caps assembled from standard components: consumer (learned map from packed evidence
to a correction) + writer (baseline-preserving output). A cap is composition, not a
privileged opaque object: which ports it reads is visible in configuration, and the
writer/consumer are themselves swappable plugins."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import (
    ContractViolation,
    Differentiability,
    FieldRole,
    ModuleDescriptor,
    ModuleState,
    ParameterGroupSpec,
    PointForecast,
    PortSchema,
    PortSpec,
    QuantileForecast,
    horizon_schema,
    quantile_schema,
)
from ..plugin_api.base import BaseModule
from ..plugin_api.module import ModuleOutputs, PortValue, RuntimeContext
from ..plugin_api.registry import PluginRegistry
from ..runtimes.jax_runtime import jnp
from .writers_consumers import ConcatLinearConsumer, MonotoneQuantileWriter, PointResidualWriter

_LOCAL = {"concat_linear_consumer": ConcatLinearConsumer, "point_residual_writer": PointResidualWriter, "monotone_quantile_writer": MonotoneQuantileWriter}


def _resolve(spec: dict[str, Any] | str, registry: PluginRegistry | None, default: str) -> Any:
    if isinstance(spec, str):
        spec = {"plugin": spec}
    plugin = spec.get("plugin", default)
    cfg = dict(spec.get("config", {}))
    if registry is not None and registry.has(plugin):
        return registry.create(plugin, cfg)
    if plugin in _LOCAL:
        return _LOCAL[plugin](cfg)
    raise ContractViolation(f"unknown cap component {plugin!r}")


class _CapBase(BaseModule):
    module_kind = "cap"
    runtime = "jax"
    requires = ("jax",)
    forecast_type = "abstract"
    default_writer = ""

    def __init__(self, config: dict[str, Any] | None = None, registry: PluginRegistry | None = None) -> None:
        self._registry = registry
        super().__init__(config)

    def validate_config(self) -> None:
        c = self.config
        self.channels = tuple(c.get("channels", ("y0", "y1")))
        h = c.get("horizons", 4)
        self.horizons = tuple(range(1, int(h) + 1)) if isinstance(h, int) else tuple(int(x) for x in h)
        feats = c.get("features", {})
        if not isinstance(feats, dict) or not feats:
            raise ContractViolation(f"{self.plugin_id}: config.features must map input port -> feature dim (at least one)")
        # canonical layout: feature blocks are concatenated in sorted port-name order, so the
        # parameter layout never depends on how a config file happens to order its keys
        self.features: dict[str, int] = {k: int(feats[k]) for k in sorted(feats)}
        self.consumer = _resolve(c.get("consumer", "concat_linear_consumer"), self._registry, "concat_linear_consumer")
        self.writer = _resolve(c.get("writer", self.default_writer), self._registry, self.default_writer)
        if self.writer.forecast_type != self.forecast_type:
            raise ContractViolation(f"{self.plugin_id}: writer {self.writer.writer_id} writes {self.writer.forecast_type}, cap is {self.forecast_type}")
        self.in_dim = sum(self.features.values())

    def canonical_config(self) -> dict[str, Any]:
        d = super().canonical_config()
        d["_writer"] = getattr(self.writer, "writer_id", "?")
        d["_consumer"] = getattr(self.consumer, "consumer_id", "?")
        return d

    def _baseline_shape(self) -> tuple[int, ...]:  # pragma: no cover
        raise NotImplementedError

    def _feature_ports(self) -> tuple[PortSpec, ...]:
        return tuple(PortSpec(n, PortSchema("dense", (d,)), "packed_evidence", FieldRole.DERIVED, Differentiability.STOP, accepts_semantic_types=("*",))
                     for n, d in self.features.items())

    def initialize(self, rng: np.random.Generator) -> ModuleState:
        out_dim = self.writer.correction_dim(self._baseline_shape())
        return ModuleState(params={"consumer": self.consumer.init_params(self.in_dim, out_dim, rng)},
                           meta={"writer": self.writer.writer_id, "consumer": self.consumer.consumer_id, "baseline_preserving_at_zero": self.writer.is_baseline_preserving_at_zero()})

    def apply(self, params: dict[str, dict[str, Any]], dense_inputs: dict[str, Any], state: ModuleState, xp: Any) -> dict[str, Any]:
        x = xp.concatenate([xp.reshape(dense_inputs[n], (-1,)) for n in self.features])
        corr = self.consumer.apply(params["consumer"], x, xp)
        return {"forecast": self.writer.write(dense_inputs["baseline"], corr, xp), "correction": corr}

    def _dense_inputs(self, inputs: dict[str, PortValue]) -> dict[str, Any]:
        d = {n: jnp.asarray(np.asarray(inputs[n].dense(), dtype=np.float64).reshape(-1)) for n in self.features}
        for n in self.features:
            if d[n].shape != (self.features[n],):
                raise ContractViolation(f"{self.plugin_id}: feature port {n} delivered {tuple(d[n].shape)}, declared ({self.features[n]},)")
        d["baseline"] = jnp.asarray(inputs["baseline"].dense())
        return d


class ResidualPointCap(_CapBase):
    plugin_id = "residual_point_cap"
    plugin_version = "1.0.0"
    state_schema_id = "ness.cap.residual_point/1"
    forecast_type = "point"
    default_writer = "point_residual_writer"

    def _baseline_shape(self) -> tuple[int, ...]:
        return (len(self.horizons), len(self.channels))

    def describe(self) -> ModuleDescriptor:
        H, D = self._baseline_shape()
        target = horizon_schema(self.channels, self.horizons)
        return self._descriptor(
            input_ports=(PortSpec("baseline", PortSchema("point_forecast", (H, D)), "point_forecast", FieldRole.DERIVED, Differentiability.STOP,
                                  target.schema_id, accepts_semantic_types=("point_forecast",)),) + self._feature_ports(),
            output_ports=(PortSpec("forecast", PortSchema("point_forecast", (H, D)), "point_forecast", FieldRole.DERIVED, Differentiability.DIFFERENTIABLE, target.schema_id),
                          PortSpec("correction", PortSchema("dense", (H * D,)), "point_correction", FieldRole.DERIVED, Differentiability.DIFFERENTIABLE)),
            parameter_groups=(ParameterGroupSpec("consumer", "trainable", "jax", True, f"{self.consumer.consumer_id} parameters"),),
            capabilities={"writer": self.writer.writer_id, "consumer": self.consumer.consumer_id, "baseline_preserving_at_zero": True,
                          "unknown_features_lower_to": 0.0, "features": list(self.features)},
        )

    def forward(self, inputs: dict[str, PortValue], state: ModuleState, ctx: RuntimeContext) -> ModuleOutputs:
        base = inputs["baseline"].payload
        if not isinstance(base, PointForecast):
            raise ContractViolation(f"{self.plugin_id}: baseline must be a PointForecast, got {type(base).__name__}")
        outs = self.apply(state.params, self._dense_inputs(inputs), state, jnp)
        fc = PointForecast(np.asarray(outs["forecast"], dtype=np.float64), base.target, base.functional)
        return ModuleOutputs(ports={"forecast": fc, "correction": np.asarray(outs["correction"], dtype=np.float64)},
                             diagnostics={"correction_norm": float(np.linalg.norm(np.asarray(outs["correction"])))}, cost=float(self.in_dim))


class ResidualQuantileCap(_CapBase):
    plugin_id = "residual_quantile_cap"
    plugin_version = "1.0.0"
    state_schema_id = "ness.cap.residual_quantile/1"
    forecast_type = "quantile"
    default_writer = "monotone_quantile_writer"

    def validate_config(self) -> None:
        super().validate_config()
        self.levels = tuple(float(q) for q in self.config.get("levels", (0.1, 0.5, 0.9)))

    def _baseline_shape(self) -> tuple[int, ...]:
        return (len(self.horizons), len(self.channels), len(self.levels))

    def describe(self) -> ModuleDescriptor:
        H, D, Q = self._baseline_shape()
        target = quantile_schema(self.channels, self.horizons, self.levels)
        return self._descriptor(
            input_ports=(PortSpec("baseline", PortSchema("quantile_forecast", (H, D, Q)), "quantile_forecast", FieldRole.DERIVED, Differentiability.STOP,
                                  target.schema_id, accepts_semantic_types=("quantile_forecast",)),) + self._feature_ports(),
            output_ports=(PortSpec("forecast", PortSchema("quantile_forecast", (H, D, Q)), "quantile_forecast", FieldRole.DERIVED, Differentiability.DIFFERENTIABLE, target.schema_id),
                          PortSpec("correction", PortSchema("dense", (H * D * Q,)), "quantile_correction", FieldRole.DERIVED, Differentiability.DIFFERENTIABLE)),
            parameter_groups=(ParameterGroupSpec("consumer", "trainable", "jax", True, f"{self.consumer.consumer_id} parameters"),),
            capabilities={"writer": self.writer.writer_id, "consumer": self.consumer.consumer_id, "baseline_preserving_at_zero": True,
                          "monotone_preserving": True, "crossed_native_policy": "fail_explicitly", "features": list(self.features)},
        )

    def forward(self, inputs: dict[str, PortValue], state: ModuleState, ctx: RuntimeContext) -> ModuleOutputs:
        base = inputs["baseline"].payload
        if not isinstance(base, QuantileForecast):
            raise ContractViolation(f"{self.plugin_id}: baseline must be a QuantileForecast")
        if tuple(base.levels) != self.levels:
            raise ContractViolation(f"{self.plugin_id}: baseline levels {base.levels} != cap levels {self.levels}")
        if not base.is_monotone():
            raise ContractViolation("crossed native quantiles: this cap's declared policy is to fail; use a repair profile explicitly")
        outs = self.apply(state.params, self._dense_inputs(inputs), state, jnp)
        fc = QuantileForecast(np.asarray(outs["forecast"], dtype=np.float64), base.target, base.levels)
        return ModuleOutputs(ports={"forecast": fc, "correction": np.asarray(outs["correction"], dtype=np.float64)},
                             diagnostics={"correction_norm": float(np.linalg.norm(np.asarray(outs["correction"]))), "monotone": fc.is_monotone()}, cost=float(self.in_dim))
