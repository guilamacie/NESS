"""``jax_dense_workspace_cap``: a dense workspace cap in the NESS JAX backend whose
architecture (input -> tanh hidden layers -> linear readout, zero-initialised readout,
residual point writer) is *exactly* the graph the FabricPC ``fabricpc_residual_cap`` builds,
so the two backends can be compared on the same computation (parity fixtures)."""

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
    horizon_schema,
)
from ..plugin_api.base import BaseModule
from ..plugin_api.module import ModuleOutputs, PortValue, RuntimeContext
from ..runtimes.jax_runtime import jnp


class JaxDenseWorkspaceCap(BaseModule):
    plugin_id = "jax_dense_workspace_cap"
    plugin_version = "1.0.0"
    module_kind = "cap"
    runtime = "jax"
    requires = ("jax",)
    state_schema_id = "ness.cap.jax_dense_workspace/1"

    def validate_config(self) -> None:
        c = self.config
        self.channels = tuple(c.get("channels", ("y0", "y1")))
        h = c.get("horizons", 4)
        self.horizons = tuple(range(1, int(h) + 1)) if isinstance(h, int) else tuple(int(x) for x in h)
        feats = c.get("features", {})
        if not feats:
            raise ContractViolation(f"{self.plugin_id}: config.features required")
        self.features = {k: int(feats[k]) for k in sorted(feats)}
        self.hidden = tuple(int(x) for x in c.get("hidden", (16,)))
        self.activation = c.get("activation", "tanh")
        if self.activation not in ("tanh", "identity"):
            raise ContractViolation("activation must be tanh|identity")
        self.in_dim = sum(self.features.values())
        self.out_dim = len(self.horizons) * len(self.channels)

    def describe(self) -> ModuleDescriptor:
        H, D = len(self.horizons), len(self.channels)
        target = horizon_schema(self.channels, self.horizons)
        return self._descriptor(
            input_ports=(PortSpec("baseline", PortSchema("point_forecast", (H, D)), "point_forecast", FieldRole.DERIVED, Differentiability.STOP, target.schema_id,
                                  accepts_semantic_types=("point_forecast",)),)
            + tuple(PortSpec(n, PortSchema("dense", (d,)), "packed_evidence", FieldRole.DERIVED, Differentiability.STOP, accepts_semantic_types=("*",)) for n, d in self.features.items()),
            output_ports=(PortSpec("forecast", PortSchema("point_forecast", (H, D)), "point_forecast", FieldRole.DERIVED, Differentiability.DIFFERENTIABLE, target.schema_id),
                          PortSpec("correction", PortSchema("dense", (H * D,)), "point_correction", FieldRole.DERIVED, Differentiability.DIFFERENTIABLE)),
            parameter_groups=(ParameterGroupSpec("workspace", "trainable", "jax", True, "dense layers + zero-init readout"),),
            capabilities={"architecture": f"dense{list(self.hidden)}", "activation": self.activation, "writer": "point_residual", "baseline_preserving_at_zero": True,
                          "parity_twin": "fabricpc_residual_cap"},
        )

    def initialize(self, rng: np.random.Generator) -> ModuleState:
        dims = (self.in_dim,) + self.hidden + (self.out_dim,)
        p: dict[str, np.ndarray] = {}
        for i in range(len(dims) - 1):
            last = i == len(dims) - 2
            p[f"w{i}"] = np.zeros((dims[i], dims[i + 1])) if last else rng.normal(0, np.sqrt(1.0 / dims[i]), (dims[i], dims[i + 1]))
            p[f"b{i}"] = np.zeros(dims[i + 1])
        return ModuleState(params={"workspace": p}, meta={"layers": len(dims) - 1})

    def apply(self, params: dict[str, dict[str, Any]], dense_inputs: dict[str, Any], state: ModuleState, xp: Any) -> dict[str, Any]:
        p = params["workspace"]
        h = xp.concatenate([xp.reshape(dense_inputs[n], (-1,)) for n in self.features])
        n_layers = int(state.meta["layers"])
        for i in range(n_layers):
            h = h @ p[f"w{i}"] + p[f"b{i}"]
            if i < n_layers - 1 and self.activation == "tanh":
                h = xp.tanh(h)
        base = dense_inputs["baseline"]
        return {"forecast": base + xp.reshape(h, base.shape), "correction": h}

    def forward(self, inputs: dict[str, PortValue], state: ModuleState, ctx: RuntimeContext) -> ModuleOutputs:
        base = inputs["baseline"].payload
        if not isinstance(base, PointForecast):
            raise ContractViolation("baseline must be a PointForecast")
        dense = {n: jnp.asarray(np.asarray(inputs[n].dense(), dtype=np.float64).reshape(-1)) for n in self.features}
        dense["baseline"] = jnp.asarray(base.values)
        outs = self.apply(state.params, dense, state, jnp)
        return ModuleOutputs(ports={"forecast": PointForecast(np.asarray(outs["forecast"], dtype=np.float64), base.target, base.functional),
                                    "correction": np.asarray(outs["correction"], dtype=np.float64)},
                             diagnostics={"correction_norm": float(np.linalg.norm(np.asarray(outs["correction"])))}, cost=float(self.in_dim))
