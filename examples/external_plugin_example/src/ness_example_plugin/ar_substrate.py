"""A third lower-provider family: frozen lag-embedding AR model. Exposes the same port
contract (state/early, state/final, forecast/point) so it can replace either core
provider through configuration alone (T31/T33)."""

from __future__ import annotations

import numpy as np

from ness.contracts import (
    ContractViolation,
    Differentiability,
    FieldRole,
    ModuleDescriptor,
    ModuleState,
    ParameterGroupSpec,
    PointForecast,
    PortSchema,
    PortSpec,
    SubstrateCapabilities,
    horizon_schema,
)
from ness.plugin_api import BaseModule, ModuleOutputs, PortValue, RuntimeContext


class ToyLinearARSubstrate(BaseModule):
    plugin_id = "toy_linear_ar_substrate"
    plugin_version = "0.1.0"
    module_kind = "substrate"
    runtime = "numpy"
    state_schema_id = "example.substrate.toy_linear_ar/1"

    def validate_config(self) -> None:
        c = self.config
        self.lags = int(c.get("lags", 4))
        self.channels = tuple(c.get("channels", ("y0", "y1")))
        h = c.get("horizons", 4)
        self.horizons = tuple(range(1, int(h) + 1)) if isinstance(h, int) else tuple(int(x) for x in h)
        self.width = int(c.get("width", 16))
        self.seed = int(c.get("seed", 3))
        self.D, self.H = len(self.channels), len(self.horizons)
        if self.width < self.lags * self.D:
            raise ContractViolation("width must be >= lags * channels (early state is the lag embedding)")
        self.target_schema = horizon_schema(self.channels, self.horizons)

    def describe(self) -> ModuleDescriptor:
        W, D, H = self.width, self.D, self.H
        return self._descriptor(
            input_ports=(PortSpec("history", PortSchema("dense", (None, D)), "observation_series", FieldRole.OBSERVED, Differentiability.STOP, accepts_semantic_types=("observation_series",)),),
            output_ports=(
                PortSpec("state/early", PortSchema("dense", (None, W)), "neural_state", FieldRole.DERIVED, Differentiability.STOP, f"neural[ar_lags:{W}]"),
                PortSpec("state/final", PortSchema("dense", (None, W)), "neural_state", FieldRole.DERIVED, Differentiability.STOP, f"neural[ar_final:{W}]"),
                PortSpec("forecast/point", PortSchema("point_forecast", (H, D)), "point_forecast", FieldRole.DERIVED, Differentiability.STOP, self.target_schema.schema_id),
            ),
            parameter_groups=(ParameterGroupSpec("frozen", "frozen", "numpy", False, "AR coefficients + random projection"),),
            capabilities=SubstrateCapabilities("numpy", "frozen", ("state/early", "state/final"), ("forecast/point",), "stop", f"seed:{self.seed}", ("observation_series",)),
        )

    def _lag_matrix(self, x: np.ndarray) -> np.ndarray:
        T = x.shape[0]
        cols = [np.vstack([np.zeros((k, self.D)), x[: T - k]]) for k in range(self.lags)]
        emb = np.concatenate(cols, axis=1)  # [T, lags*D]
        if emb.shape[1] < self.width:
            emb = np.concatenate([emb, np.zeros((T, self.width - emb.shape[1]))], axis=1)
        return emb

    def initialize(self, rng: np.random.Generator) -> ModuleState:
        prng = np.random.default_rng(self.seed)
        proj = prng.normal(0, 1 / np.sqrt(self.width), (self.width, self.width))
        # self-pretraining on an independent AR process
        n = 400
        y = np.zeros((n, self.D))
        for t in range(1, n):
            y[t] = 0.7 * y[t - 1] + 0.1 * prng.normal(size=self.D)
        X, Y = [], []
        for o in range(self.lags, n - self.H):
            X.append(self._lag_matrix(y[: o])[-1])
            Y.append(y[o: o + self.H].reshape(-1))
        X, Y = np.asarray(X), np.asarray(Y)
        coef = np.linalg.solve(X.T @ X + 1e-2 * np.eye(X.shape[1]), X.T @ Y)
        return ModuleState(params={"frozen": {"proj": proj, "coef": coef}})

    def forward(self, inputs: dict[str, PortValue], state: ModuleState, ctx: RuntimeContext) -> ModuleOutputs:
        x = np.asarray(inputs["history"].dense(), dtype=np.float64)
        mu = x.mean(0)
        early = self._lag_matrix(x - mu)
        final = np.tanh(early @ state.params["frozen"]["proj"])
        point = (early[-1] @ state.params["frozen"]["coef"]).reshape(self.H, self.D) + mu
        return ModuleOutputs(ports={"state/early": early, "state/final": final, "forecast/point": PointForecast(point, self.target_schema, "mean")},
                             diagnostics={"lags": self.lags}, cost=float(x.size))
