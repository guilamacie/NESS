"""Tiny frozen substrates (numpy runtime, forward only, explicit stop boundary).

Both providers "pretrain" themselves deterministically at ``initialize``: they generate
an independent synthetic series from their own seed (never the scenario's data), encode
it, and ridge-fit a linear readout for the baseline forecast. Their parameters then
live in the frozen group so the whole-system manifest identifies the provider by state
hash exactly as it would an external checkpoint digest.
"""

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
    SubstrateCapabilities,
    content_hash,
    horizon_schema,
    quantile_schema,
)
from ..plugin_api.base import BaseModule
from ..plugin_api.module import ModuleOutputs, PortValue, RuntimeContext

_Z = {0.01: -2.3263, 0.05: -1.6449, 0.1: -1.2816, 0.25: -0.6745, 0.5: 0.0, 0.75: 0.6745, 0.9: 1.2816, 0.95: 1.6449, 0.99: 2.3263}


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


class _FrozenSubstrateBase(BaseModule):
    module_kind = "substrate"
    runtime = "numpy"
    family = "abstract"

    def validate_config(self) -> None:
        c = self.config
        self.width = int(c.get("width", 16))
        self.channels = tuple(c.get("channels", ("y0", "y1")))
        h = c.get("horizons", 4)
        self.horizons = tuple(range(1, int(h) + 1)) if isinstance(h, int) else tuple(int(x) for x in h)
        self.seed = int(c.get("seed", 7))
        self.pretrain_steps = int(c.get("pretrain_steps", 400))
        self.context = int(c.get("context", 32))
        self.max_context = int(c.get("max_context", 128))
        self.quantile_levels = tuple(float(q) for q in c.get("quantile_levels", ()))
        for q in self.quantile_levels:
            if q not in _Z:
                raise ContractViolation(f"quantile level {q} not in the declared z-table {sorted(_Z)}")
        self.D, self.H = len(self.channels), len(self.horizons)
        fc = c.get("forecast_channels")
        self.forecast_channels = tuple(fc) if fc else self.channels
        unknown = set(self.forecast_channels) - set(self.channels)
        if unknown:
            raise ContractViolation(f"forecast_channels {sorted(unknown)} are not input channels {self.channels}")
        self.forecast_idx = np.array([self.channels.index(ch) for ch in self.forecast_channels])
        self.D_out = len(self.forecast_channels)
        self.target_schema = horizon_schema(self.forecast_channels, self.horizons)
        self.quantile_schema = quantile_schema(self.forecast_channels, self.horizons, self.quantile_levels) if self.quantile_levels else None

    # ---- to implement per family ------------------------------------------------
    def _init_encoder(self, rng: np.random.Generator) -> dict[str, np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    def _encode(self, p: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        """x: standardized [T, D] -> (early [T, W], final [T, W])"""
        raise NotImplementedError

    # ---- common -------------------------------------------------------------------
    def describe(self) -> ModuleDescriptor:
        D, W, H, Do = self.D, self.width, self.H, self.D_out
        outputs = [
            PortSpec("state/early", PortSchema("dense", (None, W)), "neural_state", FieldRole.DERIVED, Differentiability.STOP, f"neural[{self.family}:{W}]"),
            PortSpec("state/final", PortSchema("dense", (None, W)), "neural_state", FieldRole.DERIVED, Differentiability.STOP, f"neural[{self.family}:{W}]"),
            PortSpec("forecast/point", PortSchema("point_forecast", (H, Do)), "point_forecast", FieldRole.DERIVED, Differentiability.STOP, self.target_schema.schema_id),
        ]
        forecast_ports = ["forecast/point"]
        if self.quantile_schema is not None:
            outputs.append(PortSpec("forecast/quantiles", PortSchema("quantile_forecast", (H, Do, len(self.quantile_levels))), "quantile_forecast",
                                    FieldRole.DERIVED, Differentiability.STOP, self.quantile_schema.schema_id))
            forecast_ports.append("forecast/quantiles")
        caps = SubstrateCapabilities("numpy", "frozen", ("state/early", "state/final"), tuple(forecast_ports), "stop",
                                     weights_digest=f"seed:{self.seed}", accepted_observation_types=("observation_series",),
                                     hidden_state_support="verified", batching="single_request")
        return self._descriptor(
            input_ports=(PortSpec("history", PortSchema("dense", (None, D)), "observation_series", FieldRole.OBSERVED, Differentiability.STOP,
                                  accepts_semantic_types=("observation_series",)),),
            output_ports=tuple(outputs),
            parameter_groups=(ParameterGroupSpec("frozen", "frozen", "numpy", False, "pretrained frozen provider weights"),),
            capabilities=caps,
        )

    def initialize(self, rng: np.random.Generator) -> ModuleState:
        prng = np.random.default_rng(self.seed)  # provider identity comes from its own seed, not the run seed
        p = self._init_encoder(prng)
        # --- deterministic self-pretraining on an independent synthetic process ------
        n = self.pretrain_steps + self.context + self.H
        phase = 2 * np.pi * (np.arange(n) % 7) / 7
        lvl, ys = 0.0, np.zeros((n, self.D))
        for t in range(n):
            lvl = 0.85 * lvl + 0.3 * prng.normal()
            ys[t, 0] = lvl + 0.5 * np.sin(phase[t])
            for d in range(1, self.D):
                ys[t, d] = 0.4 * (ys[t - 1, 0] if t else 0.0) + 0.2 * prng.normal()
        feats, targets = [], []
        for o in range(self.context, n - self.H):
            win = ys[o - self.context: o]
            mu, sd = win.mean(0), win.std(0) + 1e-6
            _, final = self._encode(p, (win - mu) / sd)
            feats.append(final[-1])
            targets.append(((ys[o: o + self.H] - mu) / sd)[:, self.forecast_idx].reshape(-1))
        F = np.asarray(feats)
        Y = np.asarray(targets)
        lam = 1e-2
        A = F.T @ F + lam * np.eye(F.shape[1])
        Wout = np.linalg.solve(A, F.T @ Y)
        bout = (Y - F @ Wout).mean(0)
        resid = Y - (F @ Wout + bout)
        p["readout_w"], p["readout_b"] = Wout, bout
        p["residual_scale"] = resid.std(0) + 1e-6  # standardized units, per (h, d)
        return ModuleState(params={"frozen": p}, meta={"family": self.family, "pretrain_steps": self.pretrain_steps})

    def forward(self, inputs: dict[str, PortValue], state: ModuleState, ctx: RuntimeContext) -> ModuleOutputs:
        x = np.asarray(inputs["history"].dense(), dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.D:
            raise ContractViolation(f"{self.plugin_id}: history must be [T, {self.D}], got {x.shape}")
        if x.shape[0] > self.max_context:
            raise ContractViolation(f"{self.plugin_id}: context {x.shape[0]} exceeds capacity bucket {self.max_context}; no silent truncation")
        p = state.params["frozen"]
        mu, sd = x.mean(0), x.std(0) + 1e-6
        early, final = self._encode(p, (x - mu) / sd)
        z = final[-1] @ p["readout_w"] + p["readout_b"]
        point = z.reshape(self.H, self.D_out) * sd[self.forecast_idx] + mu[self.forecast_idx]
        ports: dict[str, Any] = {
            "state/early": early, "state/final": final,
            "forecast/point": PointForecast(point, self.target_schema, "mean"),
        }
        if self.quantile_schema is not None:
            zs = np.array([_Z[q] for q in self.quantile_levels])
            scale = p["residual_scale"].reshape(self.H, self.D_out) * sd[self.forecast_idx]
            qv = point[..., None] + zs * scale[..., None]
            ports["forecast/quantiles"] = QuantileForecast(qv, self.quantile_schema, self.quantile_levels)
        return ModuleOutputs(ports=ports, diagnostics={"context": int(x.shape[0]), "normalization": "per_window_history_standardization"}, cost=float(x.shape[0] * self.width))


class ToyFrozenTransformer(_FrozenSubstrateBase):
    plugin_id = "toy_frozen_transformer"
    plugin_version = "1.0.0"
    state_schema_id = "ness.substrate.toy_frozen_transformer/1"
    family = "transformer"

    def validate_config(self) -> None:
        super().validate_config()
        self.heads = int(self.config.get("heads", 2))
        self.layers = int(self.config.get("layers", 2))
        if self.width % self.heads:
            raise ContractViolation("width must be divisible by heads")
        if self.layers < 2:
            raise ContractViolation("toy_frozen_transformer needs >= 2 layers to expose distinct early/final states")

    def _init_encoder(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        W, D = self.width, self.D
        s = 1.0 / np.sqrt(W)
        p = {"embed_w": rng.normal(0, 1 / np.sqrt(D), (D, W)), "embed_b": np.zeros(W)}
        pos = np.arange(self.max_context)[:, None]
        div = np.exp(np.arange(0, W, 2) * (-np.log(10000.0) / W))
        pe = np.zeros((self.max_context, W))
        pe[:, 0::2], pe[:, 1::2] = np.sin(pos * div), np.cos(pos * div)
        p["pos"] = pe
        for l in range(self.layers):
            for nm in ("wq", "wk", "wv", "wo"):
                p[f"l{l}_{nm}"] = rng.normal(0, s, (W, W))
            p[f"l{l}_w1"], p[f"l{l}_b1"] = rng.normal(0, s, (W, 2 * W)), np.zeros(2 * W)
            p[f"l{l}_w2"], p[f"l{l}_b2"] = rng.normal(0, 1 / np.sqrt(2 * W), (2 * W, W)), np.zeros(W)
            p[f"l{l}_ln1_g"], p[f"l{l}_ln1_b"] = np.ones(W), np.zeros(W)
            p[f"l{l}_ln2_g"], p[f"l{l}_ln2_b"] = np.ones(W), np.zeros(W)
        return p

    def _encode(self, p: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T, W, Hh = x.shape[0], self.width, self.heads
        h = x @ p["embed_w"] + p["embed_b"] + p["pos"][:T]
        mask = np.tril(np.ones((T, T), dtype=bool))
        early = None
        for l in range(self.layers):
            a = _layer_norm(h, p[f"l{l}_ln1_g"], p[f"l{l}_ln1_b"])
            q, k, v = a @ p[f"l{l}_wq"], a @ p[f"l{l}_wk"], a @ p[f"l{l}_wv"]
            dh = W // Hh
            out = np.zeros_like(a)
            for hd in range(Hh):
                sl = slice(hd * dh, (hd + 1) * dh)
                sc = q[:, sl] @ k[:, sl].T / np.sqrt(dh)
                sc = np.where(mask, sc, -1e9)
                sc = sc - sc.max(-1, keepdims=True)
                w = np.exp(sc)
                w /= w.sum(-1, keepdims=True)
                out[:, sl] = w @ v[:, sl]
            h = h + out @ p[f"l{l}_wo"]
            f = _layer_norm(h, p[f"l{l}_ln2_g"], p[f"l{l}_ln2_b"])
            h = h + _relu(f @ p[f"l{l}_w1"] + p[f"l{l}_b1"]) @ p[f"l{l}_w2"] + p[f"l{l}_b2"]
            if l == 0:
                early = h.copy()
        assert early is not None
        return early, h


class ToyFrozenConv(_FrozenSubstrateBase):
    """A second provider family: causal 1-D convolution stack."""

    plugin_id = "toy_frozen_conv"
    plugin_version = "1.0.0"
    state_schema_id = "ness.substrate.toy_frozen_conv/1"
    family = "conv"

    def validate_config(self) -> None:
        super().validate_config()
        self.kernel = int(self.config.get("kernel", 3))
        self.layers = int(self.config.get("layers", 2))
        if self.layers < 2:
            raise ContractViolation("toy_frozen_conv needs >= 2 layers")

    def _init_encoder(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        W, D, K = self.width, self.D, self.kernel
        p: dict[str, np.ndarray] = {}
        cin = D
        for l in range(self.layers):
            p[f"c{l}_w"] = rng.normal(0, 1 / np.sqrt(K * cin), (K, cin, W))
            p[f"c{l}_b"] = np.zeros(W)
            cin = W
        return p

    def _encode(self, p: dict[str, np.ndarray], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        K = self.kernel
        h = x
        early = None
        for l in range(self.layers):
            w, b = p[f"c{l}_w"], p[f"c{l}_b"]
            pad = np.concatenate([np.zeros((K - 1, h.shape[1])), h], axis=0)
            out = np.zeros((h.shape[0], w.shape[2]))
            for k in range(K):
                out += pad[k: k + h.shape[0]] @ w[k]
            h = _relu(out + b)
            if l == 0:
                early = h.copy()
        assert early is not None
        return early, h
