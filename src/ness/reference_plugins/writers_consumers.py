"""Writers and consumers: small pure components a cap is assembled from."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import ContractViolation


class PointResidualWriter:
    """mu_hat = mu0 + a; zero correction reproduces the native point forecast exactly."""

    writer_id = "point_residual_writer"
    writer_version = "1.0.0"
    forecast_type = "point"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.bound = self.config.get("bound")  # optional |a| <= bound via tanh

    def correction_dim(self, baseline_shape: tuple[int, ...]) -> int:
        return int(np.prod(baseline_shape))

    def write(self, baseline: Any, correction: Any, xp: Any) -> Any:
        a = xp.reshape(correction, baseline.shape)
        if self.bound is not None:
            a = float(self.bound) * xp.tanh(a / float(self.bound))
        return baseline + a

    def is_baseline_preserving_at_zero(self) -> bool:
        return True


class MonotoneQuantileWriter:
    """PDF §9.5: Q1 = Q0_1 + a, Qk = Q_{k-1} + (Q0_k - Q0_{k-1}) exp(d_k). Zero outputs
    reproduce the native grid; positive gap multipliers preserve ordering. A zero native
    gap cannot be opened (explicit capacity limitation)."""

    writer_id = "monotone_quantile_writer"
    writer_version = "1.0.0"
    forecast_type = "quantile"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.clip = float(self.config.get("log_gap_clip", 5.0))

    def correction_dim(self, baseline_shape: tuple[int, ...]) -> int:
        return int(np.prod(baseline_shape))  # a for level 1 plus d_k for k>1 == Q per coordinate

    def write(self, baseline: Any, correction: Any, xp: Any) -> Any:
        # Algebraically Q_k = Q0_1 + a + sum_{j<=k} gap_j exp(d_j); written as
        # Q0_k + a + cumsum(gap_j expm1(d_j)) so that a zero correction is *bitwise* the
        # native grid (expm1(0) == 0 exactly), as the exact-parity contract requires.
        c = xp.reshape(correction, baseline.shape)
        a = c[..., 0]
        d = xp.clip(c[..., 1:], -self.clip, self.clip)
        gaps = baseline[..., 1:] - baseline[..., :-1]
        extra = xp.cumsum(gaps * xp.expm1(d), axis=-1)
        zeros = xp.zeros_like(a)[..., None]
        return baseline + a[..., None] + xp.concatenate([zeros, extra], axis=-1)

    def is_baseline_preserving_at_zero(self) -> bool:
        return True


class ConcatLinearConsumer:
    """Zero-initialised linear map from packed features to the correction vector."""

    consumer_id = "concat_linear_consumer"
    consumer_version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.init = self.config.get("init", "zeros")
        self.hidden = self.config.get("hidden")  # optional hidden width -> 2-layer MLP, output layer zero-init

    def param_shapes(self, in_dim: int, out_dim: int) -> dict[str, tuple[int, ...]]:
        if self.hidden:
            h = int(self.hidden)
            return {"w1": (in_dim, h), "b1": (h,), "w2": (h, out_dim), "b2": (out_dim,)}
        return {"weight": (in_dim, out_dim), "bias": (out_dim,)}

    def init_params(self, in_dim: int, out_dim: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        shapes = self.param_shapes(in_dim, out_dim)
        if self.hidden:
            return {"w1": rng.normal(0, np.sqrt(2.0 / in_dim), shapes["w1"]), "b1": np.zeros(shapes["b1"]),
                    "w2": np.zeros(shapes["w2"]), "b2": np.zeros(shapes["b2"])}
        w = np.zeros(shapes["weight"]) if self.init == "zeros" else rng.normal(0, 1e-2, shapes["weight"])
        return {"weight": w, "bias": np.zeros(shapes["bias"])}

    def apply(self, params: dict[str, Any], x: Any, xp: Any) -> Any:
        if self.hidden:
            h = xp.tanh(x @ params["w1"] + params["b1"])
            return h @ params["w2"] + params["b2"]
        return x @ params["weight"] + params["bias"]


def make_writer(kind: str, config: dict[str, Any] | None = None) -> Any:
    table = {PointResidualWriter.writer_id: PointResidualWriter, MonotoneQuantileWriter.writer_id: MonotoneQuantileWriter}
    if kind not in table:
        raise ContractViolation(f"unknown writer {kind!r}")
    return table[kind](config)
