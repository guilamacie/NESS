"""Trainable upper numerical modules in the reference differentiable runtime (JAX).

``apply`` is written against an array namespace so the same code runs eagerly and
under ``jax.grad``. "Transformer" is one implementation of the upper-module contract,
not the contract itself (addendum §4); ``TinyUpperMLP`` proves substitution."""

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
    PortSchema,
    PortSpec,
)
from ..plugin_api.base import BaseModule
from ..plugin_api.module import ModuleOutputs, PortValue, RuntimeContext
from ..runtimes.jax_runtime import jnp


def _ln(x: Any, g: Any, b: Any, xp: Any, eps: float = 1e-5) -> Any:
    mu = xp.mean(x, axis=-1, keepdims=True)
    var = xp.mean((x - mu) ** 2, axis=-1, keepdims=True)
    return (x - mu) / xp.sqrt(var + eps) * g + b


def _gelu(x: Any, xp: Any) -> Any:
    return 0.5 * x * (1.0 + xp.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


class _UpperBase(BaseModule):
    module_kind = "upper_module"
    runtime = "jax"
    requires = ("jax",)

    def validate_config(self) -> None:
        self.in_dim = int(self.config.get("in_dim", 16))
        self.model_dim = int(self.config.get("model_dim", 16))

    def _ports(self, with_sequence: bool) -> tuple[tuple[PortSpec, ...], tuple[PortSpec, ...]]:
        inputs = (PortSpec("main", PortSchema("dense", (None, self.in_dim)), "neural_state", FieldRole.DERIVED, Differentiability.STOP, accepts_semantic_types=("*",)),)
        outputs = [PortSpec("hidden", PortSchema("dense", (self.model_dim,)), "upper_hidden", FieldRole.DERIVED, Differentiability.DIFFERENTIABLE, f"upper[{self.plugin_id}:{self.model_dim}]")]
        if with_sequence:
            outputs.append(PortSpec("sequence", PortSchema("dense", (None, self.model_dim)), "upper_sequence", FieldRole.DERIVED, Differentiability.DIFFERENTIABLE, f"upper_seq[{self.plugin_id}:{self.model_dim}]"))
        return inputs, tuple(outputs)

    def forward(self, inputs: dict[str, PortValue], state: ModuleState, ctx: RuntimeContext) -> ModuleOutputs:
        x = np.asarray(inputs["main"].dense(), dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ContractViolation(f"{self.plugin_id}: main must be [T, {self.in_dim}], got {x.shape}")
        outs = self.apply(state.params, {"main": jnp.asarray(x)}, state, jnp)
        return ModuleOutputs(ports={k: np.asarray(v, dtype=np.float64) for k, v in outs.items()}, diagnostics={"T": int(x.shape[0])}, cost=float(x.size * self.model_dim))


class TinyUpperTransformer(_UpperBase):
    plugin_id = "tiny_upper_transformer"
    plugin_version = "1.0.0"
    state_schema_id = "ness.upper.tiny_transformer/1"

    def validate_config(self) -> None:
        super().validate_config()
        self.heads = int(self.config.get("heads", 2))
        self.ff_dim = int(self.config.get("ff_dim", 2 * self.model_dim))
        self.max_context = int(self.config.get("max_context", 128))
        if self.model_dim % self.heads:
            raise ContractViolation("model_dim must be divisible by heads")

    def describe(self) -> ModuleDescriptor:
        inputs, outputs = self._ports(with_sequence=True)
        return self._descriptor(input_ports=inputs, output_ports=outputs,
                                parameter_groups=(ParameterGroupSpec("upper", "trainable", "jax", True, "one pre-LN causal transformer block + projections"),),
                                capabilities={"architecture": "transformer", "blocks": 1, "heads": self.heads, "normalization": "pre_layer_norm", "masking": "causal"})

    def initialize(self, rng: np.random.Generator) -> ModuleState:
        M, F, I = self.model_dim, self.ff_dim, self.in_dim
        s = 1.0 / np.sqrt(M)
        p = {"w_in": rng.normal(0, 1 / np.sqrt(I), (I, M)), "b_in": np.zeros(M)}
        for nm in ("wq", "wk", "wv", "wo"):
            p[nm] = rng.normal(0, s, (M, M))
        p.update({"ln1_g": np.ones(M), "ln1_b": np.zeros(M), "ln2_g": np.ones(M), "ln2_b": np.zeros(M), "lnf_g": np.ones(M), "lnf_b": np.zeros(M),
                  "w1": rng.normal(0, s, (M, F)), "b1": np.zeros(F), "w2": rng.normal(0, 1 / np.sqrt(F), (F, M)), "b2": np.zeros(M)})
        pos = np.arange(self.max_context)[:, None]
        div = np.exp(np.arange(0, M, 2) * (-np.log(10000.0) / M))
        pe = np.zeros((self.max_context, M))
        pe[:, 0::2], pe[:, 1::2] = np.sin(pos * div), np.cos(pos * div)
        return ModuleState(params={"upper": p}, buffers={"pos": pe})

    def apply(self, params: dict[str, dict[str, Any]], dense_inputs: dict[str, Any], state: ModuleState, xp: Any) -> dict[str, Any]:
        p = params["upper"]
        x = dense_inputs["main"]
        T = x.shape[0]
        M, Hh = self.model_dim, self.heads
        dh = M // Hh
        h = x @ p["w_in"] + p["b_in"] + xp.asarray(state.buffers["pos"][:T])
        a = _ln(h, p["ln1_g"], p["ln1_b"], xp)
        q = xp.reshape(a @ p["wq"], (T, Hh, dh))
        k = xp.reshape(a @ p["wk"], (T, Hh, dh))
        v = xp.reshape(a @ p["wv"], (T, Hh, dh))
        scores = xp.einsum("thd,shd->hts", q, k) / np.sqrt(dh)
        mask = xp.tril(xp.ones((T, T)))
        scores = xp.where(mask > 0, scores, -1e9)
        scores = scores - xp.max(scores, axis=-1, keepdims=True)
        w = xp.exp(scores)
        w = w / xp.sum(w, axis=-1, keepdims=True)
        att = xp.reshape(xp.einsum("hts,shd->thd", w, v), (T, M))
        h = h + att @ p["wo"]
        f = _ln(h, p["ln2_g"], p["ln2_b"], xp)
        h = h + _gelu(f @ p["w1"] + p["b1"], xp) @ p["w2"] + p["b2"]
        seq = _ln(h, p["lnf_g"], p["lnf_b"], xp)
        return {"hidden": seq[-1], "sequence": seq}


class TinyUpperMLP(_UpperBase):
    plugin_id = "tiny_upper_mlp"
    plugin_version = "1.0.0"
    state_schema_id = "ness.upper.tiny_mlp/1"

    def validate_config(self) -> None:
        super().validate_config()
        self.hidden_dim = int(self.config.get("hidden_dim", 32))

    def describe(self) -> ModuleDescriptor:
        inputs, outputs = self._ports(with_sequence=False)
        return self._descriptor(input_ports=inputs, output_ports=outputs,
                                parameter_groups=(ParameterGroupSpec("upper", "trainable", "jax", True, "pooled-features MLP"),),
                                capabilities={"architecture": "mlp", "pooling": "mean+last", "normalization": "none", "masking": "none"})

    def initialize(self, rng: np.random.Generator) -> ModuleState:
        I, Hd, M = 2 * self.in_dim, self.hidden_dim, self.model_dim
        return ModuleState(params={"upper": {"w1": rng.normal(0, np.sqrt(2.0 / I), (I, Hd)), "b1": np.zeros(Hd),
                                             "w2": rng.normal(0, np.sqrt(1.0 / Hd), (Hd, M)), "b2": np.zeros(M)}})

    def apply(self, params: dict[str, dict[str, Any]], dense_inputs: dict[str, Any], state: ModuleState, xp: Any) -> dict[str, Any]:
        p = params["upper"]
        x = dense_inputs["main"]
        feats = xp.concatenate([xp.mean(x, axis=0), x[-1]])
        return {"hidden": xp.tanh(feats @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]}
