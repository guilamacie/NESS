"""The small, fixed macro-boundary operator vocabulary (addendum §3.4).

Boundary transforms: identity, linear, layer_norm, rms_norm, standardize, mean_pool,
last_step, flatten.  Merges: concat, add, gated_add, weighted_sum, select,
attention_pool, cross_attention.

Every op is written against an array namespace ``xp`` (numpy or jax.numpy) so the same
semantics run eagerly on host arrays and inside the differentiable runtime. Shape rules
are evaluated at compile time; learned ops declare their parameter shapes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import CompositionError

Shape = tuple[int | None, ...]

BOUNDARY_KINDS = ("identity", "linear", "layer_norm", "rms_norm", "standardize", "mean_pool", "last_step", "flatten")
MERGE_OPS = ("concat", "add", "gated_add", "weighted_sum", "select", "attention_pool", "cross_attention")


def _last(shape: Shape) -> int | None:
    return shape[-1] if shape else None


# ---------------------------------------------------------------- boundary: shapes
def boundary_out_shape(kind: str, params: dict[str, Any], shape: Shape) -> Shape:
    if kind not in BOUNDARY_KINDS:
        raise CompositionError(f"unknown boundary transform {kind!r}; allowed {BOUNDARY_KINDS}")
    if kind in ("identity", "layer_norm", "rms_norm", "standardize"):
        return shape
    if kind == "linear":
        if "out_dim" not in params:
            raise CompositionError("linear boundary transform requires out_dim")
        return tuple(shape[:-1]) + (int(params["out_dim"]),)
    if kind in ("mean_pool", "last_step"):
        axis = int(params.get("axis", 0))
        if len(shape) < 2:
            raise CompositionError(f"{kind} needs rank >= 2, got shape {shape}")
        return tuple(s for i, s in enumerate(shape) if i != axis)
    if kind == "flatten":
        if any(s is None for s in shape):
            return (None,)
        return (int(np.prod(shape)),)
    raise CompositionError(kind)


def boundary_is_learned(kind: str, params: dict[str, Any]) -> bool:
    if kind == "linear":
        return True
    if kind in ("layer_norm", "rms_norm"):
        return bool(params.get("affine", True))
    return False


def boundary_param_shapes(kind: str, params: dict[str, Any], in_shape: Shape) -> dict[str, tuple[int, ...]]:
    d = _last(in_shape)
    if boundary_is_learned(kind, params) and d is None:
        raise CompositionError(f"{kind} needs a known feature dimension; got shape {in_shape}")
    if kind == "linear":
        return {"weight": (int(d), int(params["out_dim"])), "bias": (int(params["out_dim"]),)}
    if kind in ("layer_norm", "rms_norm") and params.get("affine", True):
        return {"scale": (int(d),)} if kind == "rms_norm" else {"scale": (int(d),), "shift": (int(d),)}
    return {}


def init_boundary_params(kind: str, params: dict[str, Any], in_shape: Shape, rng: np.random.Generator) -> dict[str, np.ndarray]:
    shapes = boundary_param_shapes(kind, params, in_shape)
    out: dict[str, np.ndarray] = {}
    for name, shp in shapes.items():
        if name == "weight":
            fan_in = shp[0]
            init = params.get("init", "glorot")
            if init == "zeros":
                out[name] = np.zeros(shp)
            elif init == "identity":
                w = np.zeros(shp)
                n = min(shp)
                w[np.arange(n), np.arange(n)] = 1.0
                out[name] = w
            else:
                out[name] = rng.normal(0.0, np.sqrt(2.0 / (fan_in + shp[1])), size=shp)
        elif name in ("bias", "shift"):
            out[name] = np.zeros(shp)
        elif name == "scale":
            out[name] = np.ones(shp)
    return out


def apply_boundary(kind: str, params: dict[str, Any], arrays: dict[str, Any], x: Any, xp: Any) -> Any:
    if kind == "identity":
        return x
    if kind == "linear":
        return x @ arrays["weight"] + arrays["bias"]
    if kind == "layer_norm":
        eps = float(params.get("eps", 1e-5))
        mu = xp.mean(x, axis=-1, keepdims=True)
        var = xp.mean((x - mu) ** 2, axis=-1, keepdims=True)
        y = (x - mu) / xp.sqrt(var + eps)
        if params.get("affine", True):
            y = y * arrays["scale"] + arrays["shift"]
        return y
    if kind == "rms_norm":
        eps = float(params.get("eps", 1e-5))
        y = x / xp.sqrt(xp.mean(x**2, axis=-1, keepdims=True) + eps)
        if params.get("affine", True):
            y = y * arrays["scale"]
        return y
    if kind == "standardize":
        # Declared standardisation with fixed constants (never fitted on the target).
        mean = xp.asarray(params.get("mean", 0.0))
        scale = xp.asarray(params.get("scale", 1.0))
        return (x - mean) / scale
    if kind == "mean_pool":
        return xp.mean(x, axis=int(params.get("axis", 0)))
    if kind == "last_step":
        axis = int(params.get("axis", 0))
        return xp.take(x, x.shape[axis] - 1, axis=axis)
    if kind == "flatten":
        return xp.reshape(x, (-1,))
    raise CompositionError(f"unknown boundary transform {kind!r}")


# ------------------------------------------------------------------ merges: shapes
def merge_out_shape(op: str, params: dict[str, Any], shapes: tuple[Shape, ...]) -> Shape:
    if op not in MERGE_OPS:
        raise CompositionError(f"unknown merge operator {op!r}; allowed {MERGE_OPS}")
    if op == "select":
        idx = int(params.get("index", 0))
        if idx >= len(shapes):
            raise CompositionError(f"select index {idx} out of range for {len(shapes)} sources")
        return shapes[idx]
    if op == "concat":
        ranks = {len(s) for s in shapes}
        if len(ranks) != 1:
            raise CompositionError(f"concat sources must share rank; got {shapes}")
        for i in range(len(shapes[0]) - 1):
            dims = {s[i] for s in shapes if s[i] is not None}
            if len(dims) > 1:
                raise CompositionError(f"concat sources disagree on axis {i}: {shapes}")
        lasts = [_last(s) for s in shapes]
        total = None if any(d is None for d in lasts) else int(sum(lasts))  # type: ignore[arg-type]
        return tuple(shapes[0][:-1]) + (total,)
    if op in ("add", "gated_add", "weighted_sum"):
        base = shapes[0]
        for s in shapes[1:]:
            if len(s) != len(base) or any(a is not None and b is not None and a != b for a, b in zip(base, s)):
                raise CompositionError(f"{op} requires identical shapes; got {shapes}. Add an explicit linear projection.")
        return base
    if op == "attention_pool":
        if len(shapes) != 1 or len(shapes[0]) != 2:
            raise CompositionError("attention_pool takes exactly one [T, d] source")
        return (shapes[0][1],)
    if op == "cross_attention":
        if len(shapes) != 2 or len(shapes[0]) != 1 or len(shapes[1]) != 2:
            raise CompositionError("cross_attention takes a query [d_q] and a memory [T, d_kv]")
        return (shapes[0][0],)
    raise CompositionError(op)


def merge_is_learned(op: str) -> bool:
    return op in ("gated_add", "weighted_sum", "attention_pool", "cross_attention")


def merge_param_shapes(op: str, params: dict[str, Any], shapes: tuple[Shape, ...]) -> dict[str, tuple[int, ...]]:
    if op == "gated_add":
        d = _last(shapes[0])
        per_dim = bool(params.get("per_dim", False))
        return {"gate": (int(d),) if per_dim and d is not None else (1,)}
    if op == "weighted_sum":
        return {"logits": (len(shapes),)}
    if op == "attention_pool":
        return {"query": (int(shapes[0][1]),)}
    if op == "cross_attention":
        dq, dkv = int(shapes[0][0]), int(shapes[1][1])
        dm = int(params.get("model_dim", dq))
        return {"wq": (dq, dm), "wk": (dkv, dm), "wv": (dkv, dq)}
    return {}


def init_merge_params(op: str, params: dict[str, Any], shapes: tuple[Shape, ...], rng: np.random.Generator) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, shp in merge_param_shapes(op, params, shapes).items():
        if name in ("gate", "logits", "query"):
            out[name] = np.zeros(shp) if name != "query" else rng.normal(0, 0.1, size=shp)
            if name == "gate":
                out[name] = np.full(shp, float(params.get("gate_init", 0.0)))
        else:
            out[name] = rng.normal(0.0, np.sqrt(1.0 / shp[0]), size=shp)
    return out


def _softmax(x: Any, xp: Any, axis: int = -1) -> Any:
    m = xp.max(x, axis=axis, keepdims=True)
    e = xp.exp(x - m)
    return e / xp.sum(e, axis=axis, keepdims=True)


def apply_merge(op: str, params: dict[str, Any], arrays: dict[str, Any], xs: list[Any], xp: Any) -> Any:
    if op == "select":
        return xs[int(params.get("index", 0))]
    if op == "concat":
        return xp.concatenate(xs, axis=-1)
    if op == "add":
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out
    if op == "gated_add":
        g = 1.0 / (1.0 + xp.exp(-arrays["gate"]))
        out = xs[0]
        for x in xs[1:]:
            out = out + g * x
        return out
    if op == "weighted_sum":
        w = _softmax(arrays["logits"], xp)
        out = w[0] * xs[0]
        for i, x in enumerate(xs[1:], start=1):
            out = out + w[i] * x
        return out
    if op == "attention_pool":
        keys = xs[0]
        scores = keys @ arrays["query"] / xp.sqrt(float(keys.shape[-1]))
        a = _softmax(scores, xp, axis=0)
        return xp.sum(a[:, None] * keys, axis=0)
    if op == "cross_attention":
        q, mem = xs
        qh = q @ arrays["wq"]
        kh = mem @ arrays["wk"]
        vh = mem @ arrays["wv"]
        scores = kh @ qh / xp.sqrt(float(qh.shape[-1]))
        a = _softmax(scores, xp, axis=0)
        return q + xp.sum(a[:, None] * vh, axis=0)
    raise CompositionError(f"unknown merge operator {op!r}")
