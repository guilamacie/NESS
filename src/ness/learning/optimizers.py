"""Optimisers over ``{group_id: {name: array}}`` with restricted-data state snapshots.
Frozen groups are excluded from updates, decay and moment evolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..contracts import ContractViolation

Params = dict[str, dict[str, np.ndarray]]


@dataclass
class Optimizer:
    kind: str
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    clip_norm: float | None = None
    step: int = 0
    m: Params = field(default_factory=dict)
    v: Params = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Optimizer":
        kind = cfg.get("kind", "adam")
        if kind not in ("adam", "sgd"):
            raise ContractViolation(f"unknown optimizer {kind!r}")
        return cls(kind, float(cfg.get("lr", 1e-2)), float(cfg.get("beta1", 0.9)), float(cfg.get("beta2", 0.999)),
                   float(cfg.get("eps", 1e-8)), float(cfg.get("weight_decay", 0.0)), cfg.get("clip_norm"))

    def update(self, params: Params, grads: Params) -> Params:
        """Return updated copies for the groups present in ``grads``; others untouched."""
        self.step += 1
        if self.clip_norm is not None:
            total = float(np.sqrt(sum(float(np.sum(np.asarray(g) ** 2)) for grp in grads.values() for g in grp.values())))
            scale = min(1.0, self.clip_norm / (total + 1e-12))
            grads = {gid: {n: np.asarray(g) * scale for n, g in grp.items()} for gid, grp in grads.items()}
        out: Params = {gid: {n: np.array(a, copy=True) for n, a in grp.items()} for gid, grp in params.items()}
        for gid, grp in grads.items():
            for n, g in grp.items():
                g = np.asarray(g, dtype=np.float64)
                p = out[gid][n]
                if self.weight_decay:
                    p = p * (1.0 - self.lr * self.weight_decay)
                if self.kind == "sgd":
                    out[gid][n] = p - self.lr * g
                    continue
                m = self.m.setdefault(gid, {}).get(n, np.zeros_like(g))
                v = self.v.setdefault(gid, {}).get(n, np.zeros_like(g))
                m = self.beta1 * m + (1 - self.beta1) * g
                v = self.beta2 * v + (1 - self.beta2) * g**2
                self.m[gid][n], self.v[gid][n] = m, v
                mhat = m / (1 - self.beta1**self.step)
                vhat = v / (1 - self.beta2**self.step)
                out[gid][n] = p - self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return out

    def snapshot(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        # '|' never occurs in group ids (which may themselves contain '/', '#', ':' and '.')
        arrays = {f"m|{gid}|{n}": a for gid, grp in self.m.items() for n, a in grp.items()}
        arrays.update({f"v|{gid}|{n}": a for gid, grp in self.v.items() for n, a in grp.items()})
        return arrays, {"kind": self.kind, "lr": self.lr, "beta1": self.beta1, "beta2": self.beta2, "eps": self.eps,
                        "weight_decay": self.weight_decay, "clip_norm": self.clip_norm, "step": self.step}

    @classmethod
    def restore(cls, arrays: dict[str, np.ndarray], data: dict[str, Any]) -> "Optimizer":
        opt = cls(data["kind"], data["lr"], data["beta1"], data["beta2"], data["eps"], data["weight_decay"], data["clip_norm"], data["step"])
        for k, a in arrays.items():
            which, gid, n = k.split("|", 2)
            (opt.m if which == "m" else opt.v).setdefault(gid, {})[n] = np.asarray(a)
        return opt
