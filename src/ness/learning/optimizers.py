"""Optimisers over ``{group_id: {name: array}}`` with restricted-data state snapshots.
Frozen groups are excluded from updates, decay and moment evolution.

Per-update order of operations (ADR-0013), for every owned group ``g`` present in the gradients:

1. ``grad_multiplier``: ``grad_g <- m_g * grad_g`` (a per-group loss scale; logged separately
   from the learning rate);
2. per-group clipping: if the group rule sets ``clip_norm``, ``grad_g`` is rescaled so that its
   L2 norm (over the group's arrays) is at most that value;
3. global clipping (the arm-level ``clip_norm``): all gradients are rescaled together so that the
   global L2 norm is at most that value;
4. decoupled weight decay: ``p <- p * (1 - lr_g * wd_g)`` *before* the step (AdamW-style, applied
   to SGD too);
5. the step: SGD ``p <- p - lr_g * grad``; Adam with bias-corrected moments.

``lr_g = (lr of the group rule, or arm lr * lr_scale) * schedule(step)``. Without ``param_groups``
and ``schedule`` the arithmetic is exactly the v0.2 optimizer's.
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..contracts import ContractViolation, ValidationError

Params = dict[str, dict[str, np.ndarray]]

SCHEDULE_KINDS = ("constant", "linear", "cosine")
_OPT_KEYS = {"kind", "lr", "beta1", "beta2", "eps", "weight_decay", "clip_norm", "schedule", "param_groups"}
_GROUP_KEYS = {"match", "lr", "lr_scale", "weight_decay", "grad_multiplier", "clip_norm"}
_SCHEDULE_KEYS = {"kind", "warmup_steps", "total_steps", "min_lr_ratio"}


@dataclass(frozen=True)
class ScheduleSpec:
    """Step-based learning-rate factor: linear warm-up over ``warmup_steps`` updates, then
    ``constant`` (1), ``linear`` or ``cosine`` decay to ``min_lr_ratio`` at ``total_steps``.
    Steps are 1-based optimizer updates; after ``total_steps`` the factor stays at the floor."""

    kind: str = "constant"
    warmup_steps: int = 0
    total_steps: int | None = None
    min_lr_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in SCHEDULE_KINDS:
            raise ValidationError(f"optimizer.schedule.kind must be one of {SCHEDULE_KINDS}, got {self.kind!r}")
        if self.warmup_steps < 0:
            raise ValidationError("optimizer.schedule.warmup_steps must be >= 0")
        if self.kind in ("linear", "cosine"):
            if self.total_steps is None or self.total_steps <= self.warmup_steps:
                raise ValidationError(f"optimizer.schedule.kind={self.kind!r} needs total_steps > warmup_steps")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValidationError("optimizer.schedule.min_lr_ratio must lie in [0, 1]")

    @classmethod
    def from_config(cls, d: Any) -> "ScheduleSpec | None":
        if d is None:
            return None
        if not isinstance(d, dict):
            raise ValidationError("optimizer.schedule must be a mapping")
        unknown = set(d) - _SCHEDULE_KEYS
        if unknown:
            raise ValidationError(f"optimizer.schedule has unknown keys {sorted(unknown)}; allowed {sorted(_SCHEDULE_KEYS)}")
        return cls(str(d.get("kind", "constant")), int(d.get("warmup_steps", 0)),
                   None if d.get("total_steps") is None else int(d["total_steps"]), float(d.get("min_lr_ratio", 0.0)))

    def factor(self, step: int) -> float:
        if self.warmup_steps and step <= self.warmup_steps:
            return step / self.warmup_steps
        if self.kind == "constant":
            return 1.0
        assert self.total_steps is not None
        p = min(1.0, max(0.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
        if self.kind == "linear":
            return 1.0 - (1.0 - self.min_lr_ratio) * p
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))

    def canonical(self) -> dict:
        return {"kind": self.kind, "warmup_steps": self.warmup_steps, "total_steps": self.total_steps, "min_lr_ratio": self.min_lr_ratio}


@dataclass(frozen=True)
class GroupRule:
    """One ``optimizer.param_groups`` entry. ``match`` is a shell-style glob over group ids
    (``node/group`` for module groups, the edge-parameter id for learned boundaries/merges)."""

    match: str
    lr: float | None = None
    lr_scale: float | None = None
    weight_decay: float | None = None
    grad_multiplier: float = 1.0
    clip_norm: float | None = None

    def __post_init__(self) -> None:
        if self.lr is not None and self.lr_scale is not None:
            raise ValidationError(f"optimizer.param_groups[{self.match!r}]: set lr or lr_scale, not both")
        for name in ("lr", "lr_scale", "weight_decay", "clip_norm"):
            v = getattr(self, name)
            if v is not None and (not np.isfinite(v) or v < 0):
                raise ValidationError(f"optimizer.param_groups[{self.match!r}].{name} must be a finite number >= 0")
        if not np.isfinite(self.grad_multiplier):
            raise ValidationError(f"optimizer.param_groups[{self.match!r}].grad_multiplier must be finite")

    @classmethod
    def from_config(cls, d: Any) -> "GroupRule":
        if not isinstance(d, dict) or "match" not in d:
            raise ValidationError(f"optimizer.param_groups entries need a 'match' pattern, got {d!r}")
        unknown = set(d) - _GROUP_KEYS
        if unknown:
            raise ValidationError(f"optimizer.param_groups[{d['match']!r}] has unknown keys {sorted(unknown)}; allowed {sorted(_GROUP_KEYS)}")
        f = lambda k: None if d.get(k) is None else float(d[k])  # noqa: E731
        return cls(str(d["match"]), f("lr"), f("lr_scale"), f("weight_decay"), float(d.get("grad_multiplier", 1.0)), f("clip_norm"))

    def canonical(self) -> dict:
        return {"match": self.match, "lr": self.lr, "lr_scale": self.lr_scale, "weight_decay": self.weight_decay,
                "grad_multiplier": self.grad_multiplier, "clip_norm": self.clip_norm}


def _norm(grp: dict[str, np.ndarray]) -> float:
    return float(np.sqrt(sum(float(np.sum(np.asarray(g) ** 2)) for g in grp.values())))


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
    schedule: ScheduleSpec | None = None
    group_rules: tuple[GroupRule, ...] = ()
    group_of: dict[str, int] = field(default_factory=dict)          # gid -> index of the first matching rule (after bind_groups)
    last_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Optimizer":
        kind = cfg.get("kind", "adam")
        if kind not in ("adam", "sgd"):
            raise ContractViolation(f"unknown optimizer {kind!r}")
        unknown = set(cfg) - _OPT_KEYS
        if unknown:
            raise ValidationError(f"optimizer has unknown keys {sorted(unknown)}; allowed {sorted(_OPT_KEYS)}")
        rules = cfg.get("param_groups")
        if rules is not None and not isinstance(rules, list):
            raise ValidationError("optimizer.param_groups must be a list of {match, ...} rules")
        return cls(kind, float(cfg.get("lr", 1e-2)), float(cfg.get("beta1", 0.9)), float(cfg.get("beta2", 0.999)),
                   float(cfg.get("eps", 1e-8)), float(cfg.get("weight_decay", 0.0)), cfg.get("clip_norm"),
                   schedule=ScheduleSpec.from_config(cfg.get("schedule")), group_rules=tuple(GroupRule.from_config(r) for r in (rules or ())))

    # ------------------------------------------------------------------ groups
    @property
    def configured(self) -> bool:
        """True when param_groups or a schedule is declared (the v0.2 path otherwise)."""
        return bool(self.group_rules) or self.schedule is not None

    def bind_groups(self, owned_groups: tuple[str, ...] | list[str]) -> None:
        """Resolve rules against the owned (update-eligible) group ids: first match wins; a rule
        that matches no owned group is an error (fail closed)."""
        self.group_of = {}
        used = [False] * len(self.group_rules)
        for gid in sorted(owned_groups):
            for i, r in enumerate(self.group_rules):
                if fnmatch.fnmatchcase(gid, r.match):
                    self.group_of[gid] = i
                    used[i] = True
                    break
        unused = [r.match for r, u in zip(self.group_rules, used) if not u]
        if unused:
            raise ValidationError(f"optimizer.param_groups patterns {unused} match no owned parameter group; owned groups: {sorted(owned_groups)}")

    def _rule(self, gid: str) -> GroupRule | None:
        i = self.group_of.get(gid)
        return self.group_rules[i] if i is not None else None

    def group_lr(self, gid: str, step: int | None = None) -> float:
        """Effective learning rate of ``gid`` at ``step`` (default: the step of the next update)."""
        step = self.step + 1 if step is None else step
        r = self._rule(gid)
        lr = self.lr if r is None else (r.lr if r.lr is not None else (self.lr * r.lr_scale if r.lr_scale is not None else self.lr))
        return lr * self.schedule.factor(step) if self.schedule is not None else lr

    # ------------------------------------------------------------------ update
    def update(self, params: Params, grads: Params) -> Params:
        """Return updated copies for the groups present in ``grads``; others untouched."""
        self.step += 1
        stats: dict[str, dict[str, float]] = {}
        if self.configured:
            raw = {gid: _norm(grp) for gid, grp in grads.items()}
            new: Params = {}
            for gid, grp in grads.items():
                r = self._rule(gid)
                if r is None:
                    new[gid] = grp
                    continue
                g2 = {n: np.asarray(g) * r.grad_multiplier for n, g in grp.items()}
                if r.clip_norm is not None:
                    gn = _norm(g2)
                    sc = min(1.0, r.clip_norm / (gn + 1e-12))
                    g2 = {n: g * sc for n, g in g2.items()}
                new[gid] = g2
            grads = new
        if self.clip_norm is not None:
            total = float(np.sqrt(sum(float(np.sum(np.asarray(g) ** 2)) for grp in grads.values() for g in grp.values())))
            scale = min(1.0, self.clip_norm / (total + 1e-12))
            grads = {gid: {n: np.asarray(g) * scale for n, g in grp.items()} for gid, grp in grads.items()}
        out: Params = {gid: {n: np.array(a, copy=True) for n, a in grp.items()} for gid, grp in params.items()}
        for gid, grp in grads.items():
            if self.configured:
                r = self._rule(gid)
                lr = self.group_lr(gid, self.step)
                wd = r.weight_decay if (r is not None and r.weight_decay is not None) else self.weight_decay
            else:
                lr, wd = self.lr, self.weight_decay
            for n, g in grp.items():
                g = np.asarray(g, dtype=np.float64)
                p = out[gid][n]
                if wd:
                    p = p * (1.0 - lr * wd)
                if self.kind == "sgd":
                    out[gid][n] = p - lr * g
                    continue
                m = self.m.setdefault(gid, {}).get(n, np.zeros_like(g))
                v = self.v.setdefault(gid, {}).get(n, np.zeros_like(g))
                m = self.beta1 * m + (1 - self.beta1) * g
                v = self.beta2 * v + (1 - self.beta2) * g**2
                self.m[gid][n], self.v[gid][n] = m, v
                mhat = m / (1 - self.beta1**self.step)
                vhat = v / (1 - self.beta2**self.step)
                out[gid][n] = p - lr * mhat / (np.sqrt(vhat) + self.eps)
            if self.configured:
                r = self._rule(gid)
                stats[gid] = {"lr": lr, "weight_decay": wd, "grad_multiplier": r.grad_multiplier if r is not None else 1.0,
                              "grad_norm_raw": raw[gid], "grad_norm_applied": _norm(grp),
                              "update_norm": float(np.sqrt(sum(float(np.sum((out[gid][n] - np.asarray(params[gid][n])) ** 2)) for n in grp)))}
        self.last_stats = stats
        return out

    # ------------------------------------------------------------------ identity / state
    def spec(self) -> dict[str, Any]:
        """Declared optimizer semantics for the manifest's algorithm specs (configured only)."""
        return {"kind": self.kind, "lr": self.lr, "weight_decay": self.weight_decay, "clip_norm": self.clip_norm,
                "schedule": self.schedule.canonical() if self.schedule else None, "param_groups": [r.canonical() for r in self.group_rules],
                "group_assignment": {gid: self.group_rules[i].match for gid, i in sorted(self.group_of.items())},
                "order": "grad_multiplier -> per-group clip_norm -> global clip_norm -> decoupled weight decay p*(1-lr*wd) -> step"}

    def config_canonical(self) -> dict[str, Any]:
        return {"kind": self.kind, "lr": self.lr, "beta1": self.beta1, "beta2": self.beta2, "eps": self.eps, "weight_decay": self.weight_decay,
                "clip_norm": self.clip_norm, "schedule": self.schedule.canonical() if self.schedule else None,
                "param_groups": [r.canonical() for r in self.group_rules]}

    def snapshot(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        # '|' never occurs in group ids (which may themselves contain '/', '#', ':' and '.')
        arrays = {f"m|{gid}|{n}": a for gid, grp in self.m.items() for n, a in grp.items()}
        arrays.update({f"v|{gid}|{n}": a for gid, grp in self.v.items() for n, a in grp.items()})
        data = {"kind": self.kind, "lr": self.lr, "beta1": self.beta1, "beta2": self.beta2, "eps": self.eps,
                "weight_decay": self.weight_decay, "clip_norm": self.clip_norm, "step": self.step}
        if self.schedule is not None:
            data["schedule"] = self.schedule.canonical()
        if self.group_rules:
            data["param_groups"] = [r.canonical() for r in self.group_rules]
        return arrays, data

    @classmethod
    def restore(cls, arrays: dict[str, np.ndarray], data: dict[str, Any]) -> "Optimizer":
        opt = cls(data["kind"], data["lr"], data["beta1"], data["beta2"], data["eps"], data["weight_decay"], data["clip_norm"], data["step"],
                  schedule=ScheduleSpec.from_config(data.get("schedule")),
                  group_rules=tuple(GroupRule.from_config({k: v for k, v in r.items() if v is not None}) for r in data.get("param_groups", ())))
        for k, a in arrays.items():
            which, gid, n = k.split("|", 2)
            if which not in ("m", "v"):
                continue  # other learner state (e.g. anchor references) shares the optimizer blob
            (opt.m if which == "m" else opt.v).setdefault(gid, {})[n] = np.asarray(a)
        return opt
