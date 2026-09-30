"""Learning objectives (ADR-0012): what a learning update minimises, beyond the implicit task
losses on the task output ports.

An arm may declare ``learning.objectives``. Each objective has an ``id``, a ``kind``, a
``weight`` and an optional applicability filter ``applies_to``:

* ``task``: a task loss on the task's output port (the implicit default is one per task,
  weighted by the rule's ``task_weights``; declaring any ``task`` objective replaces the
  implicit set with exactly the declared ones);
* ``port_target``: a loss between a *differentiable* source port (any port the owning rule can
  differentiate) and a *constant* target: either a port of the same graph that is outside the
  differentiable region (typically a ``training_only`` node) or a revealed auxiliary target (an
  ``outcome_only`` observation field, withheld from every prediction and delivered to learning
  by ``reveal``);
* ``parameter_anchor``: ``0.5 * weight * ||theta - theta_ref||^2`` over owned parameter groups,
  with ``theta_ref`` a reference snapshot kept in learner state and checkpoints.

Every objective is reduced as sum of numerators over sum of denominators over the batch items it
applies to (items it does not apply to contribute zero to both), weighted, summed. Targets are
constants to every rule. Loss functions come from a registry that installed packages extend
through the ``ness.losses`` entry-point group.

Nothing here is domain specific: "logits", "classes" and "positions" are generic array roles.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..composition.selectors import OBSERVATION_NODE, SourceSelector
from ..contracts import (
    CompositionError,
    ContractViolation,
    Differentiability,
    FieldRole,
    GradientBoundaryError,
    UnsupportedCapability,
    ValidationError,
    content_hash,
)

OBJECTIVE_KINDS = ("task", "port_target", "parameter_anchor")
LOSS_ENTRY_POINT_GROUP = "ness.losses"
_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")
_COMMON_KEYS = {"id", "kind", "weight", "applies_to"}
_KIND_KEYS = {
    "task": {"task"},
    "port_target": {"source", "target", "loss", "loss_config", "mask"},
    "parameter_anchor": {"groups", "reference"},
}
_FILTER_KEYS = {"group", "has_outcome"}
ANCHOR_REFERENCES = ("initial",)


# --------------------------------------------------------------------------- losses
@dataclass(frozen=True)
class LossSpec:
    """``fn(pred, target, mask, xp, **config) -> (numerator, denominator)``, written against the
    array namespace ``xp`` so it runs eagerly (numpy) and traced (jax.numpy)."""

    name: str
    fn: Callable[..., tuple[Any, Any]]
    description: str = ""
    config_keys: tuple[str, ...] = ()


def _log_softmax(x: Any, xp: Any) -> Any:
    m = xp.max(x, axis=-1, keepdims=True)
    z = x - m
    return z - xp.log(xp.sum(xp.exp(z), axis=-1, keepdims=True))


def _positions_mask(mask: Any, shape: tuple[int, ...], xp: Any) -> Any:
    return xp.ones(shape) if mask is None else mask


def _mse(pred, target, mask, xp):
    per = (pred - target) ** 2
    m = _positions_mask(mask, per.shape, xp)
    m = xp.broadcast_to(xp.reshape(m, m.shape + (1,) * (per.ndim - m.ndim)), per.shape) if m.ndim < per.ndim else m
    return xp.sum(m * per), xp.sum(m)


def _as_log_probs(x, space, temperature, xp):
    if space == "logits":
        return _log_softmax(x / temperature, xp)
    if space == "log_probs":
        return x
    if space == "probs":
        return xp.log(x)
    raise ValidationError(f"unknown distribution space {space!r}; use logits | log_probs | probs")


def _kl_last_axis(pred, target, mask, xp, source="logits", target_space="logits", temperature=1.0):
    """KL(target || source) along the last axis, per position; mask over positions (all leading axes)."""
    t = float(temperature)
    lp_s = _as_log_probs(pred, source, t, xp)
    lp_t = _as_log_probs(target, target_space, t, xp)
    p_t = xp.exp(lp_t)
    kl = xp.sum(xp.where(p_t > 0, p_t * (lp_t - lp_s), 0.0), axis=-1)
    m = _positions_mask(mask, kl.shape, xp)
    return xp.sum(m * kl), xp.sum(m)


def _cross_entropy(pred, target, mask, xp, source="logits"):
    """Negative log-likelihood of integer class ids ``target`` (leading axes = positions) under
    the distribution ``pred`` over the last axis."""
    lp = _as_log_probs(pred, source, 1.0, xp)
    ids = target.astype("int32")
    nll = -xp.take_along_axis(lp, ids[..., None], axis=-1)[..., 0]
    m = _positions_mask(mask, nll.shape, xp)
    return xp.sum(m * nll), xp.sum(m)


BUILTIN_LOSSES = (
    LossSpec("mse", _mse, "sum of mask * (pred - target)^2 over elements / sum of mask", ()),
    LossSpec("kl_last_axis", _kl_last_axis, "KL(target || source) over the last axis, masked over positions; spaces logits|log_probs|probs; optional temperature (no T^2 rescaling)",
             ("source", "target_space", "temperature")),
    LossSpec("cross_entropy", _cross_entropy, "negative log-likelihood of integer ids under source logits/log_probs, masked over positions", ("source",)),
)


def loss_registry(include_external: bool = True) -> dict[str, LossSpec]:
    """Built-in losses plus those installed packages contribute via ``ness.losses`` (a
    ``LossSpec`` or an iterable of them). A name clash fails closed."""
    reg = {spec.name: spec for spec in BUILTIN_LOSSES}
    if include_external:
        from importlib.metadata import entry_points
        for ep in entry_points(group=LOSS_ENTRY_POINT_GROUP):
            try:
                obj = ep.load()
            except Exception as exc:  # noqa: BLE001
                raise ContractViolation(f"loss entry point {ep.name!r} ({ep.value}) failed to load: {type(exc).__name__}: {exc}") from exc
            for spec in ([obj] if isinstance(obj, LossSpec) else list(obj)):
                if not isinstance(spec, LossSpec):
                    raise ContractViolation(f"loss entry point {ep.name!r} must yield LossSpec objects, got {type(spec).__name__}")
                if spec.name in reg:
                    raise ContractViolation(f"loss {spec.name!r} from entry point {ep.name!r} clashes with an existing loss")
                reg[spec.name] = spec
    return reg


# --------------------------------------------------------------------------- specs
@dataclass(frozen=True)
class ApplyFilter:
    """``applies_to``: an item takes part only if every stated condition holds.
    ``group``: ``request.group[key]`` equals the value or is in the list (streams are
    conventionally ``group.stream``); ``has_outcome``: the item revealed outcomes for these tasks."""

    group: tuple[tuple[str, tuple[Any, ...]], ...] = ()
    has_outcome: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, d: Any, oid: str) -> "ApplyFilter":
        if d is None:
            return cls()
        if not isinstance(d, dict):
            raise ValidationError(f"objective {oid}: applies_to must be a mapping")
        unknown = set(d) - _FILTER_KEYS
        if unknown:
            raise ValidationError(f"objective {oid}: applies_to has unknown keys {sorted(unknown)}; allowed {sorted(_FILTER_KEYS)}")
        grp = d.get("group") or {}
        if not isinstance(grp, dict):
            raise ValidationError(f"objective {oid}: applies_to.group must map group keys to a value or a list of values")
        pairs = tuple(sorted((str(k), tuple(v) if isinstance(v, (list, tuple)) else (v,)) for k, v in grp.items()))
        return cls(pairs, tuple(d.get("has_outcome") or ()))

    def matches(self, request: Any, outcomes: dict[str, Any]) -> bool:
        grp = getattr(request, "group", {}) or {}
        for k, vals in self.group:
            if k not in grp or grp[k] not in vals:
                return False
        return all(t in outcomes for t in self.has_outcome)

    def canonical(self) -> dict:
        return {"group": {k: list(v) for k, v in self.group}, "has_outcome": list(self.has_outcome)}


@dataclass(frozen=True)
class ObjectiveSpec:
    objective_id: str
    kind: str
    weight: float
    applies_to: ApplyFilter
    task_id: str | None = None
    source: SourceSelector | None = None
    target_port: SourceSelector | None = None
    target_field: str | None = None
    mask_field: str | None = None
    loss: str | None = None
    loss_config: dict[str, Any] = field(default_factory=dict)
    group_patterns: tuple[str, ...] = ()
    reference: str = "initial"
    groups: tuple[str, ...] = ()           # resolved anchor groups (after bind)

    def canonical(self) -> dict:
        out: dict[str, Any] = {"id": self.objective_id, "kind": self.kind, "weight": self.weight, "applies_to": self.applies_to.canonical()}
        if self.kind == "task":
            out["task"] = self.task_id
        elif self.kind == "port_target":
            out.update({"source": str(self.source), "target": str(self.target_port) if self.target_port else {"auxiliary": self.target_field},
                        "mask": {"auxiliary": self.mask_field} if self.mask_field else None, "loss": self.loss, "loss_config": dict(sorted(self.loss_config.items()))})
        else:
            out.update({"groups": list(self.group_patterns), "resolved_groups": list(self.groups), "reference": self.reference})
        return out


def _aux_field(v: Any, what: str, oid: str) -> str:
    if not (isinstance(v, dict) and set(v) == {"auxiliary"} and isinstance(v["auxiliary"], str)):
        raise ValidationError(f"objective {oid}: {what} must be a selector string or {{auxiliary: <outcome_only field>}}, got {v!r}")
    return v["auxiliary"]


def parse_objectives(raw: Any) -> tuple[ObjectiveSpec, ...] | None:
    """Parse ``learning.objectives`` (None when absent: the v0.2 implicit task objectives)."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValidationError("learning.objectives must be a list")
    out: list[ObjectiveSpec] = []
    seen: set[str] = set()
    for d in raw:
        if not isinstance(d, dict) or "id" not in d or "kind" not in d:
            raise ValidationError(f"every objective needs 'id' and 'kind', got {d!r}")
        oid, kind = str(d["id"]), d["kind"]
        if not _ID.match(oid):
            raise ValidationError(f"objective id {oid!r} must match {_ID.pattern}")
        if oid in seen:
            raise ValidationError(f"duplicate objective id {oid!r}")
        seen.add(oid)
        if kind not in OBJECTIVE_KINDS:
            raise ValidationError(f"objective {oid}: unknown kind {kind!r}; kinds {OBJECTIVE_KINDS}")
        unknown = set(d) - _COMMON_KEYS - _KIND_KEYS[kind]
        if unknown:
            raise ValidationError(f"objective {oid} ({kind}): unknown keys {sorted(unknown)}; allowed {sorted(_COMMON_KEYS | _KIND_KEYS[kind])}")
        weight = float(d.get("weight", 1.0))
        if not np.isfinite(weight) or weight < 0:
            raise ValidationError(f"objective {oid}: weight must be a finite number >= 0")
        flt = ApplyFilter.from_config(d.get("applies_to"), oid)
        if kind == "task":
            if "task" not in d:
                raise ValidationError(f"objective {oid}: kind task needs 'task'")
            out.append(ObjectiveSpec(oid, kind, weight, flt, task_id=str(d["task"])))
        elif kind == "port_target":
            for k in ("source", "target", "loss"):
                if k not in d:
                    raise ValidationError(f"objective {oid}: kind port_target needs {k!r}")
            src = SourceSelector.parse(d["source"])
            tgt = d["target"]
            tport, tfield = (SourceSelector.parse(tgt), None) if isinstance(tgt, str) else (None, _aux_field(tgt, "target", oid))
            mfield = None if d.get("mask") is None else _aux_field(d["mask"], "mask", oid)
            cfg = d.get("loss_config") or {}
            if not isinstance(cfg, dict):
                raise ValidationError(f"objective {oid}: loss_config must be a mapping")
            out.append(ObjectiveSpec(oid, kind, weight, flt, source=src, target_port=tport, target_field=tfield, mask_field=mfield,
                                     loss=str(d["loss"]), loss_config=dict(cfg)))
        else:
            pats = d.get("groups")
            if not pats or not isinstance(pats, list):
                raise ValidationError(f"objective {oid}: kind parameter_anchor needs a non-empty 'groups' list of patterns")
            ref = d.get("reference", "initial")
            if ref not in ANCHOR_REFERENCES:
                raise ValidationError(f"objective {oid}: reference must be one of {ANCHOR_REFERENCES}, got {ref!r}")
            if d.get("applies_to"):
                raise ValidationError(f"objective {oid}: a parameter anchor applies to parameters, not requests; remove applies_to")
            out.append(ObjectiveSpec(oid, kind, weight, flt, group_patterns=tuple(str(p) for p in pats), reference=ref))
    return tuple(out)


# --------------------------------------------------------------------------- plan
@dataclass(frozen=True)
class ObjectivePlan:
    """Objectives bound to one compiled arm. ``task`` and ``port`` objectives are evaluated by the
    JAX differentiable region (``bp_direct``); anchors are applied by the system for any rule."""

    objectives: tuple[ObjectiveSpec, ...]
    task_objectives: tuple[ObjectiveSpec, ...]
    port_objectives: tuple[ObjectiveSpec, ...]
    anchors: tuple[ObjectiveSpec, ...]
    implicit_tasks: bool
    losses: dict[str, LossSpec]
    auxiliary_fields: frozenset[str]
    source_keys: dict[str, tuple[str, str]]
    target_keys: dict[str, tuple[str, str]]

    @property
    def uses_region(self) -> bool:
        return bool(self.port_objectives) or not self.implicit_tasks

    def canonical(self) -> dict:
        return {"objectives": [o.canonical() for o in self.objectives], "implicit_task_objectives": self.implicit_tasks,
                "reduction": "per objective: sum numerators / sum denominators over the items it applies to; weighted sum over objectives",
                "losses": {n: self.losses[n].description for n in sorted({o.loss for o in self.port_objectives if o.loss})}}

    @property
    def plan_hash(self) -> str:
        return content_hash(self.canonical())


def bind_objectives(specs: tuple[ObjectiveSpec, ...], compiled: Any, observation_fields: dict[str, Any], tasks: dict[str, Any],
                    owners: dict[str, str], rules: dict[str, Any], rule_task_weights: dict[str, float] | None = None) -> ObjectivePlan:
    """Validate declared objectives against the compiled graph, the scenario's fields, the tasks
    and the credit map; fail closed on anything that cannot hold."""
    losses = loss_registry()
    task_objs = [o for o in specs if o.kind == "task"]
    port_objs = [o for o in specs if o.kind == "port_target"]
    anchors: list[ObjectiveSpec] = []
    owned = sorted(g for g, r in owners.items() if r != "frozen")
    source_keys: dict[str, tuple[str, str]] = {}
    target_keys: dict[str, tuple[str, str]] = {}
    aux: set[str] = set()

    def node_port(sel: SourceSelector, oid: str, what: str):
        if sel.node_id == OBSERVATION_NODE:
            raise CompositionError(f"objective {oid}: {what} {sel} is an observation field; objectives read node ports (or {{auxiliary: field}} targets)")
        try:
            cn = compiled.node(sel.node_id)
        except CompositionError:
            raise CompositionError(f"objective {oid}: {what} {sel} names unknown node {sel.node_id!r}") from None
        if not sel.expects_kind(cn.descriptor.module_kind):
            raise CompositionError(f"objective {oid}: {what} {sel} uses scheme {sel.scheme!r} but node {sel.node_id} is a {cn.descriptor.module_kind}")
        try:
            return cn, cn.descriptor.output_port(sel.port)
        except KeyError:
            raise CompositionError(f"objective {oid}: node {sel.node_id} has no output port {sel.port!r}") from None

    def outcome_field(name: str, oid: str) -> None:
        f = observation_fields.get(name)
        if f is None:
            raise ValidationError(f"objective {oid}: auxiliary target {name!r} is not an observation field declared by the scenario")
        if f.role != FieldRole.OUTCOME_ONLY:
            raise ValidationError(f"objective {oid}: auxiliary target {name!r} has role {f.role.value}; auxiliary targets must be outcome_only")
        aux.add(name)

    for o in task_objs:
        if o.task_id not in tasks or o.task_id not in compiled.outputs:
            raise ValidationError(f"objective {o.objective_id}: unknown task {o.task_id!r} (tasks with outputs: {sorted(compiled.outputs)})")
    for o in port_objs:
        cn, port = node_port(o.source, o.objective_id, "source")
        if not cn.differentiable or port.differentiability != Differentiability.DIFFERENTIABLE:
            raise GradientBoundaryError(f"objective {o.objective_id}: source {o.source} is not differentiable by any learning rule "
                                        f"(node runtime {cn.descriptor.runtime!r}, port {port.differentiability}); objectives need a differentiable source port")
        source_keys[o.objective_id] = (cn.spec.node_id, port.name)
        if o.target_port is not None:
            tcn, tport = node_port(o.target_port, o.objective_id, "target")
            if tcn.differentiable:
                raise GradientBoundaryError(f"objective {o.objective_id}: target {o.target_port} lies in the differentiable region; targets must be "
                                            "constants (a training_only or other non-differentiable node)")
            target_keys[o.objective_id] = (tcn.spec.node_id, tport.name)
        else:
            outcome_field(o.target_field, o.objective_id)  # type: ignore[arg-type]
        if o.mask_field is not None:
            outcome_field(o.mask_field, o.objective_id)
        if o.loss not in losses:
            raise ValidationError(f"objective {o.objective_id}: unknown loss {o.loss!r}; available {sorted(losses)} (extend via the ness.losses entry point)")
        allowed = set(losses[o.loss].config_keys)  # type: ignore[index]
        bad = set(o.loss_config) - allowed
        if bad:
            raise ValidationError(f"objective {o.objective_id}: loss {o.loss!r} has no options {sorted(bad)}; options {sorted(allowed)}")
    for o in (x for x in specs if x.kind == "parameter_anchor"):
        groups = []
        for pat in o.group_patterns:
            hit = [g for g in owned if fnmatch.fnmatchcase(g, pat)]
            if not hit:
                raise ValidationError(f"objective {o.objective_id}: anchor pattern {pat!r} matches no owned (trainable) parameter group; owned: {owned}")
            groups.extend(g for g in hit if g not in groups)
        anchors.append(ObjectiveSpec(o.objective_id, o.kind, o.weight, o.applies_to, group_patterns=o.group_patterns, reference=o.reference, groups=tuple(groups)))
    if task_objs or port_objs:
        region_rules = {rid for rid, r in rules.items() if rid == "bp_direct"}
        others = sorted(rid for rid in rules if rid != "bp_direct" and any(r == rid for r in owners.values()))
        if others:
            raise UnsupportedCapability(f"task/port_target objectives are evaluated by bp_direct only; rules {others} also own parameter groups "
                                        "(parameter anchors work with every rule)")
        if not region_rules:
            raise UnsupportedCapability("task/port_target objectives need the bp_direct learning rule")
        cfg = rules["bp_direct"].config
        if cfg.get("batched") or cfg.get("data_parallel"):
            raise UnsupportedCapability("task/port_target objectives are verified in per-item evaluation only; remove batched/data_parallel")
    implicit = not task_objs
    if implicit:
        weights = rule_task_weights or {}
        task_objs = [ObjectiveSpec(f"task:{t}", "task", float(weights.get(t, 1.0)), ApplyFilter(), task_id=t) for t in compiled.outputs]
    anchor_ids = {o.objective_id for o in anchors}
    ordered = tuple(o for o in specs if o.objective_id not in anchor_ids) + tuple(anchors)
    return ObjectivePlan(ordered, tuple(task_objs), tuple(port_objs), tuple(anchors), implicit, losses, frozenset(aux), source_keys, target_keys)
