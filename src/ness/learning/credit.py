"""``CreditMap``: exactly one primary gradient owner per update-eligible parameter group,
explicit stop boundaries, and validation against the compiled graph (T22/T42)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..composition.compiler import CompiledGraph
from ..contracts import GradientBoundaryError, ValidationError, content_hash

FROZEN = "frozen"


@dataclass(frozen=True, slots=True)
class CreditMap:
    owners: dict[str, str]                   # group_id -> rule id (or "frozen")
    anchors: tuple[tuple[str, str, float], ...] = ()  # (group_id, anchor kind, weight) additive regularisers
    stop_boundaries: tuple[str, ...] = ()

    def groups_owned_by(self, rule_id: str) -> tuple[str, ...]:
        return tuple(sorted(g for g, r in self.owners.items() if r == rule_id))

    def canonical(self) -> dict:
        return {"owners": dict(sorted(self.owners.items())), "anchors": [list(a) for a in self.anchors], "stop_boundaries": list(self.stop_boundaries)}

    @property
    def credit_hash(self) -> str:
        return content_hash(self.canonical())


def module_group_id(node_id: str, group_name: str) -> str:
    return f"{node_id}/{group_name}"


def build_credit_map(compiled: CompiledGraph, default_owner: str, overrides: dict[str, str] | None = None,
                     rule_runtimes: dict[str, str | None] | None = None) -> CreditMap:
    """Assign owners to all update-eligible groups: module trainable groups and learned
    edge (boundary/merge) parameters. Frozen groups are recorded as ``frozen``.

    ``rule_runtimes`` maps each rule id to the runtime it can differentiate/update
    (``bp_direct -> "jax"``, ``fabricpc_pc_local -> "fabricpc"``). A rule may only own
    parameters of modules in that runtime: cross-runtime arrays are not derivative paths
    (``GradientBoundaryError``)."""
    overrides = overrides or {}
    rule_runtimes = dict(rule_runtimes or {"bp_direct": "jax"})
    owners: dict[str, str] = {}

    def check(owner: str, node_runtime: str, what: str) -> None:
        need = rule_runtimes.get(owner, None)
        if owner == FROZEN or need is None:
            return
        if need != node_runtime:
            raise GradientBoundaryError(
                f"{what} lives in runtime {node_runtime!r}; rule {owner!r} operates in runtime {need!r}. "
                "Array transfer across runtimes is not a derivative path.")

    for cn in compiled.nodes:
        for g in cn.descriptor.parameter_groups:
            gid = module_group_id(cn.spec.node_id, g.name)
            if g.mutability == FROZEN:
                owners[gid] = FROZEN
                continue
            owner = overrides.get(gid, default_owner)
            check(owner, cn.descriptor.runtime, f"parameter group {gid} (node {cn.spec.node_id})")
            owners[gid] = owner
    for gid, spec in compiled.edge_params.items():
        owner = overrides.get(gid, default_owner)
        check(owner, compiled.node(spec.owner_node).descriptor.runtime, f"edge parameters {gid} (into node {spec.owner_node})")
        owners[gid] = owner
    unknown = set(overrides) - set(owners)
    if unknown:
        raise ValidationError(f"credit overrides name unknown parameter groups {sorted(unknown)}")
    return CreditMap(owners, (), compiled.stop_edges)
