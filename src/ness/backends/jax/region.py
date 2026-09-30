"""Reference differentiable runtime (JAX). Only ``ness.backends`` modules import JAX.

``DifferentiableRegion`` re-runs the differentiable nodes of a compiled graph as a pure
function of the trainable parameters, feeding every cross-runtime / stop-gradient input
as a constant recorded during the eager forward pass: host and frozen values are
stop-gradient leaves; only jax->jax edges over ports marked ``differentiable`` carry
derivatives. ``BPDirectRule`` (profile ``ff_bp`` / ``bp_direct``) differentiates the task
loss through this region.

Two evaluation modes share one objective definition:
* per-item (default): a Python loop over batch items, any shapes;
* batched: items stacked and ``vmap``-ed, optionally sharded over a ``("data",)`` mesh with
  ``NamedSharding`` (NESS-owned data parallelism; parameters replicated). Requires uniform
  shapes across items; loss is sum-of-numerators / sum-of-denominators in both modes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import ContractViolation, ModuleDescriptor, ModuleState, PluginDependencyMissing, UnsupportedCapability
from ...learning.batch import BatchItem, LearningContext
from ...learning.credit import module_group_id
from ...runtimes.bootstrap import ensure_bootstrapped

try:  # pragma: no cover - import guard
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
except ImportError as exc:  # pragma: no cover
    raise PluginDependencyMissing("the 'jax' backend requires jax; install ness[jax]") from exc

from ...composition.compiler import CompiledGraph  # noqa: E402
from ...runtime.assembly import assemble_dense  # noqa: E402
from ...runtime.executor import ExecutionRecord  # noqa: E402

__all__ = ["jnp", "jax", "DifferentiableRegion", "BPDirectRule", "BatchItem", "sharded_mesh"]


def _split_gid(gid: str) -> tuple[str, str] | None:
    if "#" in gid:
        return None
    node, group = gid.split("/", 1)
    return node, group


def _float_dtype():
    """The active floating dtype (float64 under x64, float32 otherwise): masks are built in it
    so no float64 request is made when x64 is off (bug B-2: a JAX dtype warning per update)."""
    return jax.dtypes.canonicalize_dtype(jnp.float64)


def sharded_mesh(devices: Any = None):
    """A one-axis ``("data",)`` mesh over the selected (or all) devices."""
    devs = list(devices) if devices is not None else jax.devices()
    return jax.make_mesh((len(devs),), ("data",), devices=devs)


class DifferentiableRegion:
    def __init__(self, compiled: CompiledGraph) -> None:
        if compiled.differentiable_runtime != "jax":
            raise ContractViolation(f"DifferentiableRegion requires the 'jax' runtime; graph declares {compiled.differentiable_runtime!r}")
        ensure_bootstrapped("jax")
        self.compiled = compiled
        self.nodes = [cn for cn in compiled.nodes if cn.differentiable]
        for cn in self.nodes:
            if not hasattr(cn.module, "apply"):
                raise ContractViolation(f"differentiable node {cn.spec.node_id} ({cn.descriptor.plugin_id}) lacks a pure apply()")
        # constant (stop-gradient) source keys read by the region
        self._const_keys: list[tuple[str, str]] = []
        for cn in self.nodes:
            for ri in cn.inputs.values():
                for s in ri.sources:
                    key = (s.producer_node, s.producer_port.name)
                    if not (s.differentiable and any(n.spec.node_id == s.producer_node for n in self.nodes)) and key not in self._const_keys:
                        self._const_keys.append(key)
        for src in compiled.outputs.values():
            key = (src.producer_node, src.producer_port.name)
            if key not in self._const_keys and not any(n.spec.node_id == src.producer_node for n in self.nodes):
                self._const_keys.append(key)

    # ------------------------------------------------------------------ params
    def collect_trainable(self, node_states: dict[str, ModuleState], edge_params: dict[str, dict[str, np.ndarray]],
                          owned_groups: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for gid in owned_groups:
            split = _split_gid(gid)
            if split is None:
                out[gid] = {n: jnp.asarray(a) for n, a in edge_params[gid].items()}
            else:
                node, group = split
                out[gid] = {n: jnp.asarray(a) for n, a in node_states[node].params[group].items()}
        return out

    def constants(self, record: ExecutionRecord) -> dict[tuple[str, str], np.ndarray]:
        return {k: np.asarray(record.port_values[k].dense(), dtype=np.float64) for k in self._const_keys}

    # ------------------------------------------------------------------ traced forward
    def _forward(self, consts: dict[tuple[str, str], Any], node_states: dict[str, ModuleState], edge_params: dict[str, dict[str, Any]],
                 train_params: dict[str, dict[str, Any]]) -> dict[tuple[str, str], Any]:
        edge_view: dict[str, dict[str, Any]] = {gid: {n: jnp.asarray(a) for n, a in grp.items()} for gid, grp in edge_params.items()}
        for gid, grp in train_params.items():
            if _split_gid(gid) is None:
                edge_view[gid] = grp
        computed: dict[tuple[str, str], Any] = {}
        for cn in self.nodes:
            nid = cn.spec.node_id
            dense_inputs: dict[str, Any] = {}
            for port, ri in cn.inputs.items():
                arrays = []
                for s in ri.sources:
                    key = (s.producer_node, s.producer_port.name)
                    arrays.append(computed[key] if (s.differentiable and key in computed) else jnp.asarray(consts[key]))
                dense_inputs[port] = assemble_dense(ri, arrays, edge_view, jnp)
            params = {}
            for group, grp in node_states[nid].params.items():
                gid = module_group_id(nid, group)
                params[group] = train_params[gid] if gid in train_params else {n: jnp.asarray(a) for n, a in grp.items()}
            outs = cn.module.apply(params, dense_inputs, node_states[nid], jnp)  # type: ignore[attr-defined]
            for pspec in cn.descriptor.output_ports:
                if pspec.name in outs:
                    computed[(nid, pspec.name)] = outs[pspec.name]
        return computed

    def traced_forward(self, record: ExecutionRecord, node_states, edge_params, train_params):
        return self._forward(self.constants(record), node_states, edge_params, train_params)

    def _task_terms(self, computed, consts, outcomes_y, outcomes_mask, tasks, weights):
        total = 0.0
        terms = {}
        for task_id, src in self.compiled.outputs.items():
            if task_id not in outcomes_y:
                continue
            key = (src.producer_node, src.producer_port.name)
            pred = computed[key] if key in computed else jnp.asarray(consts[key])
            n, d = tasks[task_id].loss_terms(pred, outcomes_y[task_id], outcomes_mask[task_id], jnp)
            terms[task_id] = (n, d)
        return terms

    # ------------------------------------------------------------------ per-item objective
    def loss_and_grads(self, batch: list[BatchItem], node_states, edge_params, owned_groups: tuple[str, ...], tasks: dict[str, Any],
                       task_weights: dict[str, float] | None = None, objectives: Any = None):
        if not batch:
            raise ContractViolation("empty batch: declared protocol rejects no-op updates")
        if objectives is not None and objectives.uses_region:
            return self._loss_and_grads_objectives(batch, node_states, edge_params, owned_groups, tasks, objectives)
        weights = task_weights or {}
        train = self.collect_trainable(node_states, edge_params, owned_groups)
        consts_list = [self.constants(it.record) for it in batch]
        fdt = _float_dtype()

        def objective(tp):
            nums = {t: 0.0 for t in self.compiled.outputs}
            dens = {t: 0.0 for t in self.compiled.outputs}
            for it, consts in zip(batch, consts_list):
                computed = self._forward(consts, node_states, edge_params, tp)
                ys = {t: jnp.asarray(o.values) for t, o in it.outcomes.items()}
                ms = {t: jnp.asarray(o.mask, dtype=fdt) for t, o in it.outcomes.items()}
                for t, (n, d) in self._task_terms(computed, consts, ys, ms, tasks, weights).items():
                    nums[t] = nums[t] + n
                    dens[t] = dens[t] + d
            return sum(weights.get(t, 1.0) * nums[t] / jnp.maximum(dens[t], 1e-12) for t in nums)

        loss, grads = jax.value_and_grad(objective)(train)
        return self._finish(loss, grads, {"mode": "per_item", "batch_size": len(batch), "owned_groups": list(owned_groups)})

    def _loss_and_grads_objectives(self, batch: list[BatchItem], node_states, edge_params, owned_groups: tuple[str, ...], tasks: dict[str, Any], plan: Any):
        """Declared objectives (ADR-0012). Task objectives reproduce the default reduction exactly
        (same accumulation order and expression); ``port_target`` objectives follow. An objective
        with weight 0 is evaluated for logging only and does not enter the differentiated sum, so
        adding it leaves gradients bitwise unchanged. Items an objective does not apply to
        contribute zero to its numerator and denominator."""
        train = self.collect_trainable(node_states, edge_params, owned_groups)
        consts_list = [self.constants(it.record) for it in batch]
        fdt = _float_dtype()
        counts = {o.objective_id: 0 for o in (*plan.task_objectives, *plan.port_objectives)}
        for it in batch:
            for o in (*plan.task_objectives, *plan.port_objectives):
                if o.objective_id in it.active_objectives:
                    counts[o.objective_id] += 1

        def objective(tp):
            tnum = {o.objective_id: 0.0 for o in plan.task_objectives}
            tden = {o.objective_id: 0.0 for o in plan.task_objectives}
            pnum = {o.objective_id: 0.0 for o in plan.port_objectives}
            pden = {o.objective_id: 0.0 for o in plan.port_objectives}
            for it, consts in zip(batch, consts_list):
                computed = self._forward(consts, node_states, edge_params, tp)
                for o in plan.task_objectives:
                    if o.objective_id not in it.active_objectives:
                        continue
                    t = o.task_id
                    src = self.compiled.outputs[t]
                    key = (src.producer_node, src.producer_port.name)
                    pred = computed[key] if key in computed else jnp.asarray(consts[key])
                    oc = it.outcomes[t]
                    n, d = tasks[t].loss_terms(pred, jnp.asarray(oc.values), jnp.asarray(oc.mask, dtype=fdt), jnp)
                    tnum[o.objective_id] = tnum[o.objective_id] + n
                    tden[o.objective_id] = tden[o.objective_id] + d
                for o in plan.port_objectives:
                    if o.objective_id not in it.active_objectives:
                        continue
                    key = plan.source_keys[o.objective_id]
                    pred = computed[key] if key in computed else jnp.asarray(consts[key])
                    target, mask = it.objective_inputs[o.objective_id]
                    n, d = plan.losses[o.loss].fn(pred, jnp.asarray(target), None if mask is None else jnp.asarray(mask, dtype=fdt), jnp, **o.loss_config)
                    pnum[o.objective_id] = pnum[o.objective_id] + n
                    pden[o.objective_id] = pden[o.objective_id] + d
            total = sum(o.weight * tnum[o.objective_id] / jnp.maximum(tden[o.objective_id], 1e-12) for o in plan.task_objectives if o.weight != 0.0)
            for o in plan.port_objectives:
                if o.weight != 0.0:
                    total = total + o.weight * pnum[o.objective_id] / jnp.maximum(pden[o.objective_id], 1e-12)
            aux = {**{k: (tnum[k], tden[k]) for k in tnum}, **{k: (pnum[k], pden[k]) for k in pnum}}
            return total, aux

        (loss, aux), grads = jax.value_and_grad(objective, has_aux=True)(train)
        weights = {o.objective_id: o.weight for o in (*plan.task_objectives, *plan.port_objectives)}
        per_obj = {}
        for oid, (n, d) in aux.items():
            n, d = float(n), float(d)
            per_obj[oid] = {"value": n / d if d > 0 else float("nan"), "weight": weights[oid], "numerator": n, "denominator": d, "items": counts[oid]}
        return self._finish(loss, grads, {"mode": "per_item", "batch_size": len(batch), "owned_groups": list(owned_groups), "objectives": per_obj})

    # ------------------------------------------------------------------ batched / sharded objective
    def loss_and_grads_batched(self, batch: list[BatchItem], node_states, edge_params, owned_groups: tuple[str, ...], tasks: dict[str, Any],
                               task_weights: dict[str, float] | None = None, mesh: Any = None):
        """Stacked + vmapped objective; with ``mesh`` the batch axis is sharded over ``data``
        and parameters are replicated. Identical mathematics to ``loss_and_grads``."""
        if not batch:
            raise ContractViolation("empty batch: declared protocol rejects no-op updates")
        weights = task_weights or {}
        train = self.collect_trainable(node_states, edge_params, owned_groups)
        consts_list = [self.constants(it.record) for it in batch]
        for c in consts_list[1:]:
            for k in self._const_keys:
                if c[k].shape != consts_list[0][k].shape:
                    raise UnsupportedCapability(f"batched/sharded evaluation requires uniform shapes; port {k} differs across items. Use per-item mode.")
        task_ids = [t for t in self.compiled.outputs if all(t in it.outcomes for it in batch)]
        stacked = {k: jnp.asarray(np.stack([c[k] for c in consts_list])) for k in self._const_keys}
        ys = {t: jnp.asarray(np.stack([it.outcomes[t].values for it in batch])) for t in task_ids}
        ms = {t: jnp.asarray(np.stack([it.outcomes[t].mask.astype(np.float64) for it in batch]), dtype=_float_dtype()) for t in task_ids}
        n_dev = 1
        if mesh is not None:
            n_dev = mesh.shape["data"]
            if len(batch) % n_dev:
                raise UnsupportedCapability(f"batch of {len(batch)} not divisible by the data axis size {n_dev}; pad explicitly with zero-weight masks or change batch_size (no silent drop)")
            shard = NamedSharding(mesh, P("data"))
            repl = NamedSharding(mesh, P())
            stacked = {k: jax.device_put(v, shard) for k, v in stacked.items()}
            ys = {t: jax.device_put(v, shard) for t, v in ys.items()}
            ms = {t: jax.device_put(v, shard) for t, v in ms.items()}
            train = jax.device_put(train, repl)

        def item_terms(tp, consts_i, y_i, m_i):
            computed = self._forward(consts_i, node_states, edge_params, tp)
            terms = self._task_terms(computed, consts_i, y_i, m_i, tasks, weights)
            return {t: jnp.stack([n, d]) for t, (n, d) in terms.items()}

        def objective(tp):
            per_item = jax.vmap(item_terms, in_axes=(None, 0, 0, 0))(tp, stacked, ys, ms)  # {task: [B, 2]}
            total = 0.0
            for t, nd in per_item.items():
                total = total + weights.get(t, 1.0) * jnp.sum(nd[:, 0]) / jnp.maximum(jnp.sum(nd[:, 1]), 1e-12)
            return total

        loss, grads = jax.jit(jax.value_and_grad(objective))(train)
        return self._finish(loss, grads, {"mode": "batched_sharded" if mesh is not None else "batched", "batch_size": len(batch),
                                          "data_axis_size": n_dev, "owned_groups": list(owned_groups)})

    @staticmethod
    def _finish(loss, grads, diag):
        np_grads = {gid: {n: np.asarray(g, dtype=np.float64) for n, g in grp.items()} for gid, grp in grads.items()}
        gnorm = float(np.sqrt(sum(float(np.sum(g**2)) for grp in np_grads.values() for g in grp.values())))
        if not np.isfinite(float(loss)) or not np.isfinite(gnorm):
            raise ContractViolation("non-finite loss or gradient; refusing to update")
        return float(loss), np_grads, {"grad_norm": gnorm, **diag}

    # ------------------------------------------------------------------ parity
    def forward_parity(self, record: ExecutionRecord, node_states, edge_params) -> float:
        computed = self.traced_forward(record, node_states, edge_params, {})
        worst = 0.0
        for key, val in computed.items():
            eager = np.asarray(record.port_values[key].dense(), dtype=np.float64)
            worst = max(worst, float(np.max(np.abs(np.asarray(val) - eager))) if eager.size else 0.0)
        return worst


class BPDirectRule:
    """Learning profile ``bp_direct`` (== ``ff_bp``): exact BP through the deployed direct
    computation. Options: ``batched`` (vmap) and ``data_parallel`` (shard the batch over the
    selected devices; requires uniform shapes)."""

    rule_id = "bp_direct"
    rule_version = "1.1.0"
    plugin_id = "bp_direct"
    plugin_version = "1.1.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.task_weights = {k: float(v) for k, v in self.config.get("task_weights", {}).items()}
        self.batched = bool(self.config.get("batched", False))
        self.data_parallel = bool(self.config.get("data_parallel", False))

    def describe(self) -> ModuleDescriptor:
        return ModuleDescriptor(self.plugin_id, self.plugin_version, "learning_rule", "jax",
                                capabilities={"derivative": "reverse_mode_exact", "through": "direct_forward", "settling": None,
                                              "batched": self.batched, "data_parallel": self.data_parallel},
                                state_schema_id="ness.learning_rule.bp_direct/1", requires=("jax",))

    def required_runtime(self) -> str | None:
        return "jax"

    def algorithm_spec(self, compiled: Any = None, owned_groups: tuple[str, ...] = ()) -> dict[str, Any]:
        return {"learning_rule": "bp_direct (PDF ff_bp)", "derivative": "jax reverse-mode through the deployed direct computation of the differentiable region",
                "state": [], "clamps": [], "loss": "NESS task loss", "reductions": "sum numerators / sum denominators per task, weighted across tasks",
                "evaluation": "data_parallel" if self.data_parallel else ("batched" if self.batched else "per_item"), "owned_groups": list(owned_groups)}

    def gradients(self, ctx: LearningContext, batch: list[BatchItem], owned_groups: tuple[str, ...]):
        region = ctx.caches.get("jax_region")
        if region is None:
            region = ctx.caches["jax_region"] = DifferentiableRegion(ctx.compiled)
        plan = getattr(ctx, "objectives", None)
        if plan is not None and plan.uses_region:
            if self.data_parallel or self.batched:
                raise UnsupportedCapability("declared task/port_target objectives are verified in per-item evaluation only")
            return region.loss_and_grads(batch, ctx.node_states, ctx.edge_params, owned_groups, ctx.tasks, self.task_weights, plan)
        if self.data_parallel:
            devices = None
            rep = ctx.runtime_report
            if rep is not None and rep.devices is not None and rep.devices.selected:
                ids = {d.id for d in rep.devices.selected}
                devices = [d for d in jax.devices() if d.id in ids]
            mesh = sharded_mesh(devices)
            loss, grads, diag = region.loss_and_grads_batched(batch, ctx.node_states, ctx.edge_params, owned_groups, ctx.tasks, self.task_weights, mesh)
            diag["realized_devices"] = [int(d.id) for d in mesh.devices.flatten()]
            return loss, grads, diag
        if self.batched:
            return region.loss_and_grads_batched(batch, ctx.node_states, ctx.edge_params, owned_groups, ctx.tasks, self.task_weights, None)
        return region.loss_and_grads(batch, ctx.node_states, ctx.edge_params, owned_groups, ctx.tasks, self.task_weights)
