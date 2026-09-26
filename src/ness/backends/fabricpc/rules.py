"""FabricPC learning rules. Both operate on ``fabricpc_residual_cap`` nodes only, read the
packed inputs from the recorded (target-free) predictions, and return gradients as
restricted arrays keyed exactly like the module state."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import ContractViolation, ModuleDescriptor, UnsupportedCapability
from ...learning.batch import BatchItem, LearningContext
from ...runtimes.bootstrap import ensure_bootstrapped
from . import ADAPTER_VERSION
from .translate import learning_spec, prediction_spec


def _target_task_for(ctx: LearningContext, node_id: str) -> str:
    for task_id, src in ctx.compiled.outputs.items():
        if src.producer_node == node_id:
            return task_id
    raise ContractViolation(f"FabricPC node {node_id} does not produce a task output; nothing to learn")


def _stack_batch(ctx: LearningContext, node_id: str, batch: list[BatchItem], task_id: str):
    cn = ctx.compiled.node(node_id)
    module = cn.module
    xs, bases, ys, ms = [], [], [], []
    for it in batch:
        if task_id not in it.outcomes:
            continue
        x, base = module.pack_from_record(it.record, cn)
        xs.append(x)
        bases.append(base)
        ys.append(it.outcomes[task_id].values)
        ms.append(it.outcomes[task_id].mask.astype(np.float64))
    if not xs:
        raise ContractViolation("no batch item carries an outcome for the FabricPC node's task")
    return module, np.stack(xs), np.stack(bases), np.stack(ys), np.stack(ms)


def _mesh(ctx: LearningContext, ad, data_parallel: bool):
    if not data_parallel:
        return None
    import jax
    rep = ctx.runtime_report
    devices = jax.devices()
    if rep is not None and rep.devices is not None and rep.devices.selected:
        ids = {d.id for d in rep.devices.selected}
        devices = [d for d in devices if d.id in ids]
    return ad.mesh_for(devices)


class _FabricPCRuleBase:
    rule_version = "1.0.0"
    plugin_version = "1.0.0"
    rule_id = plugin_id = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.data_parallel = bool(self.config.get("data_parallel", False))
        self._last_spec: dict[str, Any] | None = None

    def describe(self) -> ModuleDescriptor:
        return ModuleDescriptor(self.plugin_id, self.plugin_version, "learning_rule", "fabricpc", requires=("fabricpc", "jax"),
                                capabilities={"adapter_version": ADAPTER_VERSION, "data_parallel": self.data_parallel},
                                state_schema_id=f"ness.learning_rule.{self.plugin_id}/1")

    def required_runtime(self) -> str | None:
        return "fabricpc"

    def algorithm_spec(self, compiled: Any = None, owned_groups: tuple[str, ...] = ()) -> dict[str, Any] | None:
        """Deterministic from configuration: one translated spec per owned FabricPC node."""
        if compiled is None:
            return self._last_spec
        out: dict[str, Any] = {}
        for g in owned_groups:
            nid = g.split("/", 1)[0]
            module = compiled.node(nid).module
            if hasattr(module, "digest") and hasattr(module, "plan"):
                ad = module.adapter()
                out[nid] = learning_spec(prediction_spec(module.digest(), module.plan, {"fabricpc": ad.FABRICPC_VERSION, "adapter": ADAPTER_VERSION}), self.plugin_id).canonical()
        return out or None

    def _prepare(self, ctx: LearningContext, batch, owned_groups):
        ensure_bootstrapped("fabricpc")
        nodes = sorted({g.split("/", 1)[0] for g in owned_groups})
        for g in owned_groups:
            if "#" in g or not g.endswith("/workspace"):
                raise UnsupportedCapability(f"{self.plugin_id} can only own FabricPC workspace groups, not {g!r}")
        return nodes


class FabricPCPCLocalRule(_FabricPCRuleBase):
    """Clamp input + target correction, settle with the node's declared solver, then FabricPC's
    local weight gradients (batch-summed / prediction count). PDF ``workspace_pc_local`` (sPC)
    or ``workspace_epc_local`` (ePC). Objective reported = clamped total energy per prediction."""

    rule_id = plugin_id = "fabricpc_pc_local"

    def gradients(self, ctx: LearningContext, batch: list[BatchItem], owned_groups: tuple[str, ...]):
        nodes = self._prepare(ctx, batch, owned_groups)
        grads: dict[str, dict[str, np.ndarray]] = {}
        diag: dict[str, Any] = {"objective": "clamped_energy_per_prediction"}
        loss_total = 0.0
        for nid in nodes:
            task_id = _target_task_for(ctx, nid)
            module, x, base, y, m = _stack_batch(ctx, nid, batch, task_id)
            if not np.all(m == 1.0):
                raise UnsupportedCapability("fabricpc_pc_local: partially masked outcomes are not supported (FabricPC energies are per sample); use fabricpc_bp_through_inference")
            target = (y - base).reshape(y.shape[0], -1)  # correction-space target (residual writer)
            ad = module.adapter()
            params = module.graph_params(ctx.node_states[nid])
            structure = module.structure()
            settle = module.settles()
            mesh = _mesh(ctx, ad, self.data_parallel)
            if mesh is not None:
                if x.shape[0] % mesh.shape["data"]:
                    raise UnsupportedCapability(f"batch {x.shape[0]} not divisible by data axis {mesh.shape['data']} (no silent drop)")
                sharded = ad.shard_batch(mesh, {"x": x, "y": target})
                params_r = ad.replicate(mesh, params)
                fn = ad.jit(lambda p, xx, yy: ad.clamped_local_gradients(structure, p, xx, yy, settle))
                g, d = fn(params_r, sharded["x"], sharded["y"])
                diag["realized_devices"] = [int(dv.id) for dv in mesh.devices.flatten()]
            else:
                g, d = ad.clamped_local_gradients(structure, params, x, target, settle)
            grads[f"{nid}/workspace"] = ad.grads_to_arrays(g)
            loss_total += float(d["objective"])
            diag[nid] = {"clamped_energy_per_prediction": float(d["objective"]), "energy_initial": float(d["energy_initial"]), "denominator": int(d["denominator"])}
            self._last_spec = learning_spec(prediction_spec(module.digest(), module.plan, {"fabricpc": ad.FABRICPC_VERSION, "adapter": ADAPTER_VERSION}), self.plugin_id).canonical()
        gnorm = float(np.sqrt(sum(float(np.sum(a**2)) for grp in grads.values() for a in grp.values())))
        if not np.isfinite(gnorm):
            raise ContractViolation("non-finite FabricPC gradient")
        return loss_total, grads, {"grad_norm": gnorm, "batch_size": len(batch), "owned_groups": list(owned_groups), **diag}


class FabricPCBPThroughInferenceRule(_FabricPCRuleBase):
    """Reverse-mode gradient of the NESS task loss through the *deployed* target-free FabricPC
    inference (feedforward => PDF ``ff_bp``; settled => ``workspace_bp_unroll``). Loss is the
    task's sum-of-numerators / sum-of-denominators; masks are honoured."""

    rule_id = plugin_id = "fabricpc_bp_through_inference"

    def gradients(self, ctx: LearningContext, batch: list[BatchItem], owned_groups: tuple[str, ...]):
        import jax.numpy as jnp
        nodes = self._prepare(ctx, batch, owned_groups)
        grads: dict[str, dict[str, np.ndarray]] = {}
        diag: dict[str, Any] = {"objective": "task_loss"}
        loss_total = 0.0
        for nid in nodes:
            task_id = _target_task_for(ctx, nid)
            module, x, base, y, m = _stack_batch(ctx, nid, batch, task_id)
            task = ctx.tasks[task_id]
            ad = module.adapter()
            params = module.graph_params(ctx.node_states[nid])
            structure, settle = module.structure(), module.settles()
            base_j, y_j, m_j = jnp.asarray(base), jnp.asarray(y), jnp.asarray(m)

            def make_loss(base_j, y_j, m_j):
                def loss_fn(corr):
                    pred = base_j + jnp.reshape(corr, base_j.shape)
                    n, d = task.loss_terms(pred, y_j, m_j, jnp)
                    return n / jnp.maximum(d, 1e-12)
                return loss_fn

            mesh = _mesh(ctx, ad, self.data_parallel)
            if mesh is not None:
                if x.shape[0] % mesh.shape["data"]:
                    raise UnsupportedCapability(f"batch {x.shape[0]} not divisible by data axis {mesh.shape['data']} (no silent drop)")
                sharded = ad.shard_batch(mesh, {"x": x, "base": base, "y": y, "m": m})
                params_r = ad.replicate(mesh, params)
                fn = ad.jit(lambda p, xx, bb, yy, mm: ad.bp_through_inference(structure, p, xx, make_loss(bb, yy, mm), settle))
                loss, g, d = fn(params_r, sharded["x"], sharded["base"], sharded["y"], sharded["m"])
                diag["realized_devices"] = [int(dv.id) for dv in mesh.devices.flatten()]
            else:
                loss, g, d = ad.bp_through_inference(structure, params, x, make_loss(base_j, y_j, m_j), settle)
            grads[f"{nid}/workspace"] = ad.grads_to_arrays(g)
            loss_total += float(loss)
            diag[nid] = {"task_loss": float(loss), "energy_initial": float(d["energy_initial"]), "energy_final": float(d["energy_final"])}
            self._last_spec = learning_spec(prediction_spec(module.digest(), module.plan, {"fabricpc": ad.FABRICPC_VERSION, "adapter": ADAPTER_VERSION}), self.plugin_id).canonical()
        gnorm = float(np.sqrt(sum(float(np.sum(a**2)) for grp in grads.values() for a in grp.values())))
        if not np.isfinite(loss_total) or not np.isfinite(gnorm):
            raise ContractViolation("non-finite loss or gradient; refusing to update")
        return loss_total, grads, {"grad_norm": gnorm, "batch_size": len(batch), "owned_groups": list(owned_groups), **diag}
