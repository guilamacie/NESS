"""Eager graph executor: runs compiled nodes in topological order under one request
context, wrapping outputs as provenance-carrying ``PortValue``s and appending typed
evidence to an immutable ``EvidenceGraph``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..composition.compiler import CompiledGraph, CompiledNode, ResolvedInput
from ..composition.selectors import OBSERVATION_NODE
from ..contracts import (
    ContractViolation,
    Differentiability,
    Evidence,
    EvidenceGraph,
    FeatureEvidence,
    FieldRole,
    Forecast,
    ModuleState,
    Observation,
    PortSchema,
    PortSpec,
    PredictiveEvidence,
    ProducerRef,
    Provenance,
    UnsupportedCapability,
)
from ..plugin_api.module import ModuleOutputs, PortValue, RuntimeContext
from .assembly import assemble_dense


@dataclass
class ExecutionRecord:
    port_values: dict[tuple[str, str], PortValue]
    evidence_graph: EvidenceGraph
    outputs: dict[str, PortValue]  # task_id -> forecast port value
    node_diagnostics: dict[str, dict[str, Any]]
    resolved_sources: dict[str, dict[str, list[str]]]
    node_versions: dict[str, tuple[str, str]]
    used_inputs: dict[str, tuple[str, ...]] = field(default_factory=dict)


class GraphExecutor:
    def __init__(self, compiled: CompiledGraph) -> None:
        self.compiled = compiled
        if compiled.spec.recurrent_regions:
            # Declared recurrent regions compile (T36) but no recurrent solver is verified yet.
            raise UnsupportedCapability(
                "recurrent InferenceWorkspace regions are declared-only in this release; no solver capability is verified")

    # ------------------------------------------------------------------ observation
    @staticmethod
    def _observation_value(obs: Observation, ctx: RuntimeContext, scenario_ref: tuple[str, str]) -> PortValue:
        spec = PortSpec(obs.field_name, PortSchema("dense", tuple(obs.array.shape)), obs.metadata.get("semantic_type", obs.modality),
                        obs.role, Differentiability.STOP, obs.coordinates.schema_id)
        eid = f"observation:{obs.field_name}@{ctx.request.request_id}"
        prov = Provenance(ProducerRef(OBSERVATION_NODE, scenario_ref[0], scenario_ref[1], obs.field_name), (), frozenset({obs.role}), obs.available_at, (), "observed")
        return PortValue(obs.array, spec, prov, eid)

    # ------------------------------------------------------------------ run
    def run(self, ctx: RuntimeContext, node_states: dict[str, ModuleState], edge_params: dict[str, dict[str, np.ndarray]],
            scenario_ref: tuple[str, str] = ("scenario", "0")) -> ExecutionRecord:
        bundle = ctx.permitted_bundle
        bundle.assert_prediction_safe()
        values: dict[tuple[str, str], PortValue] = {}
        graph = EvidenceGraph(f"evidence@{ctx.request.request_id}", (), tuple(sorted(v.view_id for v in ctx.memory_views.values() if hasattr(v, "view_id"))))
        for obs in bundle.observations:
            values[(OBSERVATION_NODE, obs.field_name)] = self._observation_value(obs, ctx, scenario_ref)
        graph = graph.with_fragment(tuple(self._as_evidence(values[(OBSERVATION_NODE, o.field_name)]) for o in bundle.observations))
        diagnostics: dict[str, dict[str, Any]] = {}
        used: dict[str, tuple[str, ...]] = {}
        versions: dict[str, tuple[str, str]] = {}

        for cn in self.compiled.nodes:
            if cn.training_only:
                continue  # never executed for a prediction (ADR-0012)
            nid = cn.spec.node_id
            inputs = {port: self._assemble(cn, port, ri, values, edge_params, ctx) for port, ri in cn.inputs.items()}
            # merged/transformed inputs are derived values: record them (graph + port map)
            for port, pv in inputs.items():
                if not cn.inputs[port].is_passthrough():
                    values[(nid, f"{port}#assembled")] = pv
            graph = graph.with_fragment(tuple(self._as_evidence(pv) for port, pv in inputs.items() if not cn.inputs[port].is_passthrough()))
            ctx.node_id = nid
            out = cn.module.forward(inputs, node_states[nid], ctx)
            ctx.ledger.charge(f"node/{nid}", out.cost)
            new_evidence: list[Evidence] = []
            dep_ids = tuple(inputs[p].evidence_id for p in (out.used_inputs if out.used_inputs is not None else tuple(inputs)))
            used[nid] = tuple(out.used_inputs if out.used_inputs is not None else tuple(inputs))
            roles = frozenset().union(*(inputs[p].provenance.roles for p in used[nid])) if used[nid] else frozenset({FieldRole.DERIVED})
            avail = [inputs[p].provenance.max_available_at for p in used[nid] if inputs[p].provenance.max_available_at is not None]
            resolved = tuple((p, s.selector) for p, ri in cn.inputs.items() for s in ri.sources)
            for pspec in cn.descriptor.output_ports:
                if pspec.name not in out.ports:
                    raise ContractViolation(f"node {nid}: module did not produce declared output port {pspec.name!r}")
                payload = out.ports[pspec.name]
                prov = Provenance(ProducerRef(nid, cn.descriptor.plugin_id, cn.descriptor.plugin_version, pspec.name), dep_ids,
                                  roles | {FieldRole.DERIVED}, max(avail) if avail else None, resolved, cn.descriptor.module_kind)
                pv = self._wrap_output(nid, pspec, payload, prov, ctx)
                values[(nid, pspec.name)] = pv
                new_evidence.append(self._as_evidence(pv))
            for extra in out.evidence:
                new_evidence.append(extra)
            graph = graph.with_fragment(tuple(new_evidence))
            diagnostics[nid] = dict(out.diagnostics)
            versions[nid] = (cn.descriptor.plugin_id, cn.descriptor.plugin_version)

        outputs: dict[str, PortValue] = {}
        for task_id, src in self.compiled.outputs.items():
            pv = values[(src.producer_node, src.producer_port.name)]
            if not isinstance(pv.payload, Forecast):
                raise ContractViolation(f"output for task {task_id} is not a Forecast (got {type(pv.payload).__name__})")
            outputs[task_id] = pv
        return ExecutionRecord(values, graph, outputs, diagnostics, self.compiled.resolved_wiring(), versions, used)

    def run_training_only(self, ctx: RuntimeContext, record: ExecutionRecord, node_states: dict[str, ModuleState],
                          edge_params: dict[str, dict[str, np.ndarray]], node_ids: set[str]) -> dict[tuple[str, str], PortValue]:
        """Execute the requested training-only nodes (and the training-only nodes they read) on the
        values a prediction recorded; returns their output port values. Never touches ``record``."""
        needed = set(node_ids)
        training_only = set(self.compiled.training_only_nodes)
        changed = True
        while changed:  # close over training-only producers
            changed = False
            for cn in self.compiled.nodes:
                if cn.spec.node_id in needed:
                    for ri in cn.inputs.values():
                        for s in ri.sources:
                            if s.producer_node in training_only and s.producer_node not in needed:
                                needed.add(s.producer_node)
                                changed = True
        values: dict[tuple[str, str], PortValue] = dict(record.port_values)
        out: dict[tuple[str, str], PortValue] = {}
        for cn in self.compiled.nodes:
            nid = cn.spec.node_id
            if nid not in needed:
                continue
            if not cn.training_only:
                raise ContractViolation(f"node {nid} is not training-only; its values come from the prediction record")
            inputs = {port: self._assemble(cn, port, ri, values, edge_params, ctx) for port, ri in cn.inputs.items()}
            ctx.node_id = nid
            res = cn.module.forward(inputs, node_states[nid], ctx)
            used = tuple(res.used_inputs if res.used_inputs is not None else tuple(inputs))
            dep_ids = tuple(inputs[p].evidence_id for p in used)
            roles = frozenset().union(*(inputs[p].provenance.roles for p in used)) if used else frozenset({FieldRole.DERIVED})
            resolved = tuple((p, s.selector) for p, ri in cn.inputs.items() for s in ri.sources)
            for pspec in cn.descriptor.output_ports:
                if pspec.name not in res.ports:
                    raise ContractViolation(f"training-only node {nid}: module did not produce declared output port {pspec.name!r}")
                prov = Provenance(ProducerRef(nid, cn.descriptor.plugin_id, cn.descriptor.plugin_version, pspec.name), dep_ids,
                                  roles | {FieldRole.DERIVED}, None, resolved, f"{cn.descriptor.module_kind} (training-only)")
                pv = self._wrap_output(nid, pspec, res.ports[pspec.name], prov, ctx)
                values[(nid, pspec.name)] = pv
                out[(nid, pspec.name)] = pv
        return out

    # ------------------------------------------------------------------ helpers
    def _assemble(self, cn: CompiledNode, port: str, ri: ResolvedInput, values: dict[tuple[str, str], PortValue],
                  edge_params: dict[str, dict[str, np.ndarray]], ctx: RuntimeContext) -> PortValue:
        src_values = [values[(s.producer_node, s.producer_port.name)] for s in ri.sources]
        if ri.is_passthrough():
            return src_values[0]
        dense = assemble_dense(ri, [np.asarray(v.dense(), dtype=np.float64) for v in src_values], edge_params, np)
        dense = np.asarray(dense, dtype=np.float64)
        mask = None
        if ri.merge == "concat" and not ri.post and all(not s.ref.boundary for s in ri.sources):
            ks = [v.knownness() for v in src_values]
            if all(k is not None for k in ks):
                mask = np.concatenate([np.asarray(k, dtype=bool).reshape(-1) for k in ks])
                if mask.shape != dense.shape:
                    mask = None
        deps = tuple(v.evidence_id for v in src_values)
        roles = frozenset().union(*(v.provenance.roles for v in src_values))
        avail = [v.provenance.max_available_at for v in src_values if v.provenance.max_available_at is not None]
        nid = cn.spec.node_id
        prov = Provenance(ProducerRef(nid, "composition", "1", f"{port}#assembled"), deps, roles, max(avail) if avail else None,
                          tuple((port, s.selector) for s in ri.sources), f"merge={ri.merge} post={[b.kind for b in ri.post]}")
        spec = PortSpec(f"{port}#assembled", PortSchema("dense", tuple(dense.shape)), "assembled", FieldRole.DERIVED, Differentiability.STOP)
        return PortValue(dense, spec, prov, f"{nid}/{port}#assembled@{ctx.request.request_id}", mask)

    @staticmethod
    def _wrap_output(nid: str, pspec: PortSpec, payload: Any, prov: Provenance, ctx: RuntimeContext) -> PortValue:
        eid = payload.evidence_id if isinstance(payload, Evidence) else f"{nid}/{pspec.name}@{ctx.request.request_id}"
        if isinstance(payload, np.ndarray):
            payload = np.asarray(payload, dtype=np.float64)
        elif not isinstance(payload, (Evidence, Forecast)):
            payload = np.asarray(payload, dtype=np.float64)
        if isinstance(payload, np.ndarray) and pspec.schema.shape:
            if len(payload.shape) != len(pspec.schema.shape) or any(d is not None and d != s for d, s in zip(pspec.schema.shape, payload.shape)):
                raise ContractViolation(f"node {nid} port {pspec.name}: produced shape {payload.shape}, declared {pspec.schema.shape}")
        return PortValue(payload, pspec, prov, eid)

    @staticmethod
    def _as_evidence(pv: PortValue) -> Evidence:
        p = pv.payload
        if isinstance(p, Evidence):
            return p
        if isinstance(p, Forecast):
            return PredictiveEvidence(pv.evidence_id, pv.provenance, pv.spec.semantic_type, forecast=p)
        return FeatureEvidence(pv.evidence_id, pv.provenance, pv.spec.semantic_type, value=np.asarray(p), interpretation="dense_port")
