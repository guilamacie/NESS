"""``AnalogueMemoryQuery``: the host-side MemoryQueryPort. Reads a *pinned* view supplied
by the transaction (never a live store handle), runs one bounded analogue query, and
lowers the result to typed evidence."""

from __future__ import annotations

import numpy as np

from ..contracts import (
    AccessPolicyViolation,
    ContractViolation,
    Differentiability,
    FeatureEvidence,
    FieldRole,
    ModuleDescriptor,
    ModuleState,
    PortSchema,
    PortSpec,
    ProducerRef,
    Provenance,
    RetrievalEvidence,
    ApproximationStatus,
)
from ..memory import QueryPlan
from ..plugin_api.base import BaseModule
from ..plugin_api.module import ModuleOutputs, PortValue, RuntimeContext


class AnalogueMemoryQuery(BaseModule):
    plugin_id = "analogue_memory_query"
    plugin_version = "1.0.0"
    module_kind = "memory_query"
    runtime = "host"
    state_schema_id = "ness.memory_query.analogue/1"

    def validate_config(self) -> None:
        self.store = str(self.config.get("store", "episodes"))
        self.namespace = str(self.config.get("namespace", "episodes"))
        self.k = int(self.config.get("k", 3))
        self.horizon = int(self.config.get("horizon", 4))
        self.channels = int(self.config.get("channels", 2))                      # continuation channels (target)
        self.query_channels = int(self.config.get("query_channels", self.channels))  # window channels (may include auxiliaries)

    def describe(self) -> ModuleDescriptor:
        HD = self.horizon * self.channels
        return self._descriptor(
            input_ports=(PortSpec("query", PortSchema("dense", (None, self.query_channels)), "observation_series", FieldRole.OBSERVED, Differentiability.STOP,
                                  accepts_semantic_types=("observation_series",)),),
            output_ports=(
                PortSpec("evidence", PortSchema("feature", (HD,)), "analogue_evidence", FieldRole.DERIVED, Differentiability.STOP),
                PortSpec("retrieval", PortSchema("retrieval", ()), "analogue_retrieval", FieldRole.DERIVED, Differentiability.NONE),
            ),
            capabilities={"query_kind": "analogue", "store": self.store, "namespace": self.namespace, "k": self.k, "query_channels": self.query_channels, "continuation_channels": self.channels},
        )

    def initialize(self, rng: np.random.Generator) -> ModuleState:
        return ModuleState()

    def forward(self, inputs: dict[str, PortValue], state: ModuleState, ctx: RuntimeContext) -> ModuleOutputs:
        view = ctx.memory_views.get(self.store)
        store = ctx.services.get(f"memory:{self.store}")
        if view is None or store is None:
            raise ContractViolation(f"memory query node {ctx.node_id!r} requires memory store {self.store!r} attached with a pinned view; "
                                    "attach the store in the arm's memory section or disable this node")
        if self.namespace not in ctx.policy.allowed_memory_namespaces:
            raise AccessPolicyViolation(f"task policy {ctx.policy.policy_id} does not permit memory namespace {self.namespace!r}")
        q = np.asarray(inputs["query"].dense(), dtype=np.float64)
        plan = QueryPlan("analogue", self.namespace, {"query_window": q, "k": self.k, "window_key": "window"})
        res = store.query(view, plan, ctx.query_budget)
        HD = self.horizon * self.channels
        producer = ProducerRef(ctx.node_id, self.plugin_id, self.plugin_version)
        prov = Provenance(producer, (inputs["query"].evidence_id,), frozenset({FieldRole.DERIVED}), view.cutoff.origin,
                          (("query", inputs["query"].evidence_id),), f"analogue k={self.k} view={view.view_id}")
        if res.returned:
            conts = np.stack([r.arrays["continuation"].reshape(-1) for r in res.returned])
            if conts.shape[1] != HD:
                raise ContractViolation("analogue continuation shape does not match declared horizon x channels")
            value, known = conts.mean(0), np.ones(HD, dtype=bool)
        else:
            value, known = np.zeros(HD), np.zeros(HD, dtype=bool)  # no match is NOT a zero forecast
        approx = ApproximationStatus.TRUNCATED if res.truncated else ApproximationStatus.EXACT
        fe = FeatureEvidence(f"{ctx.node_id}/evidence@{ctx.request.request_id}", prov, "analogue_evidence", approx, value=value, knownness=known,
                             interpretation=f"mean matured continuation of top-{self.k} analogues")
        re = RetrievalEvidence(f"{ctx.node_id}/retrieval@{ctx.request.request_id}", prov, "analogue_retrieval", approx,
                               returned_ids=tuple(r.record_id for r in res.returned), candidate_count=res.candidate_count,
                               completeness=res.completeness, truncated=res.truncated, view_id=view.view_id, query_plan_hash=res.plan_hash)
        return ModuleOutputs(ports={"evidence": fe, "retrieval": re},
                             diagnostics={"returned": list(re.returned_ids), "candidates": res.candidate_count, "view_id": view.view_id, "scores": list(res.scores)},
                             cost=float(res.candidate_count))
