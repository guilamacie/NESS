"""``NessSystem``: one arm's complete predictor - scenario, tasks, compiled composition,
module states, edge parameters, memory store/snapshot, inference profile, learning rule,
credit map and optimizer - with whole-system manifest, snapshot and restore."""

from __future__ import annotations

import copy
import platform
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import __version__
from ..composition import compile_graph
from ..composition.compiler import CompiledGraph
from ..contracts import (
    AvailabilityCut,
    ComponentRef,
    ContractViolation,
    CostLedger,
    ModuleState,
    PredictionRequest,
    PredictionResult,
    PredictorManifest,
    StateSnapshot,
    UnsupportedCapability,
    ValidationError,
    content_hash,
)
from ..checkpoint import SystemSnapshot
from ..experiments.spec import ArmSpec, ExperimentSpec, parse_arm, parse_experiment
from ..inference import resolve_inference_profile
from ..learning import CreditMap, LearningContext, Optimizer, build_credit_map, module_group_id, resolve_learning_profile
from ..learning.profiles import ALIASES as LEARNING_ALIASES
from ..memory import EMPTY_SNAPSHOT_ID, InMemorySnapshotStore, MemoryEvent, MemoryRecord
from ..plugin_api.module import RuntimeContext
from ..plugin_api.registry import PluginRegistry
from ..runtimes.bootstrap import RuntimeReport, RuntimeRequest, bootstrap, current_report
from .executor import ExecutionRecord, GraphExecutor

MANIFEST_SCHEMA = "ness.manifest/1"


def dependency_lock() -> dict[str, str]:
    """Exact numerical dependency versions from package metadata (never by importing them:
    reading a version must not initialise JAX or FabricPC)."""
    import importlib.metadata as md
    lock = {"python": platform.python_version(), "numpy": np.__version__, "ness": __version__}
    for name in ("jax", "jaxlib", "fabricpc", "jax-cuda12-plugin", "jax-cuda13-plugin", "optax"):
        try:
            lock[name] = md.version(name)
        except md.PackageNotFoundError:
            lock[name] = "absent"
    lock["hyperon"] = "unresolved"
    return lock


BACKEND_RANK = {"numpy": 0, "host": 0, "jax": 1, "fabricpc": 2}


def required_backend(runtimes: set[str]) -> str:
    """The single backend that must be bootstrapped for a set of module runtimes."""
    best = "numpy"
    for r in runtimes:
        if BACKEND_RANK.get(r, 0) > BACKEND_RANK[best]:
            best = r
    return best


def runtime_request_for(exp: "ExperimentSpec", runtimes: set[str]) -> RuntimeRequest:
    """Merge the experiment's ``runtime`` section with the backend the arms actually need.
    ``backend: auto`` (default) derives it; an explicit backend must be at least as capable."""
    cfg = dict(exp.runtime or {})
    need = required_backend(runtimes)
    declared = cfg.get("backend", "auto")
    if declared == "auto":
        cfg["backend"] = need
    elif BACKEND_RANK.get(declared, -1) < BACKEND_RANK[need]:
        raise ValidationError(f"runtime.backend={declared!r} cannot run modules that require {need!r}")
    return RuntimeRequest.from_config(cfg)


@dataclass
class MemoryAttachment:
    name: str
    store: InMemorySnapshotStore
    snapshot_id: str
    namespace: str
    learner_branch_id: str | None = None
    episode_window_field: str | None = None
    learner_appends: bool = False


class NessSystem:
    def __init__(self, exp: ExperimentSpec, arm: ArmSpec, registry: PluginRegistry, scenario: Any, tasks: dict[str, Any],
                 compiled: CompiledGraph, node_states: dict[str, ModuleState], edge_params: dict[str, dict[str, np.ndarray]],
                 rng: np.random.Generator, memory: MemoryAttachment | None, inference_profile: str, learning_profile: str,
                 credit: CreditMap, rules: dict[str, Any], optimizer: Optimizer | None, n_updates: int = 0,
                 runtime_report: RuntimeReport | None = None) -> None:
        self.exp, self.arm, self.registry = exp, arm, registry
        self.scenario, self.tasks, self.compiled = scenario, tasks, compiled
        self.node_states, self.edge_params, self.rng = node_states, edge_params, rng
        self.memory, self.inference_profile, self.learning_profile = memory, inference_profile, learning_profile
        self.credit, self.rules, self.optimizer, self.n_updates = credit, rules, optimizer, n_updates
        self.runtime_report = runtime_report
        self.executor = GraphExecutor(compiled)
        self._learning_caches: dict[str, Any] = {}
        self._manifest_cache: PredictorManifest | None = None

    @property
    def rule(self) -> Any:
        """The primary (default-owner) learning rule, or None (kept for callers of the first API)."""
        return self.rules.get(self.learning_profile)

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, exp: ExperimentSpec, arm: ArmSpec, registry: PluginRegistry, scenario: Any | None = None) -> "NessSystem":
        scenario = scenario if scenario is not None else registry.create(exp.scenario.plugin, exp.scenario.config)
        tasks = {t.task_id: registry.create(t.plugin, {**t.config, "task_id": t.task_id}) for t in exp.tasks}
        if not tasks:
            raise ValidationError("experiment declares no tasks")
        roles = frozenset.intersection(*(t.allowed_roles() for t in tasks.values()))
        spaces = {tid: t.prediction_space() for tid, t in tasks.items()}
        compiled = compile_graph(arm.composition, registry, scenario.observation_fields(), spaces, roles)
        # ---- runtime bootstrap: the ONLY place a system initialises a numerical backend, and it
        # happens before any module state (arrays) is created.
        runtimes = {cn.descriptor.runtime for cn in compiled.nodes}
        report = bootstrap(runtime_request_for(exp, runtimes))
        rng = np.random.default_rng(arm.seed)
        node_states = {cn.spec.node_id: cn.module.initialize(rng) for cn in compiled.nodes}
        edge_params = compiled.init_edge_params(rng)
        inference_profile = resolve_inference_profile(arm.inference.get("profile", "direct"), bool(arm.inference.get("allow_experimental", False)))
        rules, credit, optimizer, learning_profile = cls._build_learning(exp, arm, registry, compiled)
        memory = cls._attach_memory(arm, registry, scenario, exp)
        return cls(exp, arm, registry, scenario, tasks, compiled, node_states, edge_params, rng, memory, inference_profile, learning_profile, credit, rules, optimizer, 0, report)

    @staticmethod
    def _build_learning(exp: ExperimentSpec, arm: ArmSpec, registry: PluginRegistry, compiled: CompiledGraph):
        """Resolve the default rule plus per-group overrides into plugins; validate that every
        owner operates in the runtime of the parameters it owns (hybrid credit, PDF §11.6)."""
        trainable = [module_group_id(cn.spec.node_id, g.name) for cn in compiled.nodes for g in cn.descriptor.trainable_groups()] + list(compiled.edge_params)
        rule_name = arm.learning.get("rule")
        overrides_raw: dict[str, str] = dict(arm.learning.get("credit", {}).get("overrides", {}))
        if not trainable:
            profile = resolve_learning_profile(rule_name) if rule_name else "none"
            return {}, build_credit_map(compiled, "frozen"), None, profile
        if not rule_name:
            raise ValidationError(f"arm {arm.arm_id} has trainable parameter groups {trainable} but no learning.rule; declare one or freeze them")
        default_profile = resolve_learning_profile(rule_name)
        overrides = {gid: resolve_learning_profile(r) for gid, r in overrides_raw.items()}
        rule_ids = {default_profile, *overrides.values()}
        rules: dict[str, Any] = {}
        for rid in sorted(rule_ids):
            cfg = dict(arm.learning.get("rule_config", {}))
            cfg.update(arm.learning.get("rule_configs", {}).get(rid, {}))
            rules[rid] = registry.create(rid, cfg)
        rule_runtimes = {rid: r.required_runtime() for rid, r in rules.items()}
        credit = build_credit_map(compiled, default_profile, overrides, rule_runtimes)
        node_runtimes = {cn.descriptor.runtime for cn in compiled.nodes}
        for rid, need in rule_runtimes.items():
            if need is not None and need not in node_runtimes:
                raise UnsupportedCapability(f"learning rule {rid} requires runtime {need!r}; the composition has runtimes {sorted(node_runtimes)}")
        optimizer = Optimizer.from_config(arm.learning.get("optimizer", {"kind": "adam", "lr": 1e-2}))
        return rules, credit, optimizer, default_profile

    @staticmethod
    def _attach_memory(arm: ArmSpec, registry: PluginRegistry, scenario: Any, exp: ExperimentSpec, imported: tuple[dict, dict] | None = None) -> MemoryAttachment | None:
        m = arm.memory
        if not m or not m.get("store"):
            return None  # truly optional: no store instantiated, nothing initialised
        store_spec = m["store"]
        plugin = store_spec if isinstance(store_spec, str) else store_spec["plugin"]
        store = registry.create(plugin, {} if isinstance(store_spec, str) else store_spec.get("config", {}))
        namespace = m.get("namespace", "episodes")
        name = m.get("name", "episodes")
        if imported is not None:
            snap = store.import_snapshot(*imported)
            snapshot_id = snap.snapshot_id
        elif m.get("seed_from_scenario", True) and hasattr(scenario, "matured_episodes"):
            train_split = exp.protocol.get("train_split", "train")
            origins = scenario.origins(train_split)
            limit = exp.protocol.get("max_train_requests")
            last = origins[: int(limit)][-1] if limit else origins[-1]
            records = scenario.matured_episodes(last + 1, namespace)
            branch = store.fork(EMPTY_SNAPSHOT_ID, f"seed:{arm.arm_id}")
            store.append(branch, tuple(MemoryEvent(f"seed:{r.record_id}", "append", r) for r in records), branch.head)
            snapshot_id = store.seal(branch).snapshot_id
        else:
            snapshot_id = EMPTY_SNAPSHOT_ID
        return MemoryAttachment(name, store, snapshot_id, namespace, None, m.get("episode_window_field", "target_history"), bool(m.get("learner_appends", False)))

    # ------------------------------------------------------------------ predict
    def _policy_for(self, request: PredictionRequest):
        policies = {q.query_id: self.tasks[q.task_id].permitted_view(request.bundle, q) for q in request.queries}
        distinct = {(p.allowed_roles, p.cut, p.allowed_fields) for p in policies.values()}
        if len(distinct) > 1:
            raise UnsupportedCapability("tasks with different information boundaries need separate state solves; not supported by the direct profile yet")
        return policies

    def request_rng(self, request_id: str) -> np.random.Generator:
        return np.random.default_rng(int(content_hash({"seed": self.arm.seed, "request": request_id})[:16], 16))

    def predict(self, request: PredictionRequest, mode: str = "predict") -> tuple[PredictionResult, ExecutionRecord, RuntimeContext]:
        policies = self._policy_for(request)
        policy = next(iter(policies.values()))
        permitted = request.bundle.select(policy)
        permitted.assert_prediction_safe()
        ledger = CostLedger()
        views: dict[str, Any] = {}
        services: dict[str, Any] = {}
        if self.memory is not None:
            views[self.memory.name] = self.memory.store.open_view(self.memory.snapshot_id, AvailabilityCut(request.origin), policy.allowed_memory_namespaces or None)
            services[f"memory:{self.memory.name}"] = self.memory.store
        manifest_id = self.manifest().manifest_id
        ctx = RuntimeContext(request, permitted, policy, manifest_id, self.request_rng(request.request_id), ledger, views, mode, services=services)
        scen = self.scenario.describe()
        record = self.executor.run(ctx, self.node_states, self.edge_params, (scen.plugin_id, scen.plugin_version))
        outputs, statuses = {}, {}
        for q in request.queries:
            pv = record.outputs[q.task_id]
            self.tasks[q.task_id].prediction_space().validate(pv.payload)
            outputs[q.query_id] = pv.payload
            statuses[q.query_id] = "ok"
        diagnostics = {"resolved_sources": record.resolved_sources, "node_versions": record.node_versions, "node_diagnostics": record.node_diagnostics,
                       "stop_edges": list(self.compiled.stop_edges), "memory_views": {k: v.view_id for k, v in views.items()}, "inference_profile": self.inference_profile,
                       "runtime": self.runtime_diagnostics()}
        result = PredictionResult(request.request_id, manifest_id, outputs, statuses, diagnostics, record.evidence_graph.version_hash(), ledger.snapshot(), policies)
        return result, record, ctx

    # ------------------------------------------------------------------ learn
    @property
    def trainable(self) -> bool:
        return bool(self.rules)

    def owned_groups(self, rule_id: str | None = None) -> tuple[str, ...]:
        rid = rule_id or self.learning_profile
        return self.credit.groups_owned_by(rid) if self.rules else ()

    def region(self):
        """The JAX differentiable region (only meaningful when the graph has jax nodes)."""
        region = self._learning_caches.get("jax_region")
        if region is None:
            from ..backends.jax.region import DifferentiableRegion
            region = self._learning_caches["jax_region"] = DifferentiableRegion(self.compiled)
        return region

    def learning_context(self) -> LearningContext:
        weights = {t.task_id: t.weight for t in self.exp.tasks}
        return LearningContext(self.compiled, self.node_states, self.edge_params, self.tasks, weights, self.runtime_report, self._learning_caches)

    def all_params(self) -> dict[str, dict[str, np.ndarray]]:
        out = {module_group_id(nid, g): grp for nid, st in self.node_states.items() for g, grp in st.params.items()}
        out.update(self.edge_params)
        return out

    def learn(self, batch: list[Any]) -> dict[str, Any]:
        """One synchronized update: every rule computes gradients for the groups it owns from
        the same pre-update snapshot; one optimizer step applies all of them (PDF §11.6)."""
        if not self.trainable:
            raise UnsupportedCapability(f"arm {self.arm.arm_id} has no learning rule; learn() is not available")
        ctx = self.learning_context()
        grads: dict[str, dict[str, np.ndarray]] = {}
        diag: dict[str, Any] = {"rules": {}}
        total_loss = 0.0
        for rid, rule in self.rules.items():
            owned = self.owned_groups(rid)
            if not owned:
                continue
            loss, g, d = rule.gradients(ctx, batch, owned)
            grads.update(g)
            diag["rules"][rid] = {"loss": loss, **d}
            total_loss = total_loss if rid != self.learning_profile else loss
        assert self.optimizer is not None
        updated = self.optimizer.update({g: self.all_params()[g] for g in grads}, grads)
        for gid, grp in updated.items():
            if "#" in gid:
                self.edge_params[gid] = grp
            else:
                node, group = gid.split("/", 1)
                self.node_states[node].params[group] = grp
        self.n_updates += 1
        self._manifest_cache = None
        primary = diag["rules"].get(self.learning_profile, next(iter(diag["rules"].values()), {}))
        return {"loss": primary.get("loss", total_loss), "update": self.n_updates, "grad_norm": primary.get("grad_norm"), **diag}

    # ------------------------------------------------------------------ identity
    def component_snapshots(self) -> dict[str, StateSnapshot]:
        snaps = {cn.spec.node_id: cn.module.snapshot_state(self.node_states[cn.spec.node_id]) for cn in self.compiled.nodes}
        for tid, t in self.tasks.items():
            snaps[f"task:{tid}"] = t.snapshot_state(ModuleState())
        snaps["scenario"] = self.scenario.snapshot_state(ModuleState())
        return snaps

    def manifest(self) -> PredictorManifest:
        if self._manifest_cache is not None:
            return self._manifest_cache
        comps = []
        for cn in self.compiled.nodes:
            snap = cn.module.snapshot_state(self.node_states[cn.spec.node_id])
            comps.append(ComponentRef(cn.spec.node_id, cn.descriptor.plugin_id, cn.descriptor.plugin_version, cn.descriptor.module_kind,
                                      cn.descriptor.config_hash, snap.content_hash(), cn.descriptor.state_schema_id, cn.descriptor.runtime))
        edge_hash = content_hash({gid: {n: a for n, a in grp.items()} for gid, grp in self.edge_params.items()})
        comps.append(ComponentRef("composition:boundaries", "composition", "1", "cap", self.compiled.composition_hash, edge_hash, "ness.edge_params/1", self.compiled.differentiable_runtime or "none"))
        tasks = tuple(ComponentRef(f"task:{tid}", t.describe().plugin_id, t.describe().plugin_version, "task", t.config_hash, "", t.state_schema_id, "host") for tid, t in self.tasks.items())
        sd = self.scenario.describe()
        scen = ComponentRef("scenario", sd.plugin_id, sd.plugin_version, "scenario", sd.config_hash, "", sd.state_schema_id, "host")
        self._manifest_cache = PredictorManifest(
            MANIFEST_SCHEMA, self.exp.protocol_id, self.arm.arm_id, self.compiled.composition_hash, tuple(comps), tasks, scen,
            self.memory.snapshot_id if self.memory else None, self.inference_profile, self.learning_profile, self.credit.credit_hash,
            content_hash(self.rng.bit_generator.state), dependency_lock(), None,
            {"n_updates": self.n_updates, "resolved_wiring": self.compiled.resolved_wiring(), "runtime": self._runtime_extra(),
             "algorithm_specs": self._algorithm_specs(), "learning_rules": sorted(self.rules)})
        return self._manifest_cache

    def _runtime_extra(self) -> dict[str, Any]:
        """Identity-relevant runtime facts only (backend, numerical versions, platform, x64).
        Device ids/kinds, environment flags and notes are diagnostics (experiment report,
        prediction diagnostics), not predictor identity: a checkpoint must restore on another
        machine of the same platform class; a different FabricPC/JAX version must not."""
        rep = self.runtime_report or current_report()
        if rep is None:
            return {"backend": "numpy"}
        v = rep.versions
        return {"backend": rep.backend, "x64": rep.request.x64, "platform": rep.devices.platform if rep.devices else "none",
                "jax": v.get("jax", "absent"), "jaxlib": v.get("jaxlib", "absent"), "fabricpc": v.get("fabricpc", "absent"),
                "fabricpc_adapter": __import__("ness.backends.fabricpc", fromlist=["ADAPTER_VERSION"]).ADAPTER_VERSION if rep.backend == "fabricpc" else None}

    def runtime_diagnostics(self) -> dict[str, Any]:
        rep = self.runtime_report or current_report()
        return rep.canonical() if rep else {"backend": "numpy"}

    def _algorithm_specs(self) -> dict[str, Any]:
        """Every node that declares a complete AlgorithmSpec (FabricPC workspaces) is recorded;
        rule plugins add their own translation summary."""
        out: dict[str, Any] = {}
        for cn in self.compiled.nodes:
            spec = getattr(cn.module, "algorithm_spec", None)
            if callable(spec):
                out[cn.spec.node_id] = spec()
        for rid, rule in self.rules.items():
            spec = getattr(rule, "algorithm_spec", None)
            if callable(spec):
                out[f"rule:{rid}"] = spec(self.compiled, self.owned_groups(rid))
        return out

    def snapshot(self, kind: str = "serving") -> SystemSnapshot:
        mem = self.memory.store.export_snapshot(self.memory.snapshot_id) if self.memory else None
        opt = self.optimizer.snapshot() if (kind == "learner" and self.optimizer is not None) else None
        return SystemSnapshot(self.manifest(), self.component_snapshots(), {g: dict(v) for g, v in self.edge_params.items()}, opt,
                              {"bit_generator": self.rng.bit_generator.state, "n_updates": self.n_updates},
                              mem, {"experiment": copy.deepcopy(self.exp.raw), "arm_id": self.arm.arm_id, "arm": copy.deepcopy(self.arm.raw)}, kind)

    @classmethod
    def from_snapshot(cls, snap: SystemSnapshot, registry: PluginRegistry, scenario: Any | None = None) -> "NessSystem":
        exp = parse_experiment(snap.spec["experiment"])
        arm_id = snap.spec["arm_id"]
        if any(a.arm_id == arm_id for a in exp.arms):
            arm = exp.arm(arm_id)
        elif snap.spec.get("arm"):
            arm = parse_arm(arm_id, snap.spec["arm"])  # e.g. a substitution arm outside the experiment's main arm list
        else:
            raise ContractViolation(f"checkpoint arm {arm_id!r} is not in the experiment and no arm specification was stored")
        scenario = scenario if scenario is not None else registry.create(exp.scenario.plugin, exp.scenario.config)
        tasks = {t.task_id: registry.create(t.plugin, {**t.config, "task_id": t.task_id}) for t in exp.tasks}
        roles = frozenset.intersection(*(t.allowed_roles() for t in tasks.values()))
        spaces = {tid: t.prediction_space() for tid, t in tasks.items()}
        compiled = compile_graph(arm.composition, registry, scenario.observation_fields(), spaces, roles)
        if compiled.composition_hash != snap.manifest.composition_hash:
            raise ContractViolation("checkpoint composition hash differs from the recompiled composition; refusing to restore")
        report = bootstrap(runtime_request_for(exp, {cn.descriptor.runtime for cn in compiled.nodes}))
        node_states = {}
        for cn in compiled.nodes:
            nid = cn.spec.node_id
            if nid not in snap.components:
                raise ContractViolation(f"checkpoint lacks state for node {nid}")
            node_states[nid] = cn.module.restore_state(snap.components[nid])  # plugin chosen by manifest; state is data
        edge_params = {gid: {n: np.array(a) for n, a in grp.items()} for gid, grp in snap.edge_params.items()}
        if set(edge_params) != set(compiled.edge_params):
            raise ContractViolation("checkpoint edge parameter groups do not match the compiled composition")
        rng = np.random.default_rng(arm.seed)
        rng.bit_generator.state = snap.rng_state["bit_generator"]
        inference_profile = resolve_inference_profile(arm.inference.get("profile", "direct"), bool(arm.inference.get("allow_experimental", False)))
        rules, credit, optimizer, learning_profile = cls._build_learning(exp, arm, registry, compiled)
        if rules and snap.optimizer:
            optimizer = Optimizer.restore(*snap.optimizer)
        memory = cls._attach_memory(arm, registry, scenario, exp, imported=snap.memory) if snap.memory is not None else None
        sys_ = cls(exp, arm, registry, scenario, tasks, compiled, node_states, edge_params, rng, memory, inference_profile, learning_profile, credit, rules, optimizer, int(snap.rng_state.get("n_updates", 0)), report)
        if sys_.manifest().manifest_id != snap.manifest_id:
            raise ContractViolation("restored system manifest id differs from checkpoint manifest id")
        return sys_

    # ------------------------------------------------------------------ memory learner branch
    def memory_learner_append(self, records: tuple[MemoryRecord, ...]) -> None:
        """Append matured records to an isolated learner branch (never to the pinned snapshot)."""
        if self.memory is None or not records:
            return
        m = self.memory
        if m.learner_branch_id is None:
            m.learner_branch_id = f"learner:{self.arm.arm_id}:{m.snapshot_id[:8]}:{self.n_updates}"
            self._branch = m.store.fork(m.snapshot_id, m.learner_branch_id)
        events = tuple(MemoryEvent(f"exp:{r.record_id}", "append", r) for r in records)
        m.store.append(self._branch, events, self._branch.head)

    def memory_publish(self) -> str | None:
        """Seal the learner branch into a new snapshot and switch *future* predictions to it.
        In-flight predictions keep their pinned views (memory is pinned per transaction)."""
        if self.memory is None or self.memory.learner_branch_id is None:
            return None
        manifest = self.memory.store.seal(self._branch)
        self.memory.snapshot_id = manifest.snapshot_id
        self.memory.learner_branch_id = None
        self._manifest_cache = None
        return manifest.snapshot_id
