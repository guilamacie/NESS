"""Experiment runner: paired arms on identical request streams across model seeds, prequential
training, frozen evaluation, in-run checkpoint publish + restore comparison, causal check,
timing/memory accounting, structured raw artifacts, and configuration-only substitution
proofs. Every number needed for plots/reports is saved in machine-readable form."""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..checkpoint import CheckpointStore
from ..contracts import ContractViolation
from ..plugin_api.registry import PluginRegistry
from ..plugin_api.testkit import assert_target_free
from ..runtime.system import NessSystem, dependency_lock, runtime_request_for
from ..runtime.transaction import PredictionTransaction
from ..runtimes.bootstrap import bootstrap, current_report
from .artifacts import ArtifactWriter, peak_rss_mb
from .spec import ArmSpec, ExperimentSpec, parse_arm


@dataclass
class ArmReport:
    arm_id: str
    seed: int
    status: str = "ok"
    error: str | None = None
    manifest_id: str = ""
    n_updates: int = 0
    train: dict[str, dict[str, float]] = field(default_factory=dict)
    test: dict[str, dict[str, float]] = field(default_factory=dict)
    per_request: list[dict[str, Any]] = field(default_factory=list)
    train_loss_trace: list[dict[str, Any]] = field(default_factory=list)
    costs: dict[str, float] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    peak_rss_mb: float = 0.0
    checkpoint_manifest: str | None = None
    restore_check: dict[str, Any] = field(default_factory=dict)
    causal_check: str = "n/a"
    resolved_wiring: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    node_versions: dict[str, list[str]] = field(default_factory=dict)
    memory_snapshot: str | None = None
    failures: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.arm_id}@s{self.seed}"


@dataclass
class ExperimentReport:
    protocol_id: str
    protocol_hash: str
    arms: dict[str, ArmReport] = field(default_factory=dict)
    paired: dict[str, dict[str, Any]] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    substitutions: list[dict[str, Any]] = field(default_factory=list)
    run_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"protocol_id": self.protocol_id, "protocol_hash": self.protocol_hash,
                "arms": {k: {kk: vv for kk, vv in v.__dict__.items() if kk != "per_request"} for k, v in self.arms.items()},
                "paired": self.paired, "runtime": self.runtime, "substitutions": self.substitutions, "run_dir": self.run_dir}

    def summary_table(self) -> str:
        lines = [f"protocol {self.protocol_id} ({self.protocol_hash[:12]})",
                 f"{'arm':36s} {'task':10s} {'test loss':>10s} {'test MAE':>9s} {'train loss':>11s} {'upd':>4s} {'n':>4s} {'restore':>9s} {'causal':>7s} {'s':>6s}"]
        for key, r in self.arms.items():
            if r.status != "ok":
                lines.append(f"{key:36s} FAILED: {r.error}")
                continue
            for task, t in r.test.items():
                tr = r.train.get(task, {}).get("loss", float("nan"))
                rc = r.restore_check.get("max_abs_diff", float("nan"))
                lines.append(f"{key:36s} {task:10s} {t['loss']:10.5f} {t.get('mae', float('nan')):9.5f} {tr:11.5f} {r.n_updates:4d} {int(t['n_requests']):4d} {rc:9.1e} {r.causal_check:>7s} {sum(r.seconds.values()):6.1f}")
        if self.paired:
            lines.append("paired test-loss differences vs the reference arm (same requests, same seed):")
            for key, p in self.paired.items():
                for task, d in p.items():
                    lines.append(f"  {key:34s} {task:10s} delta={d['delta']:+.5f} (mean per-request diff {d['mean_diff']:+.5f}, sd {d['sd_diff']:.5f}, n={d['n']}, ref={d['reference']})")
        if self.substitutions:
            ok = sum(1 for s in self.substitutions if s["overall"] == "pass")
            lines.append(f"substitutions: {ok}/{len(self.substitutions)} passed all applicable checks")
        return "\n".join(lines)


# ---------------------------------------------------------------------------- helpers
def core_source_hash(package_dir: Path | None = None) -> str:
    """sha256 over the core package sources (see ``ness.integrity``): proves arms and substitutions ran on identical core code."""
    from ..integrity import core_source_hash as _h
    return _h(package_dir)


def _aggregate(agg: dict[str, dict[str, float]], rows: list[dict[str, Any]], task_ids: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for t in task_ids:
        a = dict(agg.get(t, {}))
        maes = [r[f"{t}/mae"] for r in rows if f"{t}/mae" in r and np.isfinite(r[f"{t}/mae"])]
        mses = [r[f"{t}/mse"] for r in rows if f"{t}/mse" in r and np.isfinite(r[f"{t}/mse"])]
        a["mae"] = float(np.mean(maes)) if maes else float("nan")
        a["rmse"] = float(np.sqrt(np.mean(mses))) if mses else float("nan")
        out[t] = a
    return out


# ---------------------------------------------------------------------------- one arm
def run_arm(exp: ExperimentSpec, arm: ArmSpec, registry: PluginRegistry, scenario: Any, writer: ArtifactWriter | None,
            store: CheckpointStore | None, seed: int | None = None) -> ArmReport:
    seed = arm.seed if seed is None else seed
    arm = replace(arm, seed=seed)
    rep = ArmReport(arm.arm_id, seed)
    proto = exp.protocol
    t0 = time.time()
    system = NessSystem.build(exp, arm, registry, scenario=scenario)
    tasks = tuple(system.tasks.values())
    task_ids = [t.task_id for t in tasks]
    rep.resolved_wiring = system.compiled.resolved_wiring()
    rep.node_versions = {cn.spec.node_id: [cn.descriptor.plugin_id, cn.descriptor.plugin_version] for cn in system.compiled.nodes}
    rep.seconds["build"] = time.time() - t0
    log = writer.log if writer else (lambda m: None)
    log(f"[{rep.key}] built: nodes={system.compiled.node_ids()} trainable={system.trainable}")
    batch_size, max_updates = int(proto.get("batch_size", 8)), int(proto.get("updates", 10))
    train_budget = batch_size * max_updates  # every arm (trainable or frozen) sees the same train request stream
    train_split, test_split = proto.get("train_split", "train"), proto.get("test_split", "test")
    pred_rows: list[dict[str, Any]] = []
    ev_rows: list[dict[str, Any]] = []

    # ---- training (prequential)
    t1 = time.time()
    tx = PredictionTransaction(system, mode="prequential")
    n_updates = 0
    n_train_seen = 0
    for req in scenario.iter_requests(train_split, tasks, {"max_requests": proto.get("max_train_requests")}):
        if n_train_seen >= train_budget:
            break
        n_train_seen += 1
        rec = tx.predict(req)
        outcomes = {q.task_id: scenario.outcome(req.request_id, q.query_id) for q in req.queries}
        outcomes = {k: v for k, v in outcomes.items() if v is not None}
        ev = tx.reveal(req.request_id, outcomes)
        pred_rows.append(ArtifactWriter.prediction_row(arm.arm_id, seed, "train", req, rec.outputs, outcomes, ev.scores))
        ex = tx.execution_of(req.request_id)
        if ex is not None and writer is not None:
            ev_rows.extend(ArtifactWriter.evidence_rows(arm.arm_id, seed, "train", req, ex))
        if tx.pending_batch_size() >= batch_size:
            diag = tx.learn_step()
            if diag is not None:
                n_updates += 1
                rep.train_loss_trace.append({"update": n_updates, "loss": float(diag["loss"]), "grad_norm": diag.get("grad_norm"),
                                             **{f"rule/{k}/loss": v.get("loss") for k, v in diag.get("rules", {}).items()}})
    rep.train = _aggregate(tx.aggregate_scores(), [r for r in pred_rows if r["phase"] == "train"], task_ids)
    rep.n_updates = system.n_updates
    rep.seconds["train"] = time.time() - t1

    # ---- checkpoint publish + restore comparison (whole-system identity)
    t2 = time.time()
    restored = None
    if store is not None:
        mid = store.stage_and_publish(system.snapshot("learner"), rep.key, store.head(rep.key))
        rep.checkpoint_manifest = mid
        restored = NessSystem.from_snapshot(store.load(mid), registry, scenario)
        if writer is not None:
            writer.json(f"manifests/{rep.key}.json", {"manifest_id": mid, **system.manifest().canonical()})
    rep.seconds["checkpoint"] = time.time() - t2

    # ---- frozen evaluation (+ restore diff on the first N requests, + causal check on the first)
    t3 = time.time()
    te = PredictionTransaction(system, mode="frozen")
    n_restore = int(proto.get("restore_check_requests", 8))
    diffs: list[dict[str, Any]] = []
    first = True
    costs: dict[str, float] = {}
    for req in scenario.iter_requests(test_split, tasks, {"max_requests": proto.get("max_test_requests")}):
        rec = te.predict(req)
        for k, v in rec.costs.items():
            costs[k] = costs.get(k, 0.0) + v
        outcomes = {q.task_id: scenario.outcome(req.request_id, q.query_id) for q in req.queries}
        outcomes = {k: v for k, v in outcomes.items() if v is not None}
        ev = te.reveal(req.request_id, outcomes)
        pred_rows.append(ArtifactWriter.prediction_row(arm.arm_id, seed, "test", req, rec.outputs, outcomes, ev.scores))
        ex = te.execution_of(req.request_id)
        if ex is not None and writer is not None:
            ev_rows.extend(ArtifactWriter.evidence_rows(arm.arm_id, seed, "test", req, ex))
        if restored is not None and len(diffs) < n_restore:
            r2, _, _ = restored.predict(req)
            d = max(float(np.max(np.abs(rec.outputs[q.query_id].point() - r2.outputs[q.query_id].point()))) for q in req.queries)
            diffs.append({"arm": arm.arm_id, "seed": seed, "request_id": req.request_id, "origin": req.origin, "max_abs_diff": d})
        if first and proto.get("causal_check", True):
            try:
                assert_target_free(lambda r: {k: v.point() for k, v in system.predict(r)[0].outputs.items()}, req)
                rep.causal_check = "pass"
            except AssertionError as exc:
                rep.causal_check = "FAIL"
                rep.failures.append(f"causal check failed: {exc}")
            first = False
    rep.test = _aggregate(te.aggregate_scores(), [r for r in pred_rows if r["phase"] == "test"], task_ids)
    rep.per_request = pred_rows
    rep.costs = costs
    rep.seconds["eval"] = time.time() - t3
    rep.seconds["total"] = time.time() - t0
    rep.peak_rss_mb = peak_rss_mb()
    rep.manifest_id = system.manifest().manifest_id
    rep.memory_snapshot = system.memory.snapshot_id if system.memory else None
    if diffs:
        vals = [d["max_abs_diff"] for d in diffs]
        rep.restore_check = {"n": len(diffs), "max_abs_diff": float(max(vals)), "mean_abs_diff": float(np.mean(vals)),
                             "restored_manifest_matches": bool(restored is not None and restored.manifest().manifest_id == rep.checkpoint_manifest)}
    if writer is not None:
        writer.csv(f"predictions/{rep.key}.csv", pred_rows)
        writer.csv(f"traces/{rep.key}_evidence.csv", ev_rows)
        if diffs:
            writer.csv(f"metrics/checkpoint_restore_{rep.key}.csv", diffs)
    log(f"[{rep.key}] done: updates={rep.n_updates} test={rep.test} restore={rep.restore_check} causal={rep.causal_check} seconds={rep.seconds}")
    return rep


# ---------------------------------------------------------------------------- substitutions
CHECKS = ("validate", "predict", "causal", "train", "restore")


def run_substitution(exp: ExperimentSpec, sub: ArmSpec, registry: PluginRegistry, scenario: Any, store: CheckpointStore | None) -> dict[str, Any]:
    row: dict[str, Any] = {"id": sub.arm_id, "description": sub.description, "resolved_plugins": {}, **{c: "n/a" for c in CHECKS}}
    t0 = time.time()
    try:
        system = NessSystem.build(exp, sub, registry, scenario=scenario)
        row["resolved_plugins"] = {cn.spec.node_id: f"{cn.descriptor.plugin_id}@{cn.descriptor.plugin_version}" for cn in system.compiled.nodes}
        row["resolved_wiring"] = system.compiled.resolved_wiring()
        row["validate"] = "pass"
    except Exception as exc:
        row["validate"] = f"fail: {type(exc).__name__}: {str(exc)[:200]}"
        row["overall"] = "fail"
        row["seconds"] = time.time() - t0
        return row
    tasks = tuple(system.tasks.values())
    test_req = next(iter(scenario.iter_requests(exp.protocol.get("test_split", "test"), tasks, {"max_requests": 1})))
    try:
        res, _, _ = system.predict(test_req)
        assert all(np.all(np.isfinite(f.point())) for f in res.outputs.values())
        row["predict"] = "pass"
    except Exception as exc:
        row["predict"] = f"fail: {type(exc).__name__}: {str(exc)[:200]}"
    try:
        assert_target_free(lambda r: {k: v.point() for k, v in system.predict(r)[0].outputs.items()}, test_req)
        row["causal"] = "pass"
    except Exception as exc:
        row["causal"] = f"fail: {type(exc).__name__}: {str(exc)[:200]}"
    if system.trainable:
        try:
            tx = PredictionTransaction(system, "prequential")
            for req in scenario.iter_requests(exp.protocol.get("train_split", "train"), tasks, {"max_requests": 4}):
                tx.predict(req)
                tx.reveal(req.request_id, {q.task_id: scenario.outcome(req.request_id, q.query_id) for q in req.queries})
            d = tx.learn_step()
            assert d is not None and np.isfinite(d["loss"])
            row["train"] = "pass"
        except Exception as exc:
            row["train"] = f"fail: {type(exc).__name__}: {str(exc)[:200]}"
    if store is not None:
        try:
            mid = store.stage_and_publish(system.snapshot("learner"), f"sub:{sub.arm_id}", store.head(f"sub:{sub.arm_id}"))
            restored = NessSystem.from_snapshot(store.load(mid), registry, scenario)
            a = system.predict(test_req)[0].outputs[test_req.queries[0].query_id].point()
            b = restored.predict(test_req)[0].outputs[test_req.queries[0].query_id].point()
            np.testing.assert_array_equal(a, b)
            row["restore"] = "pass"
            row["restore_manifest"] = mid
        except Exception as exc:
            row["restore"] = f"fail: {type(exc).__name__}: {str(exc)[:200]}"
    row["overall"] = "pass" if all(row[c] in ("pass", "n/a") for c in CHECKS) else "fail"
    row["seconds"] = time.time() - t0
    return row


# ---------------------------------------------------------------------------- experiment
def run_experiment(exp: ExperimentSpec, registry: PluginRegistry, out_dir: str | Path | None = None, arms: tuple[str, ...] | None = None,
                   with_substitutions: bool = True, only_substitutions: bool = False) -> ExperimentReport:
    scenario = registry.create(exp.scenario.plugin, exp.scenario.config)  # shared: identical request streams across arms
    out = Path(out_dir) if out_dir else None
    writer = ArtifactWriter(out) if out else None
    store = CheckpointStore(out / "checkpoints") if out else None
    report = ExperimentReport(exp.protocol_id, exp.protocol_hash, run_dir=str(out) if out else None)
    selected = [a for a in exp.arms if arms is None or a.arm_id in arms]
    if not selected:
        raise ContractViolation(f"no arms selected from {[a.arm_id for a in exp.arms]}")
    substitutions = [parse_arm(d["id"], d) for d in exp.raw.get("substitutions", [])] if with_substitutions else []
    # Bootstrap ONCE for the most capable backend any selected arm (or substitution) needs.
    runtimes: set[str] = set()
    for a in selected + substitutions:
        for n in a.composition.enabled_nodes():
            d = registry.describe(n.plugin)
            runtimes.add("fabricpc" if "fabricpc" in d.requires else ("jax" if "jax" in d.requires else "numpy"))
    rt = bootstrap(runtime_request_for(exp, runtimes))
    report.runtime = rt.canonical()
    seeds = [int(s) for s in exp.protocol.get("seeds", [])] or None
    if writer is not None:
        import yaml
        writer.text("config/experiment.yaml", yaml.safe_dump(exp.raw, sort_keys=False))
        writer.json("config/resolved_protocol.json", {"protocol_id": exp.protocol_id, "protocol_hash": exp.protocol_hash, "arms": [a.arm_id for a in selected],
                                                        "seeds": seeds or [a.seed for a in selected], "substitutions": [s.arm_id for s in substitutions], "protocol": exp.protocol})
        writer.json("environment/runtime_report.json", report.runtime)
        writer.json("environment/dependency_lock.json", dependency_lock())
        writer.json("environment/plugins.json", [d.canonical() for d in registry.descriptors()])
        writer.json("environment/core_source_hash.json", {"sha256": core_source_hash(), "note": "hash of ness core sources (reference_plugins excluded) at run time"})
        writer.write_dataset(scenario, tuple(registry.create(t.plugin, {**t.config, "task_id": t.task_id}) for t in exp.tasks))
        writer.log(f"run start protocol={exp.protocol_id} hash={exp.protocol_hash} backend={rt.backend}")
    for arm in ([] if only_substitutions else selected):
        for seed in (seeds or [arm.seed]):
            try:
                rep = run_arm(exp, arm, registry, scenario, writer, store, seed)
            except Exception as exc:  # preserve failed arms in the report, never drop them
                rep = ArmReport(arm.arm_id, seed, "failed", f"{type(exc).__name__}: {exc}")
                if writer:
                    writer.log(f"[{arm.arm_id}@s{seed}] FAILED {exc}\n{traceback.format_exc()}")
            report.arms[rep.key] = rep
    # paired contrasts vs the first selected arm, per seed
    ref_id = selected[0].arm_id
    for key, rep in report.arms.items():
        if rep.arm_id == ref_id or rep.status != "ok":
            continue
        ref = report.arms.get(f"{ref_id}@s{rep.seed}") or next((r for r in report.arms.values() if r.arm_id == ref_id and r.status == "ok"), None)
        if ref is None:
            continue
        rows = {r["request_id"]: r for r in rep.per_request if r["phase"] == "test"}
        ref_rows = {r["request_id"]: r for r in ref.per_request if r["phase"] == "test"}
        if set(rows) != set(ref_rows):
            rep.failures.append("evaluated on a different request set than the reference arm; paired contrast invalid")
            continue
        paired: dict[str, Any] = {}
        for task in rep.test:
            diffs = np.array([rows[r][f"{task}/mse"] - ref_rows[r][f"{task}/mse"] for r in ref_rows])
            paired[task] = {"delta": rep.test[task]["loss"] - ref.test[task]["loss"], "mean_diff": float(diffs.mean()),
                            "sd_diff": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0, "n": int(len(diffs)), "reference": ref.key}
        report.paired[key] = paired
    for sub in substitutions:
        row = run_substitution(exp, sub, registry, scenario, store)
        report.substitutions.append(row)
        if writer:
            writer.log(f"[substitution {sub.arm_id}] {row['overall']}: " + ", ".join(f"{c}={row[c]}" for c in CHECKS))
    if writer is not None and only_substitutions:
        # merge into an existing run directory: keep the arms' artifacts, refresh the substitution outputs
        writer.json("metrics/substitution_matrix.json", {"core_source_hash": core_source_hash(), "checks": list(CHECKS), "rows": report.substitutions})
        writer.csv("metrics/substitution_matrix.csv", [{"id": s["id"], "description": s["description"], **{c: s[c] for c in CHECKS}, "overall": s["overall"],
                                                        "resolved_plugins": s.get("resolved_plugins", {}), "seconds": s.get("seconds")} for s in report.substitutions])
        prev = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
        prev["substitutions"] = report.substitutions
        writer.json("report.json", prev)
        writer.log("substitutions refreshed")
        return report
    if writer is not None:
        summary = {k: {kk: vv for kk, vv in v.__dict__.items() if kk not in ("per_request",)} for k, v in report.arms.items()}
        writer.json("metrics/summary.json", {"protocol_id": exp.protocol_id, "protocol_hash": exp.protocol_hash, "reference_arm": ref_id, "arms": summary, "paired": report.paired})
        writer.csv("metrics/summary.csv", [{"arm": r.arm_id, "seed": r.seed, "status": r.status, "task": t, "test_loss": m.get("loss"), "test_mae": m.get("mae"), "test_rmse": m.get("rmse"),
                                             "train_loss": r.train.get(t, {}).get("loss"), "n_test": m.get("n_requests"), "updates": r.n_updates,
                                             "restore_max_abs_diff": r.restore_check.get("max_abs_diff"), "causal_check": r.causal_check, "seconds_total": r.seconds.get("total"),
                                             "peak_rss_mb": r.peak_rss_mb, "manifest_id": r.manifest_id, "checkpoint_manifest": r.checkpoint_manifest}
                                            for r in report.arms.values() for t, m in (r.test.items() if r.test else [("-", {})])])
        writer.csv("metrics/train_history.csv", [{"arm": r.arm_id, "seed": r.seed, **row} for r in report.arms.values() for row in r.train_loss_trace])
        writer.csv("metrics/timing.csv", [{"arm": r.arm_id, "seed": r.seed, "phase": ph, "seconds": s, "peak_rss_mb": r.peak_rss_mb} for r in report.arms.values() for ph, s in r.seconds.items()])
        writer.json("metrics/substitution_matrix.json", {"core_source_hash": core_source_hash(), "checks": list(CHECKS), "rows": report.substitutions})
        writer.csv("metrics/substitution_matrix.csv", [{"id": s["id"], "description": s["description"], **{c: s[c] for c in CHECKS}, "overall": s["overall"],
                                                        "resolved_plugins": s.get("resolved_plugins", {}), "seconds": s.get("seconds")} for s in report.substitutions])
        writer.json("report.json", report.to_dict())
        writer.text("summary.txt", report.summary_table())
        writer.log("run complete")
    return report
