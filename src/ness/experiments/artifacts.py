"""Raw run artifacts, independent of plots: everything needed to re-plot and re-report.

Layout under the run directory (see docs/ARCHITECTURE.md §7):
  config/ manifests/ checkpoints/ data/ predictions/ metrics/ traces/ plots/ logs/ environment/ report/
"""

from __future__ import annotations

import csv
import json
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import FeatureEvidence, HypothesisEvidence, RetrievalEvidence, ExecutionTrace

SUBDIRS = ("config", "manifests", "checkpoints", "data", "predictions", "metrics", "traces", "plots", "logs", "environment", "report")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class ArtifactWriter:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        for d in SUBDIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self._log = open(self.root / "logs" / "run.log", "a")

    # ------------------------------------------------------------------ generic
    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        self._log.write(line + "\n")
        self._log.flush()

    def json(self, rel: str, obj: Any) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=1, default=_json_default))
        return p

    def csv(self, rel: str, rows: list[dict[str, Any]], columns: list[str] | None = None) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            p.write_text("")
            return p
        cols = columns or sorted({k for r in rows for k in r})
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v, default=_json_default) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
        return p

    def text(self, rel: str, text: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    # ------------------------------------------------------------------ dataset
    def write_dataset(self, scenario: Any, tasks: tuple[Any, ...]) -> None:
        if hasattr(scenario, "ground_truth"):
            gt = scenario.ground_truth()
            keys = list(gt)
            rows = [{k: gt[k][i] for k in keys} for i in range(len(gt[keys[0]]))]
            self.csv("data/series.csv", rows, keys)
        if hasattr(scenario, "generation_manifest"):
            self.json("data/generation_manifest.json", scenario.generation_manifest())
        else:
            self.json("data/generation_manifest.json", {"plugin": scenario.describe().plugin_id, "config": dict(getattr(scenario, "config", {}))})
        self.json("data/splits.json", {s: list(scenario.origins(s)) for s in scenario.splits()} if hasattr(scenario, "origins") else {})
        self.json("data/observation_fields.json", {n: {"role": f.role.value, "semantic_type": f.semantic_type, "shape": list(f.shape), "coordinates": f.coordinates.schema_id}
                                                   for n, f in scenario.observation_fields().items()})
        rows = []
        for split in scenario.splits():
            for req in scenario.iter_requests(split, tasks):
                for q in req.queries:
                    oc = scenario.outcome(req.request_id, q.query_id)
                    if oc is None:
                        continue
                    r = {"request_id": req.request_id, "origin": req.origin, "split": split, "task": q.task_id, **{f"g_{k}": v for k, v in req.group.items()}}
                    for i, v in enumerate(np.asarray(oc.values).reshape(-1)):
                        r[f"truth_{i}"] = float(v)
                    rows.append(r)
        self.csv("data/ground_truth.csv", rows)

    # ------------------------------------------------------------------ per-request traces
    @staticmethod
    def prediction_row(arm: str, seed: int, phase: str, req: Any, outputs: dict[str, Any], outcomes: dict[str, Any], scores: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {"arm": arm, "seed": seed, "phase": phase, "request_id": req.request_id, "origin": req.origin, **{f"g_{k}": v for k, v in req.group.items()}}
        for q in req.queries:
            fc = outputs.get(q.query_id)
            if fc is None:
                continue
            pred = np.asarray(fc.point()).reshape(-1)
            for i, v in enumerate(pred):
                row[f"{q.task_id}/pred_{i}"] = float(v)
            oc = outcomes.get(q.task_id)
            if oc is not None:
                y = np.asarray(oc.values).reshape(-1)
                m = np.asarray(oc.mask).reshape(-1)
                for i in range(len(y)):
                    row[f"{q.task_id}/truth_{i}"] = float(y[i])
                    row[f"{q.task_id}/abs_err_{i}"] = float(abs(y[i] - pred[i])) if m[i] else float("nan")
                row[f"{q.task_id}/mae"] = float(np.mean(np.abs(y - pred)[m])) if m.any() else float("nan")
                row[f"{q.task_id}/mse"] = float(np.mean(((y - pred) ** 2)[m])) if m.any() else float("nan")
            sc = scores.get(q.task_id)
            if sc is not None:
                row[f"{q.task_id}/score_num"] = sc.numerator
                row[f"{q.task_id}/score_den"] = sc.denominator
        return row

    @staticmethod
    def evidence_rows(arm: str, seed: int, phase: str, req: Any, execution: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (node, port), pv in execution.port_values.items():
            if node == "observation":
                continue
            p = pv.payload
            base = {"arm": arm, "seed": seed, "phase": phase, "request_id": req.request_id, "origin": req.origin, "node": node, "port": port}
            if isinstance(p, HypothesisEvidence):
                rows.append({**base, "kind": "hypothesis", "alternatives": list(p.alternatives), "values": p.weights.tolist(), "semantics": p.weight_semantics,
                             "approximation": p.approximation.value, "sources": [s for _, s in pv.provenance.resolved_sources]})
            elif isinstance(p, FeatureEvidence):
                if p.value.size <= 64:
                    rows.append({**base, "kind": "feature", "values": p.value.tolist(), "knownness": p.knownness.astype(int).tolist(), "interpretation": p.interpretation,
                                 "approximation": p.approximation.value, "sources": [s for _, s in pv.provenance.resolved_sources]})
            elif isinstance(p, RetrievalEvidence):
                rows.append({**base, "kind": "retrieval", "returned_ids": list(p.returned_ids), "candidate_count": p.candidate_count, "completeness": p.completeness.value,
                             "truncated": p.truncated, "view_id": p.view_id, "approximation": p.approximation.value})
        for ev in execution.evidence_graph.nodes:
            if isinstance(ev, ExecutionTrace):
                rows.append({"arm": arm, "seed": seed, "phase": phase, "request_id": req.request_id, "origin": req.origin, "node": ev.provenance.producer.node_id, "port": "trace",
                             "kind": "execution_trace", "status": ev.status, "truth": ev.truth_status.value if ev.truth_status else None,
                             "completeness": ev.completeness.value if ev.completeness else None, "details": {k: v for k, v in ev.details.items() if k in ("fuel_used", "knownness", "inference_method", "log_evidence", "effective_sample_size", "error")}})
        return rows
