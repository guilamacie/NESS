"""Logical memory contract and the in-memory snapshot backend.

Snapshots are immutable and content-addressed. ``fork`` isolates; ``append`` needs an
expected head and idempotent event ids; ``seal`` produces a canonical export. The
numerical learner only ever receives bounded evidence produced under a pinned view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..contracts import (
    AvailabilityCut,
    Completeness,
    ContractViolation,
    MemoryCapabilities,
    MemoryConflict,
    QueryBudget,
    array_hash,
    content_hash,
)


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    record_id: str
    namespace: str
    kind: str                     # episode | fact | belief | program | experience | search
    available_at: int             # when the *whole record* (incl. matured continuation) became knowable
    event_span: tuple[int, int]
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    def canonical(self) -> dict:
        return {
            "record_id": self.record_id, "namespace": self.namespace, "kind": self.kind,
            "available_at": self.available_at, "event_span": list(self.event_span),
            "arrays": {k: array_hash(v) for k, v in sorted(self.arrays.items())}, "data": self.data,
        }


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    event_id: str
    op: str  # append | retract
    record: MemoryRecord | None = None
    target_id: str | None = None


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    snapshot_id: str
    parent_id: str | None
    head: str
    record_ids: tuple[str, ...]
    n_events: int

    def canonical(self) -> dict:
        return {"snapshot_id": self.snapshot_id, "parent_id": self.parent_id, "head": self.head, "record_ids": list(self.record_ids), "n_events": self.n_events}


@dataclass(frozen=True, slots=True)
class MemoryView:
    """Pinned, immutable, causally filtered view of one snapshot."""

    view_id: str
    snapshot_id: str
    cutoff: AvailabilityCut
    namespaces: frozenset[str]
    records: tuple[MemoryRecord, ...]

    def canonical(self) -> dict:
        return {"view_id": self.view_id, "snapshot_id": self.snapshot_id, "cutoff": self.cutoff.canonical(), "namespaces": sorted(self.namespaces)}


@dataclass
class BranchView:
    branch_id: str
    base_snapshot_id: str
    head: str
    _records: dict[str, MemoryRecord]
    _applied_events: list[str]

    def record_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))


@dataclass(frozen=True, slots=True)
class CommitReceipt:
    branch_id: str
    new_head: str
    applied: tuple[str, ...]
    skipped_duplicates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QueryPlan:
    kind: str                     # analogue | list
    namespace: str
    args: dict[str, Any] = field(default_factory=dict)
    required_completeness: str = "any"

    def canonical(self) -> dict:
        a = {k: (array_hash(v) if isinstance(v, np.ndarray) else v) for k, v in sorted(self.args.items())}
        return {"kind": self.kind, "namespace": self.namespace, "args": a, "required_completeness": self.required_completeness}

    @property
    def plan_hash(self) -> str:
        return content_hash(self.canonical())


@dataclass(frozen=True, slots=True)
class QueryResult:
    returned: tuple[MemoryRecord, ...]
    candidate_count: int
    completeness: Completeness
    truncated: bool
    view_id: str
    plan_hash: str
    scores: tuple[float, ...] = ()
    status: str = "ok"


@runtime_checkable
class MemoryStore(Protocol):
    def describe(self) -> MemoryCapabilities: ...
    def open_view(self, snapshot_id: str, cutoff: AvailabilityCut, namespaces: frozenset[str] | None = None) -> MemoryView: ...
    def query(self, view: MemoryView, plan: QueryPlan, budget: QueryBudget) -> QueryResult: ...
    def fork(self, snapshot_id: str, branch_id: str) -> BranchView: ...
    def append(self, branch: BranchView, events: tuple[MemoryEvent, ...], expected_head: str) -> CommitReceipt: ...
    def seal(self, branch: BranchView) -> SnapshotManifest: ...


EMPTY_SNAPSHOT_ID = "empty"


class InMemorySnapshotStore:
    """Reference backend. Deterministic, dependency-free, fully isolated forks."""

    store_id = "in_memory_snapshot_store"
    plugin_version = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self._snapshots: dict[str, tuple[SnapshotManifest, tuple[MemoryRecord, ...]]] = {}
        self._branches: dict[str, BranchView] = {}
        empty = SnapshotManifest(EMPTY_SNAPSHOT_ID, None, "genesis", (), 0)
        self._snapshots[EMPTY_SNAPSHOT_ID] = (empty, ())

    # --- capabilities --------------------------------------------------------
    def describe(self) -> MemoryCapabilities:
        return MemoryCapabilities(self.store_id, "snapshot_copy", True, "in_memory", ("analogue", "list"), True)

    def snapshot_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._snapshots))

    def manifest(self, snapshot_id: str) -> SnapshotManifest:
        return self._snapshots[snapshot_id][0]

    # --- views ---------------------------------------------------------------
    def open_view(self, snapshot_id: str, cutoff: AvailabilityCut, namespaces: frozenset[str] | None = None) -> MemoryView:
        if snapshot_id not in self._snapshots:
            raise ContractViolation(f"unknown snapshot {snapshot_id!r}")
        _, records = self._snapshots[snapshot_id]
        ns = namespaces
        visible = tuple(
            r for r in records
            if cutoff.permits(r.available_at) and (ns is None or r.namespace in ns)
        )
        view_id = content_hash({"snapshot": snapshot_id, "cutoff": cutoff.canonical(), "ns": sorted(ns) if ns else None})[:16]
        return MemoryView(view_id, snapshot_id, cutoff, frozenset(ns) if ns else frozenset(r.namespace for r in visible), visible)

    # --- queries -------------------------------------------------------------
    def query(self, view: MemoryView, plan: QueryPlan, budget: QueryBudget) -> QueryResult:
        # Eligibility filtering happens on the pinned view BEFORE ranking, never after.
        candidates = [r for r in view.records if r.namespace == plan.namespace]
        truncated = False
        if len(candidates) > budget.max_candidates:
            candidates = candidates[: budget.max_candidates]
            truncated = True
        if plan.kind == "list":
            out = tuple(candidates[: budget.max_results])
            return QueryResult(out, len(candidates), Completeness.COMPLETE if not truncated and len(out) == len(candidates) else Completeness.PARTIAL,
                               truncated or len(out) < len(candidates), view.view_id, plan.plan_hash)
        if plan.kind == "analogue":
            return self._analogue(view, plan, candidates, budget, truncated)
        raise ContractViolation(f"unsupported query kind {plan.kind!r}")

    @staticmethod
    def _analogue(view: MemoryView, plan: QueryPlan, candidates: list[MemoryRecord], budget: QueryBudget, truncated: bool) -> QueryResult:
        q = np.asarray(plan.args["query_window"], dtype=np.float64)
        k = int(plan.args.get("k", 1))
        key = plan.args.get("window_key", "window")
        scored: list[tuple[float, MemoryRecord]] = []
        for r in candidates:
            w = r.arrays.get(key)
            if w is None or w.shape != q.shape:
                continue
            qs = (q - q.mean(axis=0)) / (q.std(axis=0) + 1e-8)
            ws = (w - w.mean(axis=0)) / (w.std(axis=0) + 1e-8)
            scored.append((float(np.mean((qs - ws) ** 2)), r))
        scored.sort(key=lambda t: (t[0], t[1].record_id))
        top = scored[: min(k, budget.max_results)]
        return QueryResult(
            returned=tuple(r for _, r in top),
            candidate_count=len(scored),
            completeness=Completeness.PARTIAL if truncated else Completeness.COMPLETE,
            truncated=truncated,
            view_id=view.view_id,
            plan_hash=plan.plan_hash,
            scores=tuple(s for s, _ in top),
        )

    # --- branches ------------------------------------------------------------
    def fork(self, snapshot_id: str, branch_id: str) -> BranchView:
        if snapshot_id not in self._snapshots:
            raise ContractViolation(f"unknown snapshot {snapshot_id!r}")
        if branch_id in self._branches:
            raise ContractViolation(f"branch {branch_id!r} already exists")
        manifest, records = self._snapshots[snapshot_id]
        branch = BranchView(branch_id, snapshot_id, manifest.head, {r.record_id: r for r in records}, [])
        self._branches[branch_id] = branch
        return branch

    def append(self, branch: BranchView, events: tuple[MemoryEvent, ...], expected_head: str) -> CommitReceipt:
        if branch.head != expected_head:
            raise MemoryConflict(f"branch {branch.branch_id}: expected head {expected_head!r} but head is {branch.head!r}")
        applied, skipped = [], []
        for ev in events:
            if ev.event_id in branch._applied_events:
                skipped.append(ev.event_id)
                continue
            if ev.op == "append":
                assert ev.record is not None
                if ev.record.record_id in branch._records:
                    skipped.append(ev.event_id)
                    continue
                branch._records[ev.record.record_id] = ev.record
            elif ev.op == "retract":
                branch._records.pop(ev.target_id or "", None)
            else:
                raise ContractViolation(f"unknown memory event op {ev.op!r}")
            branch._applied_events.append(ev.event_id)
            applied.append(ev.event_id)
        if applied:
            branch.head = content_hash({"prev": branch.head, "events": applied})[:16]
        return CommitReceipt(branch.branch_id, branch.head, tuple(applied), tuple(skipped))

    def seal(self, branch: BranchView) -> SnapshotManifest:
        records = tuple(sorted(branch._records.values(), key=lambda r: r.record_id))
        snapshot_id = content_hash({"records": [r.canonical() for r in records]})[:24]
        manifest = SnapshotManifest(snapshot_id, branch.base_snapshot_id, branch.head, tuple(r.record_id for r in records), len(branch._applied_events))
        self._snapshots.setdefault(snapshot_id, (manifest, records))
        return manifest

    # --- canonical export/import (for whole-system checkpoints) -------------
    def export_snapshot(self, snapshot_id: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        manifest, records = self._snapshots[snapshot_id]
        arrays: dict[str, np.ndarray] = {}
        data: dict[str, Any] = {"manifest": manifest.canonical(), "records": []}
        for r in records:
            for k, v in r.arrays.items():
                arrays[f"{r.record_id}/{k}"] = np.asarray(v)
            data["records"].append({"record_id": r.record_id, "namespace": r.namespace, "kind": r.kind,
                                    "available_at": r.available_at, "event_span": list(r.event_span),
                                    "array_keys": sorted(r.arrays), "data": r.data})
        return arrays, data

    def import_snapshot(self, arrays: dict[str, np.ndarray], data: dict[str, Any]) -> SnapshotManifest:
        records = []
        for rd in data["records"]:
            arrs = {k: arrays[f"{rd['record_id']}/{k}"] for k in rd["array_keys"]}
            records.append(MemoryRecord(rd["record_id"], rd["namespace"], rd["kind"], rd["available_at"], tuple(rd["event_span"]), arrs, rd["data"]))
        records_t = tuple(sorted(records, key=lambda r: r.record_id))
        snapshot_id = content_hash({"records": [r.canonical() for r in records_t]})[:24]
        m = data["manifest"]
        if m["snapshot_id"] != snapshot_id:
            raise ContractViolation("imported snapshot content hash does not match its manifest")
        manifest = SnapshotManifest(snapshot_id, m["parent_id"], m["head"], tuple(m["record_ids"]), m["n_events"])
        self._snapshots[snapshot_id] = (manifest, records_t)
        return manifest
