"""The prediction transaction (PDF §12.1):

    pin manifest + memory view -> validate access -> retrieve/execute -> pack -> target-free
    prediction -> immutable record -> only then accept the outcome -> idempotent experience
    event -> schedule learning on learner copies.

Three evaluation modes are distinct: ``frozen`` (no updates of any kind), ``prequential``
(predict and score before permitted post-outcome updates), and ``adaptive`` (not
implemented; explicit error).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..contracts import ContractViolation, PredictionRequest, ScoreResult, TrainingOutcome, UnsupportedCapability
from ..memory import MemoryRecord
from ..observability import ExperienceEvent, PredictionRecord
from .executor import ExecutionRecord
from .system import NessSystem

MODES = ("frozen", "prequential")
RETENTION = ("full", "outputs")  # what a transaction keeps per request after the outcome is revealed


@dataclass
class PendingItem:
    record: PredictionRecord
    execution: ExecutionRecord
    request: PredictionRequest
    outcomes: dict[str, TrainingOutcome] = field(default_factory=dict)  # task_id -> outcome
    auxiliary: dict[str, np.ndarray] = field(default_factory=dict)        # revealed auxiliary targets (learning only)


class PredictionTransaction:
    def __init__(self, system: NessSystem, mode: str = "frozen", retention: str = "full") -> None:
        """``retention``: ``full`` (default, v0.2) keeps every execution record until learning
        consumes it (in ``frozen`` mode: forever); ``outputs`` drops a frozen prediction's execution
        record (all port values) as soon as its outcome is revealed, keeping the immutable
        prediction record and scores, so frozen evaluation memory stays flat in the stream length."""
        if mode not in MODES:
            raise UnsupportedCapability(f"evaluation mode {mode!r} is not implemented; modes {MODES}")
        if retention not in RETENTION:
            raise UnsupportedCapability(f"retention {retention!r} is not implemented; policies {RETENTION}")
        self.system = system
        self.mode = mode
        self.retention = retention
        self.records: dict[str, PredictionRecord] = {}
        self.events: dict[str, ExperienceEvent] = {}
        self._pending: dict[str, PendingItem] = {}
        self._batch: list[Any] = []
        self._memory_pending: list[MemoryRecord] = []

    # ------------------------------------------------------------------ predict
    def predict(self, request: PredictionRequest) -> PredictionRecord:
        if request.request_id in self.records:
            raise ContractViolation(f"request {request.request_id} already has an immutable prediction record")
        result, execution, ctx = self.system.predict(request, mode="train" if self.mode == "prequential" else "predict")
        rec = PredictionRecord.from_result(result, request.origin, {k: v.view_id for k, v in ctx.memory_views.items()}, request.group)
        self.records[request.request_id] = rec
        self._pending[request.request_id] = PendingItem(rec, execution, request)
        return rec

    def execution_of(self, request_id: str):
        """The execution record of a pending (not yet learned-from) prediction, or None."""
        item = self._pending.get(request_id)
        return item.execution if item is not None else None

    def scores_of(self, request_id: str) -> dict[str, Any]:
        for ev in self.events.values():
            if ev.request_id == request_id:
                return ev.scores
        return {}

    # ------------------------------------------------------------------ reveal
    def reveal(self, request_id: str, outcomes: dict[str, TrainingOutcome]) -> ExperienceEvent:
        """Accept outcomes for an already-recorded prediction. Idempotent per release
        sequence; a duplicate release returns the stored event without re-scoring."""
        if request_id not in self.records:
            raise ContractViolation(f"outcome for {request_id} arrived before any prediction was recorded")
        rec = self.records[request_id]
        release = max((o.release_sequence for o in outcomes.values()), default=0)  # outcome-free items (e.g. a preservation stream) are allowed
        eid = ExperienceEvent.make_id(request_id, release)
        if eid in self.events:
            return self.events[eid]
        scores: dict[str, ScoreResult] = {}
        for task_id, oc in outcomes.items():
            query_id = next(q for q in rec.outputs if q.endswith(f"/{task_id}"))
            scores[task_id] = self.system.tasks[task_id].score(rec.outputs[query_id], oc)
        ev = ExperienceEvent(eid, rec.record_id, request_id, dict(outcomes), scores)
        self.events[eid] = ev
        item = self._pending.get(request_id)
        if item is not None and self.mode == "prequential":
            item.outcomes.update(outcomes)
            # auxiliary targets: outcome_only fields that declared objectives read. They were never in any
            # permitted view; they reach learning only here, after reveal (ADR-0012).
            for name in self.system.auxiliary_fields():
                if item.request.bundle.has(name):
                    item.auxiliary[name] = np.array(item.request.bundle.field(name).array, copy=True)
            self._batch.append(item)
            if self.system.memory is not None and self.system.memory.learner_appends and item.outcomes:
                self._memory_pending.append(self._episode_record(item))
        elif item is not None and self.retention == "outputs":
            self._pending.pop(request_id, None)
        return ev

    def _episode_record(self, item: PendingItem) -> MemoryRecord:
        m = self.system.memory
        assert m is not None
        window = item.request.bundle.field(m.episode_window_field or "target_history").array
        oc = next(iter(item.outcomes.values()))
        return MemoryRecord(f"episode@{item.request.origin}", m.namespace, "episode", oc.available_at,
                            (item.request.origin - window.shape[0], item.request.origin + oc.values.shape[0]),
                            {"window": np.array(window), "continuation": np.array(oc.values)}, {"origin": item.request.origin, "source": "prequential"})

    # ------------------------------------------------------------------ learn
    def pending_batch_size(self) -> int:
        return len(self._batch)

    def learn_step(self) -> dict[str, Any] | None:
        """One complete optimizer update on the pending batch, then publish pending memory.
        Returns None when there is nothing trainable (a valid control arm)."""
        if self.mode != "prequential":
            raise UnsupportedCapability("learning is only permitted in prequential mode; frozen test performs no updates")
        if not self._batch:
            raise ContractViolation("learn_step called with an empty batch; the protocol declares this a hard error")
        diag = None
        if self.system.trainable:
            from ..learning import BatchItem
            batch = [BatchItem(it.execution, it.outcomes, it.request, dict(it.auxiliary)) for it in self._batch]
            diag = self.system.learn(batch)
            for it in self._batch:
                ev_ids = [e for e, ev in self.events.items() if ev.request_id == it.request.request_id]
                for e in ev_ids:
                    self.events[e] = ExperienceEvent(self.events[e].event_id, self.events[e].record_id, self.events[e].request_id,
                                                     self.events[e].outcomes, self.events[e].scores, self.system.n_updates)
        for it in self._batch:
            self._pending.pop(it.request.request_id, None)
        self._batch.clear()
        if self._memory_pending:
            self.system.memory_learner_append(tuple(self._memory_pending))
            self._memory_pending.clear()
            self.system.memory_publish()
        return diag

    # ------------------------------------------------------------------ reporting
    def aggregate_scores(self) -> dict[str, dict[str, float]]:
        """Sum numerators and denominators once per task (never mean of means)."""
        agg: dict[str, dict[str, float]] = {}
        for ev in self.events.values():
            for task_id, sc in ev.scores.items():
                a = agg.setdefault(task_id, {"numerator": 0.0, "denominator": 0.0, "n_requests": 0.0})
                a["numerator"] += sc.numerator
                a["denominator"] += sc.denominator
                a["n_requests"] += 1
        for a in agg.values():
            a["loss"] = a["numerator"] / a["denominator"] if a["denominator"] > 0 else float("nan")
        return agg
