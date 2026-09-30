"""Backend-neutral learning batch and context types (numpy only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..contracts import ModuleState, TrainingOutcome


@dataclass
class BatchItem:
    """One recorded prediction with its revealed outcomes (task_id -> outcome)."""

    record: Any  # runtime.executor.ExecutionRecord
    outcomes: dict[str, TrainingOutcome]
    request: Any = None                                              # the PredictionRequest (applicability filters, training-only nodes)
    auxiliary: dict[str, np.ndarray] = field(default_factory=dict)   # revealed auxiliary targets (outcome_only fields), learning only
    active_objectives: frozenset[str] = frozenset()                  # objective ids that apply to this item (set by the system)
    objective_inputs: dict[str, tuple[np.ndarray, np.ndarray | None]] = field(default_factory=dict)  # port_target id -> (target, mask)


@dataclass
class LearningContext:
    """Everything a learning rule may read: compiled graph, current states, edge parameters,
    tasks, and the runtime report. Rules never receive the system object itself."""

    compiled: Any
    node_states: dict[str, ModuleState]
    edge_params: dict[str, dict[str, np.ndarray]]
    tasks: dict[str, Any]
    task_weights: dict[str, float] = field(default_factory=dict)
    runtime_report: Any = None
    caches: dict[str, Any] = field(default_factory=dict)  # per-rule memo (e.g. a DifferentiableRegion)
    objectives: Any = None  # learning.objectives.ObjectivePlan when the arm declares objectives (None: v0.2 behaviour)
