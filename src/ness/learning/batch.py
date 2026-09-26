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
