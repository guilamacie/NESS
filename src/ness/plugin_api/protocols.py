"""Family protocols. All execution goes through ``MacroModule.forward``; these protocols
add the family-specific *declarations* (capabilities, prediction spaces, scoring) that
the compiler, tasks and learning rules rely on.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable

import numpy as np

from ..contracts import (
    AccessPolicy,
    FieldRole,
    ModuleDescriptor,
    ObservationBundle,
    ObservationField,
    PredictionRequest,
    PredictionSpace,
    ScenarioCapabilities,
    ScoreResult,
    TaskQuery,
    TrainingOutcome,
    Forecast,
)
from .module import MacroModule


@runtime_checkable
class Substrate(MacroModule, Protocol):
    """Lower provider of declared representations and optional native forecasts.
    ``describe().capabilities`` is a ``SubstrateCapabilities`` record."""


@runtime_checkable
class UpperModule(MacroModule, Protocol):
    """Trainable numerical module above substrates (transformer, MLP, GNN, SSM, ...).
    Also implements ``DifferentiableModule.apply`` when trainable."""


@runtime_checkable
class SemanticAdapter(MacroModule, Protocol):
    """Interprets representations/observations as typed features, regimes, hypotheses."""


@runtime_checkable
class Writer(Protocol):
    """Baseline-preserving output writer: maps (baseline, correction) -> forecast values.
    Pure functions over an array namespace so they run in the differentiable runtime."""

    writer_id: str
    writer_version: str
    forecast_type: str  # point | quantile

    def correction_dim(self, baseline_shape: tuple[int, ...]) -> int: ...

    def write(self, baseline: Any, correction: Any, xp: Any) -> Any: ...

    def is_baseline_preserving_at_zero(self) -> bool: ...


@runtime_checkable
class Consumer(Protocol):
    """Learned map from packed evidence features to a correction vector."""

    consumer_id: str
    consumer_version: str

    def param_shapes(self, in_dim: int, out_dim: int) -> dict[str, tuple[int, ...]]: ...

    def init_params(self, in_dim: int, out_dim: int, rng: np.random.Generator) -> dict[str, np.ndarray]: ...

    def apply(self, params: dict[str, Any], x: Any, xp: Any) -> Any: ...


@runtime_checkable
class PredictionTask(Protocol):
    """Owns the information boundary, prediction space, loss and score for one head."""

    task_id: str

    def describe(self) -> ModuleDescriptor: ...

    def allowed_roles(self) -> frozenset[FieldRole]: ...

    def permitted_view(self, bundle: ObservationBundle, query: TaskQuery) -> AccessPolicy: ...

    def prediction_space(self, query: TaskQuery | None = None) -> PredictionSpace: ...

    def make_query(self, request_id: str) -> TaskQuery: ...

    def score(self, forecast: Forecast, outcome: TrainingOutcome) -> ScoreResult: ...

    def loss_terms(self, pred_dense: Any, y: Any, mask: Any, xp: Any) -> tuple[Any, Any]:
        """(numerator, denominator) of the task loss, traceable in ``xp``."""
        ...


@runtime_checkable
class ScenarioProvider(Protocol):
    """Materialises ObservationBundle/PredictionRequest/TrainingOutcome. Never bypasses
    ``PredictionTask.permitted_view``: it supplies raw role-tagged records only."""

    def describe(self) -> ModuleDescriptor: ...

    def capabilities(self) -> ScenarioCapabilities: ...

    def observation_fields(self) -> dict[str, ObservationField]: ...

    def splits(self) -> tuple[str, ...]: ...

    def iter_requests(self, split: str, tasks: tuple[PredictionTask, ...], protocol: dict[str, Any] | None = None) -> Iterable[PredictionRequest]: ...

    def outcome(self, request_id: str, query_id: str) -> TrainingOutcome | None: ...

    def grouping_metadata(self, request_id: str) -> dict[str, Any]: ...


@runtime_checkable
class LearningRule(Protocol):
    """Parameter-learning rule, independent of the inference profile."""

    rule_id: str
    rule_version: str

    def describe(self) -> ModuleDescriptor: ...

    def required_runtime(self) -> str | None: ...
