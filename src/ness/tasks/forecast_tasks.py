"""Point and quantile forecast tasks: information boundary, prediction space, loss, score."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import (
    AccessPolicy,
    ContractViolation,
    FieldRole,
    Forecast,
    ModuleDescriptor,
    ObservationBundle,
    PredictionSpace,
    ScoreResult,
    TaskQuery,
    TrainingOutcome,
    horizon_schema,
    quantile_schema,
)
from ..plugin_api.base import BaseModule
from .access import AS_OF_ORIGIN_ROLES, as_of_origin_policy


class _ForecastTaskBase(BaseModule):
    module_kind = "task"
    runtime = "host"
    state_schema_id = "ness.task/1"
    forecast_type = "abstract"

    def validate_config(self) -> None:
        self.task_id: str = self.config.get("task_id", self.plugin_id)
        self.channels: tuple[str, ...] = tuple(self.config.get("channels", ("y0", "y1")))
        h = self.config.get("horizons", 4)
        self.horizons: tuple[int, ...] = tuple(range(1, int(h) + 1)) if isinstance(h, int) else tuple(int(x) for x in h)
        self.allowed_memory_namespaces = frozenset(self.config.get("memory_namespaces", ("episodes",)))
        if not self.channels or not self.horizons:
            raise ContractViolation(f"{self.plugin_id}: channels and horizons required")

    def describe(self) -> ModuleDescriptor:
        return self._descriptor(capabilities={"forecast_type": self.forecast_type, "scoring_rule": self.prediction_space().scoring_rule})

    def allowed_roles(self) -> frozenset[FieldRole]:
        return AS_OF_ORIGIN_ROLES

    def permitted_view(self, bundle: ObservationBundle, query: TaskQuery) -> AccessPolicy:
        return as_of_origin_policy(self.task_id, bundle.origin, None, self.allowed_memory_namespaces)

    def make_query(self, request_id: str) -> TaskQuery:
        return TaskQuery(f"{request_id}/{self.task_id}", self.task_id, self.prediction_space().target)

    def prediction_space(self, query: TaskQuery | None = None) -> PredictionSpace:  # pragma: no cover - overridden
        raise NotImplementedError


class PointForecastTask(_ForecastTaskBase):
    plugin_id = "point_forecast_task"
    plugin_version = "1.0.0"
    forecast_type = "point"

    def validate_config(self) -> None:
        super().validate_config()
        self.functional = self.config.get("functional", "mean")
        self.loss = self.config.get("loss", "mse")
        if self.loss not in ("mse", "mae"):
            raise ContractViolation("point loss must be mse|mae")

    def prediction_space(self, query: TaskQuery | None = None) -> PredictionSpace:
        return PredictionSpace(f"point:{self.task_id}", "point", horizon_schema(self.channels, self.horizons), self.loss, ("point_residual",))

    def loss_terms(self, pred: Any, y: Any, mask: Any, xp: Any) -> tuple[Any, Any]:
        err = y - pred
        per = err**2 if self.loss == "mse" else xp.abs(err)
        return xp.sum(mask * per), xp.sum(mask)

    def score(self, forecast: Forecast, outcome: TrainingOutcome) -> ScoreResult:
        self.prediction_space().validate(forecast)
        pred = forecast.point()
        if forecast.functional != self.functional:  # type: ignore[attr-defined]
            raise ContractViolation(f"point functional {forecast.functional!r} != task functional {self.functional!r}")  # type: ignore[attr-defined]
        m = outcome.mask.astype(np.float64)
        num, den = self.loss_terms(pred, outcome.values, m, np)
        mae = float(np.sum(m * np.abs(outcome.values - pred)) / max(m.sum(), 1e-12))
        return ScoreResult(float(num), float(den), {"mae": mae, "n_valid": float(m.sum())})


class QuantileForecastTask(_ForecastTaskBase):
    plugin_id = "quantile_forecast_task"
    plugin_version = "1.0.0"
    forecast_type = "quantile"

    def validate_config(self) -> None:
        super().validate_config()
        self.levels: tuple[float, ...] = tuple(float(q) for q in self.config.get("levels", (0.1, 0.5, 0.9)))
        lw = self.config.get("level_weights")
        self.level_weights = np.asarray(lw if lw is not None else [1.0] * len(self.levels), dtype=np.float64)
        if self.level_weights.shape != (len(self.levels),):
            raise ContractViolation("level_weights must match levels")

    def prediction_space(self, query: TaskQuery | None = None) -> PredictionSpace:
        return PredictionSpace(f"quantile:{self.task_id}", "quantile", quantile_schema(self.channels, self.horizons, self.levels), "pinball",
                               ("quantile_residual",), {"levels": list(self.levels)})

    def loss_terms(self, pred: Any, y: Any, mask: Any, xp: Any) -> tuple[Any, Any]:
        """Weighted pinball: sum_{h,d,q} a_hd v_q rho_q(y_hd - Q_hdq) / sum a_hd v_q."""
        taus = xp.asarray(np.asarray(self.levels))
        v = xp.asarray(self.level_weights)
        u = y[..., None] - pred
        rho = u * (taus - (u < 0).astype(u.dtype) if hasattr(u, "astype") else taus - xp.where(u < 0, 1.0, 0.0))
        w = mask[..., None] * v
        return xp.sum(w * rho), xp.sum(w)

    def score(self, forecast: Forecast, outcome: TrainingOutcome) -> ScoreResult:
        self.prediction_space().validate(forecast)
        q = forecast.values  # type: ignore[attr-defined]
        m = outcome.mask.astype(np.float64)
        num, den = self.loss_terms(q, outcome.values, m, np)
        lo, hi = q[..., 0], q[..., -1]
        inside = ((outcome.values >= lo) & (outcome.values <= hi)).astype(np.float64)
        cov = float(np.sum(m * inside) / max(m.sum(), 1e-12))
        width = float(np.sum(m * (hi - lo)) / max(m.sum(), 1e-12))
        return ScoreResult(float(num), float(den), {"coverage": cov, "width": width, "monotone": float(forecast.is_monotone()), "n_valid": float(m.sum())})  # type: ignore[attr-defined]
