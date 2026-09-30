"""Forecasts expose *capabilities*, not a fictitious universal probability API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .coordinates import CoordinateSchema
from .errors import ContractViolation, UnsupportedCapability
from .hashing import array_hash


@dataclass(frozen=True, slots=True)
class ForecastCapabilities:
    point: bool = False
    quantiles: bool = False
    log_prob: bool = False
    cdf: bool = False
    sample: bool = False
    joint_dependence: bool = False

    def canonical(self) -> dict:
        return {k: getattr(self, k) for k in ("point", "quantiles", "log_prob", "cdf", "sample", "joint_dependence")}


class Forecast:
    """Base type. Subclasses implement only the operations they truly support; every
    other operation raises ``UnsupportedCapability``."""

    forecast_type: str = "abstract"
    target: CoordinateSchema
    capabilities: ForecastCapabilities

    def point(self) -> np.ndarray:
        raise UnsupportedCapability(f"{self.forecast_type} forecast does not support point()")

    def quantiles(self, levels: tuple[float, ...]) -> np.ndarray:
        raise UnsupportedCapability(f"{self.forecast_type} forecast does not support quantiles()")

    def log_prob(self, y: np.ndarray) -> np.ndarray:
        raise UnsupportedCapability(f"{self.forecast_type} forecast does not support log_prob()")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise UnsupportedCapability(f"{self.forecast_type} forecast does not support sample()")

    def dense(self) -> np.ndarray:
        """Lowering used by the numerical packer. Documented per subclass."""
        raise UnsupportedCapability(f"{self.forecast_type} has no dense lowering")

    def canonical(self) -> dict:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PointForecast(Forecast):
    """Estimate plus functional (mean/median/...). Not a Dirac distribution."""

    values: np.ndarray  # axes named by target (e.g. horizon, channel)
    target: CoordinateSchema
    functional: str = "mean"
    capabilities: ForecastCapabilities = field(default_factory=lambda: ForecastCapabilities(point=True))
    forecast_type: str = "point"

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=np.float64)
        if v.ndim != len(self.target.axes):
            raise ContractViolation(f"point forecast rank {v.ndim} != target axes {self.target.axes}")
        if not np.all(np.isfinite(v)):
            raise ContractViolation("point forecast contains non-finite values")
        object.__setattr__(self, "values", v)
        v.setflags(write=False)

    def point(self) -> np.ndarray:
        return self.values

    def dense(self) -> np.ndarray:
        return self.values

    def canonical(self) -> dict:
        return {"type": "point", "functional": self.functional, "target": self.target.schema_id, "values": array_hash(self.values)}


@dataclass(frozen=True, slots=True)
class QuantileForecast(Forecast):
    """Values at declared increasing levels. Not a density; no tails implied."""

    values: np.ndarray  # [..., quantile] with quantile the LAST axis
    target: CoordinateSchema
    levels: tuple[float, ...]
    capabilities: ForecastCapabilities = field(default_factory=lambda: ForecastCapabilities(quantiles=True, point=True))
    forecast_type: str = "quantile"
    point_functional: str = "median"

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=np.float64)
        if v.ndim != len(self.target.axes) or self.target.axes[-1] != "quantile":
            raise ContractViolation("quantile forecast must have 'quantile' as last named axis")
        if v.shape[-1] != len(self.levels):
            raise ContractViolation(f"quantile axis {v.shape[-1]} != levels {len(self.levels)}")
        if list(self.levels) != sorted(self.levels) or len(set(self.levels)) != len(self.levels):
            raise ContractViolation("quantile levels must be strictly increasing")
        if self.target.quantile_levels is not None and tuple(self.target.quantile_levels) != tuple(self.levels):
            raise ContractViolation(f"quantile levels {self.levels} differ from the target schema grid {self.target.quantile_levels}")
        if not np.all(np.isfinite(v)):
            raise ContractViolation("quantile forecast contains non-finite values")
        object.__setattr__(self, "values", v)
        v.setflags(write=False)

    def quantiles(self, levels: tuple[float, ...]) -> np.ndarray:
        idx = []
        for q in levels:
            if q not in self.levels:
                raise UnsupportedCapability(f"level {q} not in declared grid {self.levels}; no interpolation is implied")
            idx.append(self.levels.index(q))
        return self.values[..., idx]

    def point(self) -> np.ndarray:
        if 0.5 not in self.levels:
            raise UnsupportedCapability("no median level declared; point() unavailable")
        return self.values[..., self.levels.index(0.5)]

    def is_monotone(self) -> bool:
        return bool(np.all(np.diff(self.values, axis=-1) >= 0))

    def dense(self) -> np.ndarray:
        return self.values

    def canonical(self) -> dict:
        return {"type": "quantile", "levels": list(self.levels), "target": self.target.schema_id, "values": array_hash(self.values)}


@dataclass(frozen=True, slots=True)
class CategoricalForecast(Forecast):
    """Normalised mass over a versioned vocabulary."""

    log_probs: np.ndarray
    target: CoordinateSchema
    vocabulary_id: str
    capabilities: ForecastCapabilities = field(default_factory=lambda: ForecastCapabilities(log_prob=True, sample=True))
    forecast_type: str = "categorical"

    def __post_init__(self) -> None:
        raw = np.asarray(self.log_probs)
        lp = np.asarray(raw, dtype=np.float64)
        z = np.log(np.sum(np.exp(lp), axis=-1))
        # tolerance follows the producer's precision: float32 log-softmax over many classes carries
        # rounding far above 1e-6; float64 keeps the v0.2 bound
        eps = np.finfo(raw.dtype).eps if np.issubdtype(raw.dtype, np.floating) else np.finfo(np.float64).eps
        atol = max(1e-6, 32.0 * float(eps) * float(np.sqrt(lp.shape[-1])))
        if not np.allclose(z, 0.0, atol=atol):
            raise ContractViolation(f"categorical log_probs must be normalised along the last axis (max |logsumexp| {float(np.max(np.abs(z))):.3g} > {atol:.3g})")
        object.__setattr__(self, "log_probs", lp)
        lp.setflags(write=False)

    def log_prob(self, y: np.ndarray) -> np.ndarray:
        return np.take_along_axis(self.log_probs, np.asarray(y)[..., None], axis=-1)[..., 0]

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` samples of every row: shape ``[n, *leading axes]`` (bug B-1: v0.2 sampled the first row only)."""
        p = np.exp(self.log_probs)
        rows = p.reshape(-1, p.shape[-1])
        lead = p.shape[:-1]
        return np.stack([np.array([rng.choice(p.shape[-1], p=r / r.sum()) for r in rows]).reshape(lead) for _ in range(n)])

    def dense(self) -> np.ndarray:
        return self.log_probs

    def canonical(self) -> dict:
        return {"type": "categorical", "vocabulary_id": self.vocabulary_id, "target": self.target.schema_id, "values": array_hash(self.log_probs)}


@dataclass(frozen=True, slots=True)
class SampleForecast(Forecast):
    """Weighted samples with declared joint/marginal meaning."""

    samples: np.ndarray  # [n, ...target axes]
    weights: np.ndarray  # [n]
    target: CoordinateSchema
    joint_semantics: str = "marginal_per_coordinate"
    capabilities: ForecastCapabilities = field(default_factory=lambda: ForecastCapabilities(point=True, sample=True))
    forecast_type: str = "sample"

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=np.float64)
        if not np.isclose(w.sum(), 1.0):
            raise ContractViolation("sample weights must sum to 1")
        object.__setattr__(self, "weights", w)

    def point(self) -> np.ndarray:
        return np.tensordot(self.weights, np.asarray(self.samples, dtype=np.float64), axes=(0, 0))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.choice(len(self.weights), size=n, p=self.weights)
        return np.asarray(self.samples)[idx]

    def canonical(self) -> dict:
        return {"type": "sample", "target": self.target.schema_id, "samples": array_hash(np.asarray(self.samples)), "weights": array_hash(self.weights)}


@dataclass(frozen=True, slots=True)
class PredictionSpace:
    """What a task head may output and how it is validated, corrected and scored."""

    space_id: str
    forecast_type: str
    target: CoordinateSchema
    scoring_rule: str
    allowed_correction_operators: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, forecast: Forecast) -> None:
        if forecast.forecast_type != self.forecast_type:
            raise ContractViolation(
                f"prediction space {self.space_id} expects {self.forecast_type} forecasts, got {forecast.forecast_type}"
            )
        forecast.target.require_compatible(self.target, where=f"in prediction space {self.space_id}")
        if self.forecast_type == "quantile":
            if tuple(forecast.levels) != tuple(self.target.quantile_levels or ()):
                raise ContractViolation("quantile forecast levels differ from the prediction space grid")

    def canonical(self) -> dict:
        return {
            "space_id": self.space_id,
            "forecast_type": self.forecast_type,
            "target": self.target.schema_id,
            "scoring_rule": self.scoring_rule,
            "allowed_correction_operators": list(self.allowed_correction_operators),
        }
