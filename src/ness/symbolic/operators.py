"""Domain-neutral primitive registry for the typed IR.

Each primitive declares typed inputs/outputs, semantic version, purity, availability
propagation and a deterministic cost. The interpreter resolves primitives by signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..contracts import ContractViolation, Knownness


@dataclass(frozen=True, slots=True)
class PrimitiveResult:
    value: Any
    knownness: Knownness = Knownness.KNOWN
    detail: str = ""


@dataclass(frozen=True, slots=True)
class PrimitiveSpec:
    name: str
    input_types: dict[str, str]
    output_type: str
    semantic_version: str
    purity: str  # pure | fit_only
    availability_propagation: str  # "inputs" -> derived availability is max of inputs
    differentiable: bool
    cost: int
    fn: Callable[..., PrimitiveResult]
    description: str = ""


class PrimitiveRegistry:
    def __init__(self) -> None:
        self._items: dict[str, PrimitiveSpec] = {}

    def register(self, spec: PrimitiveSpec) -> None:
        if spec.name in self._items:
            raise ContractViolation(f"primitive {spec.name!r} already registered")
        self._items[spec.name] = spec

    def get(self, name: str) -> PrimitiveSpec:
        if name not in self._items:
            raise ContractViolation(f"unknown primitive {name!r}; unknown primitives fail closed")
        return self._items[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))


def _window_slope(series: np.ndarray, window: int) -> PrimitiveResult:
    """OLS slope over the trailing ``window`` points, per channel. Units: value/step."""
    s = np.asarray(series, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, None]
    if not isinstance(window, (int, np.integer)) or window < 2:
        raise ContractViolation("window_slope requires integer window >= 2")
    if s.shape[0] < window:
        return PrimitiveResult(np.zeros(s.shape[1]), Knownness.MISSING, "insufficient history")
    x = np.arange(window, dtype=np.float64)
    x = x - x.mean()
    y = s[-window:]
    slope = (x[:, None] * (y - y.mean(axis=0))).sum(axis=0) / (x**2).sum()
    return PrimitiveResult(slope, Knownness.KNOWN)


def _window_mean(series: np.ndarray, window: int) -> PrimitiveResult:
    s = np.asarray(series, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, None]
    if s.shape[0] < window or window < 1:
        return PrimitiveResult(np.zeros(s.shape[1]), Knownness.MISSING, "insufficient history")
    return PrimitiveResult(s[-window:].mean(axis=0), Knownness.KNOWN)


def _last_value(series: np.ndarray) -> PrimitiveResult:
    s = np.asarray(series, dtype=np.float64)
    if s.ndim == 1:
        s = s[:, None]
    if s.shape[0] == 0:
        return PrimitiveResult(np.zeros(s.shape[1]), Knownness.MISSING, "empty series")
    return PrimitiveResult(s[-1], Knownness.KNOWN)


def _event_active_at_origin(events: np.ndarray, horizon: int) -> PrimitiveResult:
    """1.0 if any scheduled event is active within the first ``horizon`` delivered steps
    (the known-future window at the forecast origin), else 0.0. Knownness requires the
    window to be fully available: a shorter delivery is MISSING, never a silent 0."""
    e = np.asarray(events, dtype=np.float64).reshape(-1)
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ContractViolation("event_active_at_origin requires integer horizon >= 1")
    if e.shape[0] < horizon:
        return PrimitiveResult(np.zeros(1), Knownness.MISSING, "event window shorter than horizon")
    return PrimitiveResult(np.array([1.0 if np.any(e[:horizon] > 0.5) else 0.0]), Knownness.KNOWN)


def default_registry() -> PrimitiveRegistry:
    reg = PrimitiveRegistry()
    reg.register(PrimitiveSpec("window_slope", {"series": "numeric_series", "window": "int"}, "numeric_vector", "1", "pure", "inputs", False, 4, _window_slope, "trailing OLS slope per channel (value/step)"))
    reg.register(PrimitiveSpec("window_mean", {"series": "numeric_series", "window": "int"}, "numeric_vector", "1", "pure", "inputs", False, 2, _window_mean, "trailing mean per channel"))
    reg.register(PrimitiveSpec("last_value", {"series": "numeric_series"}, "numeric_vector", "1", "pure", "inputs", False, 1, _last_value, "last observed value per channel"))
    reg.register(PrimitiveSpec("event_active_at_origin", {"events": "numeric_series", "horizon": "int"}, "numeric_vector", "1", "pure", "inputs", False, 1, _event_active_at_origin,
                               "bounded event predicate over the known-future window at the origin"))
    return reg
