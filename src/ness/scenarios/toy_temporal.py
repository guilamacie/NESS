"""``ToyTemporalDataset``: two observed channels, a calendar feature, two latent regimes,
fixed seed, rolling-origin requests and separately released outcomes.

The raw bundle carries role-tagged fields *including* ``target_future`` (outcome_only)
and ``true_regime`` (evaluation_oracle) so the access policy is exercised on every
request (T44). Only the task's permitted view ever reaches the predictor.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..contracts import (
    ContractViolation,
    CoordinateSchema,
    FieldRole,
    ModuleDescriptor,
    Observation,
    ObservationBundle,
    ObservationField,
    PredictionRequest,
    ScenarioCapabilities,
    TrainingOutcome,
    horizon_schema,
    series_schema,
)
from ..memory import MemoryRecord
from ..plugin_api.base import BaseModule
from .rolling_origin import chronological_split, rolling_origins


class ToyTemporalDataset(BaseModule):
    plugin_id = "toy_temporal_dataset"
    plugin_version = "1.0.0"
    module_kind = "scenario"
    runtime = "host"
    state_schema_id = "ness.scenario.toy_temporal/1"

    def validate_config(self) -> None:
        c = self.config
        self.n_steps = int(c.get("n_steps", 600))
        self.context = int(c.get("context", 32))
        self.horizon = int(c.get("horizon", 4))
        self.channels = tuple(c.get("channels", ("y0", "y1")))
        self.seed = int(c.get("seed", 1234))
        self.period = int(c.get("season_period", 7))
        self.switch_prob = float(c.get("regime_switch_prob", 0.03))
        self.train_fraction = float(c.get("train_fraction", 0.7))
        self.stride = int(c.get("stride", 1))
        if len(self.channels) != 2:
            raise ContractViolation("toy_temporal_dataset generates exactly two channels")
        self._generate()

    # ------------------------------------------------------------------ data
    def _generate(self) -> None:
        rng = np.random.default_rng(self.seed)
        n = self.n_steps
        z = np.zeros(n, dtype=int)
        for t in range(1, n):
            z[t] = 1 - z[t - 1] if rng.random() < self.switch_prob else z[t - 1]
        drift = np.where(z == 0, 0.05, -0.05)
        vol = np.where(z == 0, 0.15, 0.45)
        phase = 2 * np.pi * (np.arange(n) % self.period) / self.period
        season = 0.6 * np.sin(phase)
        y0 = np.zeros(n)
        y1 = np.zeros(n)
        level = 0.0
        for t in range(n):
            level = 0.9 * level + drift[t] + vol[t] * rng.normal()
            y0[t] = level + season[t]
            lag = y0[t - 2] if t >= 2 else 0.0
            y1[t] = 0.5 * lag + 0.3 * season[t] + 0.2 * vol[t] * rng.normal()
        self._y = np.stack([y0, y1], axis=1)
        self._z = z
        self._calendar = np.stack([np.sin(phase), np.cos(phase)], axis=1)
        origins = rolling_origins(n, self.context, self.horizon, self.stride)
        self._splits = chronological_split(origins, self.train_fraction, self.horizon)
        self.series_schema = series_schema(self.channels)
        self.calendar_schema = CoordinateSchema(f"calendar[sin,cos]@period{self.period}", ("time", "feature"), channel_names=("sin", "cos"))
        self.future_calendar_schema = CoordinateSchema(f"calendar_future[sin,cos]@period{self.period}", ("horizon", "feature"), channel_names=("sin", "cos"))
        self.target_schema = horizon_schema(self.channels, tuple(range(1, self.horizon + 1)))
        self.regime_schema = CoordinateSchema("regime[oracle]", ("time",))

    # ------------------------------------------------------------------ descriptors
    def describe(self) -> ModuleDescriptor:
        return self._descriptor(capabilities=self.capabilities())

    def capabilities(self) -> ScenarioCapabilities:
        return ScenarioCapabilities("synthetic_timeseries", tuple(self._splits), True, "coordinate_wise",
                                    tuple(self.observation_fields()), ("target_future",))

    def observation_fields(self) -> dict[str, ObservationField]:
        D = len(self.channels)
        return {
            "target_history": ObservationField("target_history", "timeseries", FieldRole.OBSERVED, "observation_series", self.series_schema, (None, D)),
            "calendar_history": ObservationField("calendar_history", "calendar", FieldRole.OBSERVED, "calendar_features", self.calendar_schema, (None, 2)),
            "calendar_future": ObservationField("calendar_future", "calendar", FieldRole.KNOWN_FUTURE, "calendar_features", self.future_calendar_schema, (self.horizon, 2)),
            "target_future": ObservationField("target_future", "timeseries", FieldRole.OUTCOME_ONLY, "outcome", self.target_schema, (self.horizon, D)),
            "true_regime": ObservationField("true_regime", "oracle", FieldRole.EVALUATION_ORACLE, "regime_oracle", self.regime_schema, (None,)),
        }

    def splits(self) -> tuple[str, ...]:
        return tuple(self._splits)

    def origins(self, split: str) -> tuple[int, ...]:
        if split not in self._splits:
            raise ContractViolation(f"unknown split {split!r}; splits {tuple(self._splits)}")
        return self._splits[split]

    # ------------------------------------------------------------------ requests
    def request_id(self, origin: int) -> str:
        return f"toy@{origin}"

    def bundle(self, origin: int) -> ObservationBundle:
        T, H = self.context, self.horizon
        hist = self._y[origin - T: origin]
        fut = self._y[origin: origin + H]
        md = {"semantic_type": None}
        obs = (
            Observation(f"obs:target_history@{origin}", "target_history", "timeseries", FieldRole.OBSERVED, self.series_schema, hist, origin, "toy", (origin - T, origin), {"semantic_type": "observation_series"}),
            Observation(f"obs:calendar_history@{origin}", "calendar_history", "calendar", FieldRole.OBSERVED, self.calendar_schema, self._calendar[origin - T: origin], origin, "toy", (origin - T, origin), {"semantic_type": "calendar_features"}),
            Observation(f"obs:calendar_future@{origin}", "calendar_future", "calendar", FieldRole.KNOWN_FUTURE, self.future_calendar_schema, self._calendar[origin: origin + H], origin, "toy", (origin, origin + H), {"semantic_type": "calendar_features"}),
            Observation(f"obs:target_future@{origin}", "target_future", "timeseries", FieldRole.OUTCOME_ONLY, self.target_schema, fut, origin + H, "toy", (origin, origin + H), {"semantic_type": "outcome"}),
            Observation(f"obs:true_regime@{origin}", "true_regime", "oracle", FieldRole.EVALUATION_ORACLE, self.regime_schema, self._z[origin - T: origin].astype(np.float64), origin, "toy", (origin - T, origin), {"semantic_type": "regime_oracle"}),
        )
        del md
        return ObservationBundle(f"bundle:{self.request_id(origin)}", obs, origin, {"series_id": "toy"})

    def iter_requests(self, split: str, tasks: tuple[Any, ...], protocol: dict[str, Any] | None = None) -> Iterable[PredictionRequest]:
        protocol = protocol or {}
        origins = self.origins(split)
        limit = protocol.get("max_requests")
        if limit is not None:
            origins = origins[: int(limit)]
        for o in origins:
            rid = self.request_id(o)
            yield PredictionRequest(rid, self.bundle(o), tuple(t.make_query(rid) for t in tasks), o, self.grouping_metadata(rid))

    def outcome(self, request_id: str, query_id: str) -> TrainingOutcome | None:
        origin = int(request_id.split("@", 1)[1])
        H = self.horizon
        if origin + H > self.n_steps:
            return None
        values = self._y[origin: origin + H]
        return TrainingOutcome(request_id, query_id, values, np.ones_like(values, dtype=bool), origin + H, origin + H, self.target_schema)

    def grouping_metadata(self, request_id: str) -> dict[str, Any]:
        origin = int(request_id.split("@", 1)[1])
        return {"origin": origin, "series_id": "toy", "regime_at_origin": int(self._z[origin - 1])}

    # ------------------------------------------------------------------ memory seeding
    def matured_episodes(self, before_origin: int, namespace: str = "episodes") -> tuple[MemoryRecord, ...]:
        """Episodes (context window + continuation) whose continuation had fully matured
        strictly before ``before_origin``; ``available_at`` = maturity time."""
        T, H = self.context, self.horizon
        out = []
        for o in range(T, before_origin - H + 1):
            out.append(MemoryRecord(
                f"episode@{o}", namespace, "episode", available_at=o + H, event_span=(o - T, o + H),
                arrays={"window": self._y[o - T: o].copy(), "continuation": self._y[o: o + H].copy()},
                data={"origin": o, "series_id": "toy"}))
        return tuple(out)
