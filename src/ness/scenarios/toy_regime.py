"""``ToyRegimeDataset``: an interpretable synthetic forecasting problem with a hidden regime.

Variables (all deterministic from ``seed``):

* auxiliary observed channel      x_t = phi_x x_{t-1} + sigma_x eps^x_t
* known event indicator           e_t in {0, 1}: scheduled blocks of ``event_length`` steps every
                                  ``event_period`` steps (known in advance -> role known_future)
* hidden regime                   r_t in {NORMAL, SHIFT}: piecewise constant, switching every
                                  ``segment_length`` steps (change points are recorded)
* target                          NORMAL: y_t = a y_{t-1} + b_N x_{t-1} + c_N e_t + sigma_N eps_t
                                  SHIFT : y_t = a y_{t-1} + b_S x_{t-2} + c_S e_t + sigma_S eps_t
                                  (different lag, opposite sign, stronger event effect, more noise)

The predictor sees ``series_history`` (y, x), ``event_history`` and the known-future
``event_future``. ``target_future`` (outcome_only) and ``true_regime`` (evaluation_oracle) are
carried in the raw bundle for orchestration/diagnostics and removed by every task view (T44).
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
from .rolling_origin import rolling_origins

NORMAL, SHIFT = 0, 1


class ToyRegimeDataset(BaseModule):
    plugin_id = "toy_regime_dataset"
    plugin_version = "1.0.0"
    module_kind = "scenario"
    runtime = "host"
    state_schema_id = "ness.scenario.toy_regime/1"

    def validate_config(self) -> None:
        c = self.config
        self.n_steps = int(c.get("n_steps", 720))
        self.context = int(c.get("context", 32))
        self.horizon = int(c.get("horizon", 4))
        self.seed = int(c.get("seed", 2026))
        self.event_period = int(c.get("event_period", 25))
        self.event_length = int(c.get("event_length", 3))
        self.segment_length = int(c.get("segment_length", 90))
        self.a = float(c.get("a", 0.6))
        self.phi_x, self.sigma_x = float(c.get("phi_x", 0.8)), float(c.get("sigma_x", 0.3))
        self.b = (float(c.get("b_normal", 0.4)), float(c.get("b_shift", -0.4)))
        self.lag = (int(c.get("lag_normal", 1)), int(c.get("lag_shift", 2)))
        self.cc = (float(c.get("c_normal", 0.5)), float(c.get("c_shift", 1.5)))
        self.sigma = (float(c.get("sigma_normal", 0.10)), float(c.get("sigma_shift", 0.20)))
        self.splits_frac = tuple(float(v) for v in c.get("split_fractions", (0.6, 0.15, 0.25)))
        self.stride = int(c.get("stride", 1))
        if abs(sum(self.splits_frac) - 1.0) > 1e-9 or len(self.splits_frac) != 3:
            raise ContractViolation("split_fractions must be three fractions summing to 1 (train, dev, test)")
        self._generate()

    # ------------------------------------------------------------------ generation
    def _generate(self) -> None:
        rng = np.random.default_rng(self.seed)
        n = self.n_steps
        r = np.array([(t // self.segment_length) % 2 for t in range(n)], dtype=int)  # alternating segments
        e = np.array([1.0 if (t % self.event_period) < self.event_length else 0.0 for t in range(n)])
        x = np.zeros(n)
        y = np.zeros(n)
        for t in range(1, n):
            x[t] = self.phi_x * x[t - 1] + self.sigma_x * rng.normal()
        for t in range(2, n):
            k = r[t]
            lagged = x[t - self.lag[k]]
            y[t] = self.a * y[t - 1] + self.b[k] * lagged + self.cc[k] * e[t] + self.sigma[k] * rng.normal()
        self._y, self._x, self._e, self._r = y, x, e, r
        self._change_points = tuple(int(t) for t in range(1, n) if r[t] != r[t - 1])
        origins = rolling_origins(n, self.context, self.horizon, self.stride)
        n_o = len(origins)
        a = int(n_o * self.splits_frac[0])
        b = int(n_o * (self.splits_frac[0] + self.splits_frac[1]))
        train = origins[:a]
        dev = tuple(o for o in origins[a:b] if o >= train[-1] + self.horizon)
        test = tuple(o for o in origins[b:] if not dev or o >= dev[-1] + self.horizon)
        self._splits = {"train": train, "dev": dev, "test": test}
        self.series_schema = series_schema(("y", "x"))
        self.event_schema = CoordinateSchema("event_indicator@step", ("time", "feature"), channel_names=("event",))
        self.event_future_schema = CoordinateSchema("event_indicator_future@step", ("horizon", "feature"), channel_names=("event",))
        self.target_schema = horizon_schema(("y",), tuple(range(1, self.horizon + 1)))
        self.regime_schema = CoordinateSchema("regime[oracle]", ("time",))

    # ------------------------------------------------------------------ descriptors
    def describe(self) -> ModuleDescriptor:
        return self._descriptor(capabilities=self.capabilities())

    def capabilities(self) -> ScenarioCapabilities:
        return ScenarioCapabilities("synthetic_regime_timeseries", tuple(self._splits), True, "coordinate_wise",
                                    tuple(self.observation_fields()), ("target_future",))

    def observation_fields(self) -> dict[str, ObservationField]:
        H = self.horizon
        return {
            "series_history": ObservationField("series_history", "timeseries", FieldRole.OBSERVED, "observation_series", self.series_schema, (None, 2)),
            "event_history": ObservationField("event_history", "event", FieldRole.OBSERVED, "event_indicator", self.event_schema, (None, 1)),
            "event_future": ObservationField("event_future", "event", FieldRole.KNOWN_FUTURE, "event_indicator", self.event_future_schema, (H, 1)),
            "target_future": ObservationField("target_future", "timeseries", FieldRole.OUTCOME_ONLY, "outcome", self.target_schema, (H, 1)),
            "true_regime": ObservationField("true_regime", "oracle", FieldRole.EVALUATION_ORACLE, "regime_oracle", self.regime_schema, (None,)),
        }

    def splits(self) -> tuple[str, ...]:
        return tuple(self._splits)

    def origins(self, split: str) -> tuple[int, ...]:
        if split not in self._splits:
            raise ContractViolation(f"unknown split {split!r}; splits {tuple(self._splits)}")
        return self._splits[split]

    # ------------------------------------------------------------------ ground truth (evaluation / artifacts only)
    def ground_truth(self) -> dict[str, np.ndarray]:
        return {"t": np.arange(self.n_steps), "y": self._y.copy(), "x": self._x.copy(), "event": self._e.copy(), "regime": self._r.copy()}

    def regime_change_points(self) -> tuple[int, ...]:
        return self._change_points

    def generation_manifest(self) -> dict[str, Any]:
        return {"plugin": self.plugin_id, "version": self.plugin_version, "config": dict(self.config), "seed": self.seed,
                "equations": {"x": "x_t = phi_x x_{t-1} + sigma_x eps", "y_normal": "y_t = a y_{t-1} + b_N x_{t-1} + c_N e_t + sigma_N eps",
                              "y_shift": "y_t = a y_{t-1} + b_S x_{t-2} + c_S e_t + sigma_S eps", "e": "1 for event_length steps every event_period steps",
                              "r": "alternates every segment_length steps starting NORMAL"},
                "parameters": {"a": self.a, "phi_x": self.phi_x, "sigma_x": self.sigma_x, "b": list(self.b), "lag": list(self.lag), "c": list(self.cc), "sigma": list(self.sigma)},
                "change_points": list(self._change_points), "splits": {k: [v[0], v[-1], len(v)] if v else [] for k, v in self._splits.items()},
                "context": self.context, "horizon": self.horizon}

    # ------------------------------------------------------------------ requests
    def request_id(self, origin: int) -> str:
        return f"regime@{origin}"

    def bundle(self, origin: int) -> ObservationBundle:
        T, H = self.context, self.horizon
        sl = slice(origin - T, origin)
        obs = (
            Observation(f"obs:series@{origin}", "series_history", "timeseries", FieldRole.OBSERVED, self.series_schema, np.stack([self._y[sl], self._x[sl]], axis=1), origin, "toy", (origin - T, origin), {"semantic_type": "observation_series"}),
            Observation(f"obs:events@{origin}", "event_history", "event", FieldRole.OBSERVED, self.event_schema, self._e[sl][:, None].copy(), origin, "toy", (origin - T, origin), {"semantic_type": "event_indicator"}),
            Observation(f"obs:events_future@{origin}", "event_future", "event", FieldRole.KNOWN_FUTURE, self.event_future_schema, self._e[origin: origin + H][:, None].copy(), origin, "toy", (origin, origin + H), {"semantic_type": "event_indicator"}),
            Observation(f"obs:target_future@{origin}", "target_future", "timeseries", FieldRole.OUTCOME_ONLY, self.target_schema, self._y[origin: origin + H][:, None].copy(), origin + H, "toy", (origin, origin + H), {"semantic_type": "outcome"}),
            Observation(f"obs:true_regime@{origin}", "true_regime", "oracle", FieldRole.EVALUATION_ORACLE, self.regime_schema, self._r[sl].astype(np.float64), origin, "toy", (origin - T, origin), {"semantic_type": "regime_oracle"}),
        )
        return ObservationBundle(f"bundle:{self.request_id(origin)}", obs, origin, {"series_id": "toy_regime"})

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
        values = self._y[origin: origin + H][:, None]
        return TrainingOutcome(request_id, query_id, values, np.ones_like(values, dtype=bool), origin + H, origin + H, self.target_schema)

    def grouping_metadata(self, request_id: str) -> dict[str, Any]:
        origin = int(request_id.split("@", 1)[1])
        dist = min(abs(origin - cp) for cp in self._change_points) if self._change_points else -1
        return {"origin": origin, "series_id": "toy_regime", "regime_at_origin": int(self._r[origin - 1]),
                "distance_to_change_point": int(dist), "event_in_horizon": int(self._e[origin: origin + self.horizon].max())}

    # ------------------------------------------------------------------ memory seeding
    def matured_episodes(self, before_origin: int, namespace: str = "episodes") -> tuple[MemoryRecord, ...]:
        T, H = self.context, self.horizon
        out = []
        for o in range(T, before_origin - H + 1):
            out.append(MemoryRecord(f"episode@{o}", namespace, "episode", available_at=o + H, event_span=(o - T, o + H),
                                    arrays={"window": np.stack([self._y[o - T: o], self._x[o - T: o]], axis=1), "continuation": self._y[o: o + H][:, None].copy()},
                                    data={"origin": o, "series_id": "toy_regime"}))
        return tuple(out)
