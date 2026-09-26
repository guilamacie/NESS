"""Causality / target-isolation helpers (T04, T21, T38, T44)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np

from ...contracts import FieldRole, Observation, ObservationBundle, PredictionRequest


def perturb_outcome_fields(request: PredictionRequest, scale: float = 1e3, seed: int = 0) -> PredictionRequest:
    """Return a request whose outcome-only / oracle fields are replaced by garbage.
    A target-free predictor must produce identical outputs for both requests."""
    rng = np.random.default_rng(seed)
    obs = []
    for o in request.bundle.observations:
        if o.role in (FieldRole.OUTCOME_ONLY, FieldRole.EVALUATION_ORACLE):
            obs.append(replace(o, payload=np.asarray(o.array) + scale * rng.normal(size=o.array.shape)))
        else:
            obs.append(o)
    bundle = ObservationBundle(request.bundle.bundle_id, tuple(obs), request.bundle.origin, dict(request.bundle.metadata))
    return replace(request, bundle=bundle)


def drop_outcome_fields(request: PredictionRequest) -> PredictionRequest:
    obs = tuple(o for o in request.bundle.observations if o.role not in (FieldRole.OUTCOME_ONLY, FieldRole.EVALUATION_ORACLE))
    return replace(request, bundle=ObservationBundle(request.bundle.bundle_id, obs, request.bundle.origin, dict(request.bundle.metadata)))


def assert_target_free(predict: Callable[[PredictionRequest], dict[str, Any]], request: PredictionRequest) -> None:
    """``predict`` maps a request to {query_id: dense forecast}. Asserts equality across
    the original, outcome-perturbed and outcome-dropped variants."""
    base = predict(request)
    for variant in (perturb_outcome_fields(request), drop_outcome_fields(request)):
        other = predict(variant)
        assert set(base) == set(other)
        for k in base:
            np.testing.assert_array_equal(np.asarray(base[k]), np.asarray(other[k]), err_msg=f"prediction {k} changed with withheld outcomes")
