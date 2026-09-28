"""A program primitive contributed by a third-party package through the `ness.primitives`
entry point: typed programs in any configuration can `call` it once this package is installed,
without any change to ness core."""

from __future__ import annotations

import numpy as np

from ness.contracts import Knownness
from ness.symbolic.operators import PrimitiveResult, PrimitiveSpec


def _window_max(series: np.ndarray, window: int) -> PrimitiveResult:
    s = np.asarray(series, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] < int(window) or int(window) < 1:
        return PrimitiveResult(np.zeros(s.shape[1] if s.ndim == 2 else 1), Knownness.MISSING, f"need a [T, D] series with T >= window >= 1, got {s.shape}")
    return PrimitiveResult(s[-int(window):].max(axis=0), Knownness.KNOWN)


WINDOW_MAX = PrimitiveSpec("window_max", {"series": "numeric_series", "window": "int"}, "numeric_vector", "1", "pure", "inputs", False, 2,
                           _window_max, "trailing maximum per channel (third-party primitive)")
