"""Alternative cap writer: bounded residual. Selecting it is a config change on the cap."""

from __future__ import annotations

from typing import Any

import numpy as np


class BoundedPointWriter:
    writer_id = "bounded_point_writer"
    writer_version = "0.1.0"
    forecast_type = "point"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.bound = float((config or {}).get("bound", 1.0))

    def correction_dim(self, baseline_shape: tuple[int, ...]) -> int:
        return int(np.prod(baseline_shape))

    def write(self, baseline: Any, correction: Any, xp: Any) -> Any:
        return baseline + self.bound * xp.tanh(xp.reshape(correction, baseline.shape) / self.bound)

    def is_baseline_preserving_at_zero(self) -> bool:
        return True
