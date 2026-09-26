"""Numerical runtimes. ``numpy`` is always available; ``jax`` is imported lazily and only
by modules that declare it. The differentiable reference runtime lives in ``jax_runtime``."""

from .ops import BOUNDARY_KINDS, MERGE_OPS

DIFFERENTIABLE_RUNTIMES = frozenset({"jax"})


def array_namespace(runtime: str):
    """Return the array namespace for a runtime name (numpy | jax)."""
    if runtime == "numpy" or runtime == "host":
        import numpy as np
        return np
    if runtime == "jax":
        from .jax_runtime import jnp  # lazy; raises PluginDependencyMissing if absent
        return jnp
    raise ValueError(f"unknown runtime {runtime!r}")


__all__ = ["BOUNDARY_KINDS", "MERGE_OPS", "DIFFERENTIABLE_RUNTIMES", "array_namespace"]
