"""NESS JAX backend (Level B): differentiable region, BP-through-direct-computation rule,
NESS-owned batched/sharded gradient evaluation. Importing this package imports JAX."""

from .region import BatchItem, BPDirectRule, DifferentiableRegion, jax, jnp, sharded_mesh

__all__ = ["BatchItem", "BPDirectRule", "DifferentiableRegion", "jax", "jnp", "sharded_mesh"]
