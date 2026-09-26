"""Compatibility shim: the JAX runtime now lives in ``ness.backends.jax``."""

from ..backends.jax.region import BatchItem, BPDirectRule, DifferentiableRegion, jax, jnp, sharded_mesh  # noqa: F401

__all__ = ["jnp", "jax", "DifferentiableRegion", "BPDirectRule", "BatchItem", "sharded_mesh"]
