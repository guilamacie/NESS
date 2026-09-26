"""Explicit failure vocabulary.

NESS never falls back silently. Every unsupported capability, causal violation, or
incompatible version is a distinct exception so tests can assert on the *reason*.
"""

from __future__ import annotations


class NessError(Exception):
    """Base class for all NESS errors."""


class ContractViolation(NessError):
    """A typed contract (schema, port, evidence, forecast) was violated."""


class ValidationError(ContractViolation):
    """Configuration or specification failed validation before execution."""


class CompositionError(ValidationError):
    """The macro composition graph is invalid (bad selector, cycle, incompatible ports)."""


class AccessPolicyViolation(ContractViolation):
    """A component tried to read evidence the task's access policy forbids."""


class CausalityViolation(AccessPolicyViolation):
    """Information not available at the request's origin reached target-free prediction."""


class UnsupportedCapability(NessError):
    """A requested capability is declared but not verified/implemented. No fallback."""


class GradientBoundaryError(ContractViolation):
    """A derivative path was requested across a declared stop boundary or runtime."""


class IncompatibleVersion(ContractViolation):
    """Schema/plugin/state version mismatch without an explicit migration."""


class CheckpointError(NessError):
    """Checkpoint staging, publication, or restore failed."""


class MemoryConflict(NessError):
    """A memory append used a stale expected head."""


class BudgetExceeded(NessError):
    """A declared deterministic budget (fuel, rows, queries) was exhausted."""


class PluginNotFound(NessError):
    """No registered plugin matches the requested id/kind."""


class PluginDependencyMissing(UnsupportedCapability):
    """A plugin's declared optional dependency is not importable."""
