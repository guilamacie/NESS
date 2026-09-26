"""Named learning profiles (PDF §11.1). Existing in the vocabulary is not the same as
being verified: selecting an unverified profile fails explicitly."""

from __future__ import annotations

from ..contracts import Capability, CapabilitySet, CapabilityStatus, UnsupportedCapability

LEARNING_PROFILES = CapabilitySet((
    Capability("bp_direct", CapabilityStatus.VERIFIED, "tests/test_learning.py (jax backend)"),  # == ff_bp on the jax backend
    Capability("ff_bp", CapabilityStatus.VERIFIED, "alias of bp_direct"),
    Capability("fabricpc_pc_local", CapabilityStatus.VERIFIED, "tests/fabricpc_backend/test_fabricpc_integration.py: clamped settle + local weight gradients"),
    Capability("fabricpc_bp_through_inference", CapabilityStatus.VERIFIED, "tests/fabricpc_backend/test_fabricpc_parity.py: BP of the task loss through the deployed (free) FabricPC inference"),
    Capability("workspace_pc_local", CapabilityStatus.VERIFIED, "alias of fabricpc_pc_local with inference profile fabricpc_spc"),
    Capability("workspace_epc_local", CapabilityStatus.VERIFIED, "alias of fabricpc_pc_local with inference profile fabricpc_epc (local rule after ePC settle)"),
    Capability("workspace_bp_unroll", CapabilityStatus.VERIFIED, "alias of fabricpc_bp_through_inference with a settling inference profile"),
    Capability("reliability_bp_unroll", CapabilityStatus.UNSUPPORTED, "reference44 reliability settling not ported"),
    Capability("workspace_ep_centered", CapabilityStatus.UNSUPPORTED, "centered nudge rule: FabricPC 0.6 has no nudge phase and NESS has not implemented one"),
))

# Scientific profile names resolve to the plugin that implements them; the plugin's own
# AlgorithmSpec (recorded in manifests) states the exact semantics, never the name alone.
ALIASES = {"ff_bp": "bp_direct", "workspace_pc_local": "fabricpc_pc_local", "workspace_epc_local": "fabricpc_pc_local",
           "workspace_bp_unroll": "fabricpc_bp_through_inference"}


def resolve_learning_profile(name: str) -> str:
    cap = LEARNING_PROFILES.get(name)
    if cap.status != CapabilityStatus.VERIFIED:
        raise UnsupportedCapability(f"learning profile {name!r} is {cap.status.value}: {cap.evidence}. No fallback is applied.")
    return ALIASES.get(name, name)
