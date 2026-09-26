"""Inference profiles are independent of learning rules (PDF §11)."""

from __future__ import annotations

from ..contracts import Capability, CapabilitySet, CapabilityStatus, UnsupportedCapability

INFERENCE_PROFILES = CapabilitySet((
    Capability("direct", CapabilityStatus.VERIFIED, "single forward pass; no temporary inference state"),
    Capability("fabricpc_feedforward", CapabilityStatus.VERIFIED, "FabricPC FeedforwardStateInit, no settling (tests/fabricpc_backend)"),
    Capability("fabricpc_spc", CapabilityStatus.VERIFIED, "FabricPC InferenceSGD state-based settling with target free (tests/fabricpc_backend)"),
    Capability("fabricpc_epc", CapabilityStatus.VERIFIED, "FabricPC EPCInference error-parameterised settling with target free (tests/fabricpc_backend)"),
    Capability("fabricpc_spc_recurrent", CapabilityStatus.EXPERIMENTAL, "sPC settling on a cyclic FabricPC graph (unroll=U): runs and lowers energy; unrolled-cycle semantics are FabricPC's"),
    Capability("reliability_settling", CapabilityStatus.UNSUPPORTED, "finite reliability-logit updates (reference44) not ported"),
    Capability("recurrent_workspace", CapabilityStatus.UNSUPPORTED, "composition-level InferenceWorkspace regions have no verified solver; use a FabricPC workspace node"),
    Capability("query_settle", CapabilityStatus.UNSUPPORTED, "outer query-settle loop not implemented"),
))


def resolve_inference_profile(name: str, allow_experimental: bool = False) -> str:
    cap = INFERENCE_PROFILES.get(name)
    if cap.status == CapabilityStatus.VERIFIED or (allow_experimental and cap.status == CapabilityStatus.EXPERIMENTAL):
        return name
    raise UnsupportedCapability(f"inference profile {name!r} is {cap.status.value}: {cap.evidence}"
                                + (" (set inference.allow_experimental: true to run it as an experimental, non-scientific profile)" if cap.status == CapabilityStatus.EXPERIMENTAL else ""))
