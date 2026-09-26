"""Plugin descriptors for the FabricPC backend (metadata only; no FabricPC import)."""

from ...plugin_api.registry import PluginDescriptor

_P = "ness.backends.fabricpc"

FABRICPC_RESIDUAL_CAP = PluginDescriptor("fabricpc_residual_cap", "cap", "1.0.0", f"{_P}.module:FabricPCResidualCap", ("fabricpc", "jax"),
                                         "FabricPC-backed workspace cap: dense PC graph (optionally cyclic) with feedforward/sPC/ePC free inference and a residual point writer")
FABRICPC_PC_LOCAL = PluginDescriptor("fabricpc_pc_local", "learning_rule", "1.0.0", f"{_P}.rules:FabricPCPCLocalRule", ("fabricpc", "jax"),
                                     "clamped-target settle + FabricPC local weight gradients (means per prediction)")
FABRICPC_BP_THROUGH_INFERENCE = PluginDescriptor("fabricpc_bp_through_inference", "learning_rule", "1.0.0", f"{_P}.rules:FabricPCBPThroughInferenceRule", ("fabricpc", "jax"),
                                                 "reverse-mode BP of the NESS task loss through the deployed (target-free) FabricPC inference")
ALL = (FABRICPC_RESIDUAL_CAP, FABRICPC_PC_LOCAL, FABRICPC_BP_THROUGH_INFERENCE)
