"""Contributor test kit: fixture harness, contract-test mixins and causality helpers."""

from .causality import assert_target_free, drop_outcome_fields, perturb_outcome_fields
from .contract_tests import (
    DifferentiableModuleMixin,
    FrozenModuleMixin,
    MemoryStoreContractMixin,
    ModuleContractMixin,
    ReasonerContractMixin,
    ScenarioContractMixin,
    WriterContractMixin,
)
from .harness import as_of_origin, make_context, port_value, random_inputs_for, synthetic_request

__all__ = ["assert_target_free", "drop_outcome_fields", "perturb_outcome_fields", "DifferentiableModuleMixin", "FrozenModuleMixin",
           "MemoryStoreContractMixin", "ModuleContractMixin", "ReasonerContractMixin", "ScenarioContractMixin", "WriterContractMixin",
           "as_of_origin", "make_context", "port_value", "random_inputs_for", "synthetic_request"]
