"""Descriptors only. Importing this module must stay cheap and must not import numpy-heavy
implementations or the (absent) heavy dependency."""

from ness.plugin_api import PluginDescriptor

_P = "ness_example_plugin"

TOY_LINEAR_AR_SUBSTRATE = PluginDescriptor("toy_linear_ar_substrate", "substrate", "0.1.0", f"{_P}.ar_substrate:ToyLinearARSubstrate", (), "frozen lag-embedding AR provider (third family)")
DETERMINISTIC_THRESHOLD_REASONER = PluginDescriptor("deterministic_threshold_reasoner", "reasoner", "0.1.0", f"{_P}.reasoner:DeterministicThresholdReasoner", (), "one-hot regime assignment by threshold")
BOUNDED_POINT_WRITER = PluginDescriptor("bounded_point_writer", "writer", "0.1.0", f"{_P}.writer:BoundedPointWriter", (), "tanh-bounded point residual writer")
HEAVY_DEPENDENCY_SUBSTRATE = PluginDescriptor("heavy_dependency_substrate", "substrate", "0.1.0", f"{_P}.heavy:HeavyDependencySubstrate", ("torch_fake_dependency",), "declares a dependency that is not installed; exists to prove scoped-dependency behaviour (T41)")
