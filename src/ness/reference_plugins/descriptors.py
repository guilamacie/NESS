"""Lightweight plugin descriptors for the reference plugins.

Registered through the ``ness.plugins`` entry-point group in ``pyproject.toml`` exactly
as an external package would. Importing this module imports no plugin implementation
(and in particular no JAX).
"""

from __future__ import annotations

from ..plugin_api.registry import PluginDescriptor

_RP = "ness.reference_plugins"

TOY_TEMPORAL_DATASET = PluginDescriptor("toy_temporal_dataset", "scenario", "1.0.0", "ness.scenarios.toy_temporal:ToyTemporalDataset", (), "synthetic 2-channel, 2-regime rolling-origin series")
TOY_REGIME_DATASET = PluginDescriptor("toy_regime_dataset", "scenario", "1.0.0", "ness.scenarios.toy_regime:ToyRegimeDataset", (), "interpretable y/x/event series with a hidden two-state regime and known change points")
TOY_FROZEN_TRANSFORMER = PluginDescriptor("toy_frozen_transformer", "substrate", "1.0.0", f"{_RP}.frozen_substrates:ToyFrozenTransformer", (), "tiny frozen causal transformer (numpy)")
TOY_FROZEN_CONV = PluginDescriptor("toy_frozen_conv", "substrate", "1.0.0", f"{_RP}.frozen_substrates:ToyFrozenConv", (), "tiny frozen causal conv stack (numpy)")
TINY_UPPER_TRANSFORMER = PluginDescriptor("tiny_upper_transformer", "upper_module", "1.0.0", f"{_RP}.upper_modules:TinyUpperTransformer", ("jax",), "trainable one-block transformer (jax)")
TINY_UPPER_MLP = PluginDescriptor("tiny_upper_mlp", "upper_module", "1.0.0", f"{_RP}.upper_modules:TinyUpperMLP", ("jax",), "trainable pooled MLP (jax)")
SIMPLE_TEMPORAL_SEMANTICS = PluginDescriptor("simple_temporal_semantics", "semantic_adapter", "1.0.0", f"{_RP}.semantic:SimpleTemporalSemantics", (), "trend/volatility/regime-cue features")
TYPED_PROGRAM = PluginDescriptor("typed_program", "program", "1.0.0", f"{_RP}.program_module:TypedProgram", (), "executes one ness.program/1 IR with the reference interpreter")
EXACT_FINITE_REGIME_REASONER = PluginDescriptor("exact_finite_regime_reasoner", "reasoner", "1.0.0", "ness.probabilistic.reasoners:ExactFiniteRegimeReasoner", (), "exact enumeration over a finite latent")
IMPORTANCE_SAMPLING_REGIME_REASONER = PluginDescriptor("importance_sampling_regime_reasoner", "reasoner", "1.0.0", "ness.probabilistic.reasoners:ImportanceSamplingRegimeReasoner", (), "self-normalised IS over the same finite latent")
ANALOGUE_MEMORY_QUERY = PluginDescriptor("analogue_memory_query", "memory_query", "1.0.0", f"{_RP}.memory_query:AnalogueMemoryQuery", (), "bounded analogue retrieval on a pinned view")
RESIDUAL_POINT_CAP = PluginDescriptor("residual_point_cap", "cap", "1.0.0", f"{_RP}.caps:ResidualPointCap", ("jax",), "consumer + point residual writer")
RESIDUAL_QUANTILE_CAP = PluginDescriptor("residual_quantile_cap", "cap", "1.0.0", f"{_RP}.caps:ResidualQuantileCap", ("jax",), "consumer + monotone quantile writer")
POINT_RESIDUAL_WRITER = PluginDescriptor("point_residual_writer", "writer", "1.0.0", f"{_RP}.writers_consumers:PointResidualWriter", (), "mu0 + a")
MONOTONE_QUANTILE_WRITER = PluginDescriptor("monotone_quantile_writer", "writer", "1.0.0", f"{_RP}.writers_consumers:MonotoneQuantileWriter", (), "PDF §9.5 gap-multiplier writer")
CONCAT_LINEAR_CONSUMER = PluginDescriptor("concat_linear_consumer", "consumer", "1.0.0", f"{_RP}.writers_consumers:ConcatLinearConsumer", (), "zero-init linear/MLP consumer")
POINT_FORECAST_TASK = PluginDescriptor("point_forecast_task", "task", "1.0.0", "ness.tasks.forecast_tasks:PointForecastTask", (), "as-of-origin point forecast task")
QUANTILE_FORECAST_TASK = PluginDescriptor("quantile_forecast_task", "task", "1.0.0", "ness.tasks.forecast_tasks:QuantileForecastTask", (), "as-of-origin quantile forecast task (pinball)")
IN_MEMORY_SNAPSHOT_STORE = PluginDescriptor("in_memory_snapshot_store", "memory_store", "1.0.0", "ness.memory.store:InMemorySnapshotStore", (), "reference immutable snapshot store")
BP_DIRECT = PluginDescriptor("bp_direct", "learning_rule", "1.1.0", "ness.backends.jax.region:BPDirectRule", ("jax",), "BP through the deployed direct computation (jax backend; optional batched/data-parallel evaluation)")
JAX_DENSE_WORKSPACE_CAP = PluginDescriptor("jax_dense_workspace_cap", "cap", "1.0.0", f"{_RP}.jax_dense_workspace:JaxDenseWorkspaceCap", ("jax",), "dense workspace cap on the jax backend; parity twin of fabricpc_residual_cap")

ALL = (TOY_TEMPORAL_DATASET, TOY_REGIME_DATASET, TOY_FROZEN_TRANSFORMER, TOY_FROZEN_CONV, TINY_UPPER_TRANSFORMER, TINY_UPPER_MLP, SIMPLE_TEMPORAL_SEMANTICS,
       TYPED_PROGRAM, EXACT_FINITE_REGIME_REASONER, IMPORTANCE_SAMPLING_REGIME_REASONER, ANALOGUE_MEMORY_QUERY, RESIDUAL_POINT_CAP,
       RESIDUAL_QUANTILE_CAP, POINT_RESIDUAL_WRITER, MONOTONE_QUANTILE_WRITER, CONCAT_LINEAR_CONSUMER, POINT_FORECAST_TASK,
       QUANTILE_FORECAST_TASK, IN_MEMORY_SNAPSHOT_STORE, BP_DIRECT, JAX_DENSE_WORKSPACE_CAP)


def register_all(registry) -> None:
    """Register the reference plugins in-process (for tests/notebooks without an install)."""
    for d in ALL:
        registry.register(d)
