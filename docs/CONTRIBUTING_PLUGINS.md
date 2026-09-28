# Contributing plugins

A plugin package depends on `ness.contracts` and `ness.plugin_api` only. It never imports
`ness.runtime` or edits core code. Registration is a Python entry point in the group
`ness.plugins` whose value resolves to a `PluginDescriptor` (or a list of them):

```toml
[project.entry-points."ness.plugins"]
my_substrate = "my_pkg.descriptors:MY_SUBSTRATE"
```

```python
MY_SUBSTRATE = PluginDescriptor("my_substrate", "substrate", "1.0.0", "my_pkg.substrate:MySubstrate", requires=("torch",))
```

## Lifecycle and state contract (addendum §7)

Every module: `plugin_id`, semantic `plugin_version`, `module_kind`, `runtime`,
`state_schema_id`, `describe() -> ModuleDescriptor` (ports, parameter groups, capabilities,
`config_hash`), `initialize(rng) -> ModuleState`, `forward(inputs, state, ctx) -> ModuleOutputs`,
`snapshot_state(state) -> StateSnapshot`, `restore_state(snapshot) -> ModuleState`, optional
`migrate_state(old_version, snapshot)`. `BaseModule` implements canonical config hashing and a
restricted-data snapshot (arrays + JSON; object dtype refused) that fails closed on plugin id,
major version, state schema id and config hash. Trainable modules add
`apply(params, dense_inputs, state, xp)`.

Kinds: `substrate, upper_module, semantic_adapter, program, reasoner, memory_query, cap,
consumer, writer, task, scenario, memory_store, learning_rule`. Selector schemes:
`substrate://`, `module://` (upper_module, cap), `semantic://`, `program://`, `reasoner://`,
`memory://`, `observation://`.

## Descriptor checklist

* every input port: kind, shape (`None` variable, `()` any), `accepts_semantic_types`, `optional`;
* every output port: kind, shape, `semantic_type`, `availability_role` (never `outcome_only`),
  `differentiability` (`differentiable` only for ports whose producer runs in the differentiable
  runtime and whose parameters you intend to train), `coordinate_schema_id` for forecasts
  (must equal the task's target schema id);
* parameter groups with truthful `mutability` and `runtime`;
* `capabilities` record (`SubstrateCapabilities`, `ProbabilisticReasonerCapabilities`, or a dict);
* `requires` for optional dependencies.

## Runtimes and bootstrap

Declare `runtime` truthfully (`numpy`/`host`, `jax`, `fabricpc`, or your own). Never query
devices, call `setup_jax`, or touch XLA flags at import or inside `forward`; if your plugin needs
an initialised backend outside a `NessSystem`, call
`ness.runtimes.bootstrap.ensure_bootstrapped("jax"|"fabricpc")`. Learning rules declare
`required_runtime()` and may only own parameter groups in that runtime. Optional heavy
dependencies go in `requires` so unrelated configurations validate without them.

## Test kit

`ness.plugin_api.testkit` provides `ModuleContractMixin`, `FrozenModuleMixin`,
`DifferentiableModuleMixin`, `ReasonerContractMixin`, `ScenarioContractMixin`,
`MemoryStoreContractMixin`, `WriterContractMixin`, plus `synthetic_request`, `make_context`,
`port_value`, `random_inputs_for`, and causality helpers (`assert_target_free`,
`perturb_outcome_fields`, `drop_outcome_fields`). Subclass a mixin, override `make_module()`
(and `make_inputs()` when random dense inputs are not appropriate), and pytest collects the
checks. See `examples/external_plugin_example/tests`.

## Definition of done for a plugin

descriptor + implementation + deterministic fixture (`initialize` from a seed) + contract tests
+ a config fragment showing how it is wired + documentation of its capabilities and
limitations. If it cannot register and run without editing `src/ness`, the plugin (or the core
API) is not finished - open an issue against the core rather than patching it locally.
* The plugin package modifies nothing under `src/ness/`; `ness audit --strict` passes in its environment (see `docs/AGENT_HANDOFF.md` §1 for the file boundary).
