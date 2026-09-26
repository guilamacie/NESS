# Handoff: implementing your own NESS instance with your own models

Audience: an AI coding agent (or engineer) who will plug **their own models** into NESS as the
layers of a new instance - a real lower substrate, a real upper module, their own semantic
module, reasoner, memory backend, cap, task and dataset - and run controlled experiments.

Read `docs/ARCHITECTURE.md` §1-§5 first. This document is the operational checklist.

## 0. Non-negotiables you must preserve

1. **Target-free prediction.** Nothing with role `outcome_only` or `evaluation_oracle` may be
   read by a node. The compiler enforces it on every edge and the transaction enforces it on the
   permitted bundle; do not add code paths that bypass `PredictionTask.permitted_view`.
2. **Typed evidence, honest semantics.** A forecast implements only the operations it supports.
   A reasoner's `HypothesisEvidence.weight_semantics` must be true (`exact_posterior` only for
   exact inference over a complete latent). Unknown values carry `knownness=False`, never 0.
3. **Memory is pinned.** A module reads `ctx.memory_views[store]`; it never receives a store
   handle to mutate inside a prediction. Learner appends go to a branch and are published
   between updates.
4. **Runtimes do not exchange gradients.** Declare `runtime` truthfully (`numpy`, `jax`,
   `fabricpc`, or your own such as `torch`). A module in another runtime is a constant to the JAX
   region and to FabricPC workspaces; a rule may only own parameters in its own runtime
   (`GradientBoundaryError` otherwise). A cross-runtime derivative bridge is a *new verified
   capability*, not an array copy.
4b. **One bootstrap owner.** Never call `jax.devices()`, `fabricpc.setup_jax()` or mutate
   `XLA_FLAGS` in a module or at import. `NessSystem.build`/the runner/CLI call
   `ness.runtimes.bootstrap.bootstrap` first; a plugin needing the backend outside a system calls
   `ensure_bootstrapped("jax"|"fabricpc")`. Device counts come from the experiment's
   `runtime.devices` spec, never from code.
5. **Inference profile and learning rule are separate declarations.** Do not fold settling or
   PC rules into a module's `forward`.
6. **No silent fallbacks.** If a capability is missing, raise `UnsupportedCapability` (or a
   `PluginDependencyMissing`). Never substitute another algorithm.
7. **Whole-system identity.** Anything that changes predictions must be in module state
   (`params`/`buffers`/`meta`) or config, so it lands in the manifest and checkpoint.

## 1. Create a plugin package (never edit `src/ness`)

```
my_ness_instance/
  pyproject.toml            # entry points under [project.entry-points."ness.plugins"]
  src/my_ness_instance/
    descriptors.py          # PluginDescriptor objects only; no heavy imports here
    substrate.py            # your lower model
    upper.py                # your trainable upper module (runtime "jax" for BP; else STOP)
    semantic.py, reasoner.py, memory.py, cap.py, task.py, scenario.py  (as needed)
  configs/my_experiment.yaml
  tests/test_contracts.py   # subclass ness.plugin_api.testkit mixins
```

`examples/external_plugin_example` is a complete, tested template. Copy it.

Descriptor rules: `PluginDescriptor(plugin_id, kind, version, "module:Class", requires=(...))`.
`requires` lists import names of optional heavyweight dependencies (`"torch"`, `"transformers"`,
`"timesfm"`); the registry checks importability before importing your module, so a config that
does not use your plugin validates on machines without those dependencies.

## 2. Wrap a real lower model as a substrate

```python
class MyFrozenProvider(BaseModule):
    plugin_id = "my_frozen_provider"; plugin_version = "1.0.0"
    module_kind = "substrate"; runtime = "torch"          # truthful runtime
    state_schema_id = "my.substrate/1"; requires = ("torch",)

    def describe(self):
        return self._descriptor(
            input_ports=(PortSpec("history", PortSchema("dense", (None, D)), "observation_series",
                                  FieldRole.OBSERVED, Differentiability.STOP, accepts_semantic_types=("observation_series",)),),
            output_ports=(
                PortSpec("state/layer_8", PortSchema("dense", (None, W)), "neural_state", FieldRole.DERIVED, Differentiability.STOP, "neural[my:W]"),
                PortSpec("forecast/quantiles", PortSchema("quantile_forecast", (H, D, Q)), "quantile_forecast", FieldRole.DERIVED,
                         Differentiability.STOP, quantile_schema(channels, horizons, levels).schema_id),
            ),
            parameter_groups=(ParameterGroupSpec("frozen", "frozen", "torch", False),),
            capabilities=SubstrateCapabilities("torch", "frozen_external", ("state/layer_8",), ("forecast/quantiles",), "stop",
                                               weights_digest="<sha256 of the checkpoint>", hidden_state_support="verified", license_review="approved:<ref>"),
        )

    def initialize(self, rng):
        # load weights in eval mode; put the checkpoint digest in meta so it enters the manifest
        return ModuleState(meta={"weights_digest": ..., "revision": ...})

    def forward(self, inputs, state, ctx):
        x = inputs["history"].dense()                      # numpy [T, D]
        with torch.no_grad(): ...                          # eval mode, detached
        return ModuleOutputs(ports={"state/layer_8": np.asarray(h), "forecast/quantiles": QuantileForecast(q, target, levels)},
                             diagnostics={"context": T}, cost=float(T * W))
```

Rules: expose only ports you can actually read (do not invent hidden states from a
forecast-only API); store raw native outputs unmodified (a repair is a separate declared step);
declare normalisation behaviour in `diagnostics`/capabilities; put anything that affects
outputs (dropout off, preprocessing version, quantile grid) in `meta`/config so it is hashed.
Run `FrozenModuleMixin` on it. Cache heavy outputs *outside* the module keyed by
`(request canonical hash, config_hash)` if you need speed; the cached value must be identical to
the uncached one.

## 2b. Choose the numerical backend for trainable parts

| you want | use | learning rules |
|---|---|---|
| generic differentiable module trained by BP | NESS JAX backend (`runtime = "jax"`, `apply(...)`) | `bp_direct` (per-item, `batched`, or `data_parallel`) |
| predictive-coding / ePC / recurrent workspace | FabricPC backend: reuse `fabricpc_residual_cap` (dense PC graph) or write a cap that builds its own FabricPC graph through the compat adapter | `fabricpc_pc_local` (PC local rule), `fabricpc_bp_through_inference` (BP through the deployed, target-free inference) |
| both in one arm | jax upper + FabricPC cap with `learning.credit.overrides` | hybrid: each rule owns its runtime's groups |

Declare `runtime:` in the experiment (`backend: auto|numpy|jax|fabricpc`, `platform`,
`devices: {platform, selection|ids, count, fallback}`, `x64`, `profile: development|scientific`).
`profile: scientific` refuses unverified FabricPC versions and a late `setup_jax`. Read
`docs/FABRICPC_BACKEND.md` before writing a FabricPC-backed module: prediction must clamp
inputs only; the PC clamp belongs to the learning rule; record a complete `algorithm_spec()`.

## 3. Wrap a trainable upper module

For BP through NESS your module must run in the differentiable reference runtime (`jax`) and
implement `apply(params, dense_inputs, state, xp)` as a pure function (see
`reference_plugins/upper_modules.py`). Flax/Equinox/Haiku modules work: convert their parameter
pytree to `{group: {name: array}}` in `initialize` and call them inside `apply`. Declare
`ParameterGroupSpec(name, "trainable", "jax", True)` and mark output ports `DIFFERENTIABLE`.
Run `DifferentiableModuleMixin` (checks apply==forward and gradients vs finite differences).

If your upper module is PyTorch and you want to train it: there is no verified torch runtime
or bridge. Options: (a) port to JAX, (b) freeze it (`STOP`) and train only JAX/FabricPC caps on
its detached arrays (the intended architecture for TimesFM/ViT providers), (c) implement a
`torch` backend under `ness/backends/torch/` with its own region, rule, bootstrap hook,
capability probes, tests and an ADR. Do not fake (c) with array copies.

## 4. Semantic / neuro-symbolic modules reading several earlier ports

Declare one input port per source you want (`raw`, `neural`, `base_context`, ...), with
`accepts_semantic_types`. In config, wire each to any *permitted* earlier port:
`observation://...`, `substrate://base/state/layer_8`, `module://upper/hidden`. Return
`FeatureEvidence`/`HypothesisEvidence`/`BindingEvidence` with `knownness`, and set
`ModuleOutputs.used_inputs` to the ports you actually consumed so provenance is exact. Use
boundary transforms in config (`last_step`, `mean_pool`, `flatten`) instead of reshaping inside
the module when the reduction is a wiring choice rather than a scientific one.

## 5. Probabilistic reasoners (Pyro / NumPyro / custom)

Implement kind `reasoner`, return `ProbabilisticReasonerCapabilities` (runtime, inference
methods, exact/approximate, gradient boundary `stop` unless you verified a JAX-native path,
serialization version), consume declared feature ports, and lower outputs to
`HypothesisEvidence` (+ optional `FeatureEvidence` moments, `PredictiveEvidence` only if you
really produce a forecast in a declared space) plus an `ExecutionTrace` with method, samples,
ESS/ELBO, convergence, truncation and cost. Weight semantics: `approximate_posterior` for MC/VI,
`normalized_beam` for retained beams (state `omitted_mass` if unknown), `deterministic_assignment`
for decisions. Compare against `exact_finite_regime_reasoner` on a 2-4 state model in your tests
(the pattern of `test_T37_*`). Separate model parameters, guide parameters, request-local latent
state and RNG in `ModuleState` groups so they checkpoint independently.

## 6. Memory backends (Hyperon or other)

Implement the `MemoryStore` protocol (`describe, open_view, query, fork, append, seal`, plus
`export_snapshot/import_snapshot` for checkpoints). `fork` must isolate (not alias); `append`
must honour `expected_head` and idempotent event ids; `open_view` must filter by
`AvailabilityCut` and namespaces **before** any ranking; `seal` must be content-addressed.
Run `MemoryStoreContractMixin`. Build a `memory_query` module that reads only
`ctx.memory_views[store]`. Atomspace-backed: rebuild a private read-only space from a sealed
snapshot; `SpaceRef.copy()` is not a fork.

## 7. Caps, writers, consumers

Prefer reusing `residual_point_cap` / `residual_quantile_cap` with a new writer or consumer
plugin (both are pure functions over `xp`). A new writer must state
`is_baseline_preserving_at_zero()` truthfully and pass `WriterContractMixin`. If you need a new
cap kind (e.g. categorical tilt), implement kind `cap` with `apply` and one output port of the
matching forecast kind and coordinate schema.

## 8. Tasks and datasets

Task: implement `PredictionTask` (`allowed_roles`, `permitted_view`, `prediction_space`,
`make_query`, `score`, `loss_terms(pred, y, mask, xp)`). The loss must be traceable in `xp` and
return `(numerator, denominator)`.

Dataset: implement `ScenarioProvider` (`observation_fields` with truthful roles, `iter_requests`
that yields role-tagged bundles + queries and *no targets*, `outcome(request_id, query_id)` with
`available_at` after the origin and coordinate masks, `grouping_metadata`). Optionally
`matured_episodes(before_origin)` for memory seeding. Run `ScenarioContractMixin`.

## 9. Assemble the instance in configuration

```yaml
schema_version: ness.experiment/3
protocol_id: my_instance_v1
scenario: {plugin: my_dataset, config: {...}}
tasks: [{id: future_values, plugin: quantile_forecast_task, config: {horizons: 24, channels: [...], levels: [0.1,0.5,0.9]}}]
protocol: {train_split: train, test_split: test, batch_size: 16, updates: 200, max_test_requests: 500}
arms:
  native:      {seed: 0, composition: {nodes: [<substrate>], output: {task: future_values, from: substrate://base/forecast/quantiles}}}
  context_cap: {seed: 0, learning: {rule: bp_direct, optimizer: {kind: adam, lr: 0.001}}, composition: {...}}
  symbolic_cap: ...
```

Then: `ness validate my.yaml` (prints resolved wiring per arm and fails closed on any contract
violation), `ness run my.yaml --out runs/x`, `ness inspect runs/x/checkpoints <manifest>`.

## 9b. Run, record, report

Put the whole study in one experiment file (`scenario`, `tasks`, `protocol` with `seeds`,
`arms`, `substitutions`) and run `ness run cfg.yaml --out runs/<study>/<id>`. You get raw
artifacts (predictions, evidence traces, training history, timing, checkpoint-restore diffs,
substitution matrix, manifests, environment) and `ness report <run>` turns them into figures and
a Markdown report without touching models. Substitutions are complete arms built by
configuration; if one needs a core edit, that is an interface bug to fix in core, not in the
example. See `examples/vertical_slice_timeseries/configs/toy_vertical_slice.yaml` and
`reports/toy_vertical_slice_report.tex` for the reference study.

## 10. Before claiming anything scientific

* Keep the native arm and a same-fact context-only cap as controls; report paired differences
  (the runner does) with the right grouping unit (`request.group`).
* Record the dependency lock (`ness audit`); pin your model revision/digest in module `meta`.
* Run your plugin's contract tests plus the platform suite; the manifest of every reported
  prediction must be reproducible from the checkpoint (`NessSystem.from_snapshot`).
* Anything you could not verify (hidden-state access, differentiable path, license) stays
  `unverified`/`unsupported` in capabilities and blocks launch.

## 11. Where the seams for the next platform capabilities are

| capability | seam | what to add |
|---|---|---|
| FabricPC 0.7+ | `ness/backends/fabricpc/compat/` | a `v0_7.py` module with the same function surface, a `SUPPORTED_FAMILIES` row and matrix entries after `docs/FABRICPC_UPGRADE.md` |
| new FabricPC node families (conv, transformer) in NESS caps | `ness/backends/fabricpc/module.py` | a cap whose `structure()` builds the graph; keep prediction target-free and record `algorithm_spec()` |
| nudge / EP learning rule | `ness/backends/fabricpc/rules.py` + `learning/profiles.py` | NESS-controlled positive/negative phases using `initialize_graph_state`/`run_inference`; audit against BP-through-unroll |
| recurrent workspace at composition level | `composition.InferenceWorkspaceSpec` (already parsed/validated) | a solver for declared regions; today recurrence lives inside a FabricPC node (`fabricpc_spc_recurrent`) |
| model parallelism / multi-host | `runtimes/devices.py` + backend capabilities | a mesh spec with a `model` axis and verified probes; keep `data_parallel`, `model_parallel`, `multi_host` separate |
| Hyperon memory | `memory.MemoryStore` | adapter + snapshot export/import; `memory_store` plugin |
| caches | `runtime/executor.py` (`GraphExecutor.run`) | cache keyed by the PDF's key family; sampled uncached audit |
| separate task views | `NessSystem._policy_for` | one solve per distinct information boundary |
