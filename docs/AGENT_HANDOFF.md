# Handoff: implementing your own NESS instance with your own models

Audience: an AI coding agent (or engineer) who will plug **their own models, datasets, reasoners,
memory backend and caps** into NESS as a new *instance*, and run controlled experiments on it.

Read `docs/QUICKSTART.md` first (concepts, the experiment file, the shipped plugins, `ness graph`)
and `docs/ARCHITECTURE.md` §2 to §7 for the interfaces. This document is the operational
contract: **what you may create, what you must never modify, what you must add so that your
plugins run, and how you prove you stayed inside the lines.**

**Contents**

0. [The one rule](#0-the-one-rule)
1. [The boundary: files you may and may not touch](#1-the-boundary-files-you-may-and-may-not-touch)
2. [A setup that makes the boundary physical](#2-a-setup-that-makes-the-boundary-physical)
3. [Your package: everything a plugin needs in order to run](#3-your-package-everything-a-plugin-needs-in-order-to-run)
4. [Non-negotiables inside every plugin](#4-non-negotiables-inside-every-plugin)
5. [Per-layer contracts](#5-per-layer-contracts)
6. [Assemble, validate, draw, run, report](#6-assemble-validate-draw-run-report)
7. [Definition of done](#7-definition-of-done)
8. [When you really need a core change](#8-when-you-really-need-a-core-change)

## 0. The one rule

**NESS core is a dependency you install, not code you edit.** An instance of NESS is:

* your **plugin package** (a separate Python project registering plugins through the
  `ness.plugins` entry point),
* your **experiment configurations** (YAML files),
* your **tests**, **data**, **run directories** and **reports**.

Nothing in the platform (`ness/`) changes when an instance is created. If something in your
plan seems to require a change inside `ness/`, it is either (a) something configuration or a
plugin can already express (check `docs/QUICKSTART.md` §7 and §8), or (b) a core proposal
(section 8). It is never a local patch.

## 1. The boundary: files you may and may not touch

| you may create and modify freely | you must never modify | you may only propose (section 8) |
|---|---|---|
| your package: `my_ness_instance/pyproject.toml`, `src/my_ness_instance/**`, `tests/**`, `configs/**`, `README.md`, `data/**`, `scripts/**` (fit scripts, data preparation) | `src/ness/**` (every core package: `contracts`, `plugin_api`, `composition`, `runtimes`, `runtime`, `learning`, `inference`, `symbolic`, `memory`, `probabilistic`, `tasks`, `scenarios`, `checkpoint`, `experiments`, `visualize`, `observability`, `config`, `cli`, `backends/**`, and `reference_plugins/**`) | new module kinds, selector schemes, boundary transforms or merge operators, port kinds, evidence types |
| your run directories (`runs/**`), your generated reports and diagrams | the platform's `pyproject.toml`, `tests/**`, `compat/**`, `.github/**`, `tools/**`, `examples/**`, `docs/**`, `reports/**` | new learning rules or inference profiles in the platform vocabularies (`learning/profiles.py`, `inference/`) |
| an `examples/<your_instance>/` directory inside a NESS clone **only** as a pull request contributing an example (never as your working copy) | the installed package under `site-packages/ness/**` | new numerical backends (`backends/<name>/`), a new FabricPC compat module |
| | the baseline `src/ness/_core_hash.py` | changes to the compiler, executor, transaction, credit map, bootstrap, checkpoint format, CLI, or any contract |

The reference plugins (`ness/reference_plugins/**`) are ordinary plugins that ship with the
platform. Do not edit them either: **copy** the one you want into your package under a new
`plugin_id`, and change the copy.

Signs that you are about to cross the line, all of which are forbidden:

* editing any file under `src/ness/` or `site-packages/ness/`;
* monkeypatching anything in `ness.*` at import time or in a test fixture;
* subclassing a core class to *override* compiler, executor, transaction, credit-map,
  bootstrap or checkpoint behaviour (subclassing `BaseModule` or a reference plugin to build
  *your* plugin is fine);
* catching `ContractViolation`, `AccessPolicyViolation`, `GradientBoundaryError`,
  `UnsupportedCapability` or `PluginDependencyMissing` in order to continue;
* reading a field with role `outcome_only` or `evaluation_oracle`, or a memory namespace the
  task does not permit, through any side channel (file, global, cache);
* extending `MODULE_KINDS`, `SCHEMES`, `BOUNDARY_KINDS`, `MERGE_OPS`, `INFERENCE_PROFILES` or
  `LEARNING_PROFILES` from outside core;
* calling `jax.devices()`, `fabricpc.setup_jax()` or mutating `XLA_FLAGS` / `JAX_PLATFORMS`
  yourself (the platform's bootstrap owns that);
* copying core modules into your package and "adjusting" them.

## 2. A setup that makes the boundary physical

Install NESS as a *non-editable* dependency in a fresh environment and keep your instance in
its own repository. With that layout there is no core source tree in your working copy to edit.

```bash
python -m venv .venv && . .venv/bin/activate
pip install "ness[jax,report] @ git+https://github.com/guilamacie/NESS.git@v0.2.1"   # add fabricpc if you need it
mkdir my_ness_instance && cd my_ness_instance && git init
```

Verify, and keep verifying in your CI:

```bash
ness audit --strict     # exit 0 only if the installed core matches the release baseline hash
pip show ness           # Location: must be inside site-packages, not your working tree; no "Editable project location"
ness plugins            # after `pip install -e .` of your package: your ids appear with provided_by = your package
```

`ness audit` compares a content hash of every core source file with the baseline recorded at
release (`ness.integrity`). Any modification of core, anywhere in the environment, turns it red;
`--strict` makes it a non-zero exit so a pipeline fails. Put it in your instance's test command.

Do not `git clone` NESS into your instance's environment and `pip install -e` it. That is the
workflow for people changing core, and it makes core edits silent. If you must read core code,
read it in `site-packages` or on GitHub.

## 3. Your package: everything a plugin needs in order to run

`examples/external_plugin_example/` is a complete, tested template. Copy its layout:

```
my_ness_instance/
  pyproject.toml            # name, dependencies, entry points
  README.md                 # install, data provenance, weights digests and licenses, how to run
  src/my_ness_instance/
    __init__.py
    descriptors.py          # PluginDescriptor objects only; no heavy imports here
    substrate.py            # your lower model(s)
    upper.py                # your trainable upper module (runtime "jax" for BP)
    semantic.py, program_primitives.py, reasoner.py, memory.py, cap.py, writer.py, task.py, scenario.py   # as needed
  configs/
    my_study.yaml           # the experiment file(s): scenario, tasks, protocol, arms, substitutions
  scripts/
    fit_emissions.py        # anything that fits parameters you then paste into configs (records the fit population)
    prepare_data.py
  tests/
    test_contracts.py       # ness.plugin_api.testkit mixins for every plugin
    test_configs.py         # `ness validate` of every config; `ness audit --strict`
  data/                     # or paths in config; never inside the ness package
```

What each piece must contain for the plugin to be discovered and to run:

**`pyproject.toml`**

```toml
[project]
name = "my-ness-instance"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["ness>=0.2,<0.3", "numpy>=1.26"]      # pin the NESS minor you developed against

[project.optional-dependencies]
torch = ["torch>=2.3", "transformers>=4.44"]          # heavy model libraries as extras, never as hard requirements
numpyro = ["numpyro>=0.15"]

[project.entry-points."ness.plugins"]                 # one line per plugin: id = "module:DESCRIPTOR"
my_timesfm_substrate = "my_ness_instance.descriptors:MY_TIMESFM_SUBSTRATE"
my_gru_upper_module = "my_ness_instance.descriptors:MY_GRU_UPPER"
my_energy_market_dataset = "my_ness_instance.descriptors:MY_DATASET"

[tool.setuptools.packages.find]
where = ["src"]
```

**`descriptors.py`**: `PluginDescriptor(plugin_id, kind, version, "my_ness_instance.substrate:MyTimesFM", requires=("timesfm", "torch"), summary="...")`.
`requires` lists the *import names* of heavy dependencies; the registry checks importability
before importing your module, so a configuration that does not use the plugin still validates on
a machine without those libraries, and `ness plugins` flags what is missing instead of crashing.
Import nothing heavy at the top of `descriptors.py`; import heavy libraries inside the plugin
module, ideally inside `initialize` / `forward`.

**The plugin class** (one per module): `plugin_id`, `plugin_version`, `module_kind`, `runtime`
(`numpy`, `host`, `jax`, `fabricpc`, or the truthful name of another runtime such as `torch`,
which is then a frozen constant to every learning rule), `state_schema_id`, `validate_config`,
`describe()`, `initialize(rng)`, `forward(inputs, state, ctx)`; trainable modules add
`apply(params, dense_inputs, state, xp)`; stateful modules rely on `BaseModule`'s
`snapshot_state` / `restore_state` or implement `migrate_state` when the schema changes. Anything
that affects outputs (checkpoint digest, preprocessing version, quantile grid, dropout mode)
goes in config or `ModuleState.meta` so it enters the manifest.

**Tests**: subclass the test-kit mixins for every plugin (`ModuleContractMixin`,
`FrozenModuleMixin`, `DifferentiableModuleMixin`, `ReasonerContractMixin`,
`ScenarioContractMixin`, `MemoryStoreContractMixin`, `WriterContractMixin`); validate every
config with `ness validate` in a test; run `ness audit --strict`; keep a checkpoint round-trip
test for each stateful plugin (the mixins include one).

**Configs**: the experiment file(s) that name your plugins (`docs/QUICKSTART.md` §5 and §6).
Fitted parameters (reasoner emissions, thresholds, normalisation constants) come from a script in
`scripts/` that fits them on the training split only and writes them into the config with a
comment naming the fit population.

**Data**: a `scenario` plugin that reads your files from paths given in its config; the plugin
declares every field with a truthful role, yields target-free requests, and reveals outcomes
through `outcome()`. Data never lives inside the `ness` package and never bypasses
`PredictionTask.permitted_view`.

**README**: install command (with extras), how to obtain the data and weights (digests,
licenses, revisions), the commands to validate, draw, run and report, and what is verified
versus unverified in your capabilities.

Sanity check after `pip install -e .` of your package: `ness plugins` lists your ids with
`provided_by` equal to your package; `ness validate configs/my_study.yaml` prints the resolved
wiring; `ness graph configs/my_study.yaml --out graphs/` draws it.

## 4. Non-negotiables inside every plugin

1. **Target-free prediction.** Nothing with role `outcome_only` or `evaluation_oracle` may be
   read by a node. The compiler enforces it on every edge and the transaction enforces it on the
   permitted bundle; do not add code paths that bypass `PredictionTask.permitted_view`.
2. **Typed evidence, honest semantics.** A forecast implements only the operations it supports.
   A reasoner's `HypothesisEvidence.weight_semantics` must be true (`exact_posterior` only for
   exact inference over a complete latent). Unknown values carry `knownness=False`, never 0.
3. **Memory is pinned.** A module reads `ctx.memory_views[store]`; it never receives a store
   handle to mutate inside a prediction. Learner appends go to a branch and are published
   between updates.
4. **Runtimes do not exchange gradients.** Declare `runtime` truthfully. A module in another
   runtime is a constant to the JAX region and to FabricPC workspaces; a rule may only own
   parameters in its own runtime (`GradientBoundaryError` otherwise). A cross-runtime derivative
   bridge is a *new verified core capability* (section 8), not an array copy.
5. **One bootstrap owner.** Never call `jax.devices()`, `fabricpc.setup_jax()` or mutate
   `XLA_FLAGS` in a module or at import. `NessSystem.build`, the runner and the CLI call
   `ness.runtimes.bootstrap.bootstrap` first; a plugin needing the backend outside a system calls
   `ensure_bootstrapped("jax"|"fabricpc")`. Device counts come from the experiment's
   `runtime.devices` spec, never from code.
6. **Inference profile and learning rule are separate declarations.** Do not fold settling or
   PC rules into a module's `forward`.
7. **No silent fallbacks.** If a capability is missing, raise `UnsupportedCapability` (or a
   `PluginDependencyMissing`). Never substitute another algorithm.
8. **Whole-system identity.** Anything that changes predictions must be in module state
   (`params` / `buffers` / `meta`) or config, so it lands in the manifest and checkpoint.

## 5. Per-layer contracts

### 5.1 Wrap a real lower model as a substrate

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
declare normalisation behaviour in `diagnostics` / capabilities; put anything that affects
outputs (dropout off, preprocessing version, quantile grid) in `meta` / config so it is hashed.
Run `FrozenModuleMixin` on it. Cache heavy outputs *outside* the module keyed by
`(request canonical hash, config_hash)` if you need speed; the cached value must be identical to
the uncached one.

### 5.2 Choose the numerical backend for trainable parts

| you want | use | learning rules |
|---|---|---|
| generic differentiable module trained by BP | NESS JAX backend (`runtime = "jax"`, `apply(...)`) | `bp_direct` (per-item, `batched`, or `data_parallel`) |
| predictive-coding / ePC / recurrent workspace | FabricPC backend: reuse `fabricpc_residual_cap` (dense PC graph) or write a cap that builds its own FabricPC graph through the compat adapter | `fabricpc_pc_local`, `fabricpc_bp_through_inference` |
| both in one arm | jax upper + FabricPC cap with `learning.credit.overrides` | hybrid: each rule owns its runtime's groups |

Declare `runtime:` in the experiment (`backend: auto|numpy|jax|fabricpc`, `platform`,
`devices: {platform, selection|ids, count, fallback}`, `x64`, `profile: development|scientific`).
`profile: scientific` refuses unverified FabricPC versions and a late `setup_jax`. Read
`docs/FABRICPC_BACKEND.md` before writing a FabricPC-backed module: prediction must clamp
inputs only; the PC clamp belongs to the learning rule; record a complete `algorithm_spec()`.

### 5.3 Wrap a trainable upper module

For BP through NESS your module must run in the differentiable reference runtime (`jax`) and
implement `apply(params, dense_inputs, state, xp)` as a pure function (see
`reference_plugins/upper_modules.py`). Flax/Equinox/Haiku modules work: convert their parameter
pytree to `{group: {name: array}}` in `initialize` and call them inside `apply`. Declare
`ParameterGroupSpec(name, "trainable", "jax", True)` and mark output ports `DIFFERENTIABLE`.
Run `DifferentiableModuleMixin` (checks apply == forward and gradients against finite differences).

If your upper module is PyTorch and you want to *train* it inside NESS: there is no verified
torch runtime or bridge. Your options inside an instance are (a) port it to JAX, or (b) freeze
it (`STOP`) and train only JAX/FabricPC caps on its detached arrays (the intended architecture
for TimesFM/ViT providers). A torch backend would be a core capability with its own region,
rule, bootstrap hook, capability probes, tests and an ADR: propose it (section 8); do not build
it in your package and never fake it with array copies.

### 5.4 Semantic / neuro-symbolic modules reading several earlier ports

Declare one input port per source you want (`raw`, `neural`, `base_context`, ...), with
`accepts_semantic_types`. In config, wire each to any *permitted* earlier port:
`observation://...`, `substrate://base/state/layer_8`, `module://upper/hidden`. Return
`FeatureEvidence` / `HypothesisEvidence` / `BindingEvidence` with `knownness`, and set
`ModuleOutputs.used_inputs` to the ports you actually consumed so provenance is exact. Use
boundary transforms in config (`last_step`, `mean_pool`, `flatten`) instead of reshaping inside
the module when the reduction is a wiring choice rather than a scientific one.

### 5.5 Typed programs and new primitives

Programs are configuration (`typed_program` node with a `ness.program/1` IR). A new primitive
(a typed numeric operator your programs need) is a `PrimitiveSpec` in your package, exposed
through the `ness.primitives` entry-point group:

```toml
[project.entry-points."ness.primitives"]
window_max = "my_ness_instance.primitives:WINDOW_MAX"     # a PrimitiveSpec, or an iterable of them
```

`ness.symbolic.operators.default_registry()` loads it, so any program in any configuration can
`call` it; a name clash with a built-in fails closed. The primitive must declare its typed
signature, be pure, and return `PrimitiveResult(value, Knownness.KNOWN)` or
`PrimitiveResult(placeholder, Knownness.MISSING, reason)` when its preconditions fail (a missing
value is never treated as a fact downstream). `examples/external_plugin_example`
contributes `window_max` this way. If the interpreter itself lacks an *operation* (a new IR op),
that is a core proposal.

### 5.6 Probabilistic reasoners (Pyro / NumPyro / custom)

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

### 5.7 Memory backends (Hyperon or other)

Implement the `MemoryStore` protocol (`describe, open_view, query, fork, append, seal`, plus
`export_snapshot` / `import_snapshot` for checkpoints) as a plugin of kind `memory_store`.
`fork` must isolate (not alias); `append` must honour `expected_head` and idempotent event ids;
`open_view` must filter by `AvailabilityCut` and namespaces **before** any ranking; `seal` must
be content-addressed. Run `MemoryStoreContractMixin`. Build a `memory_query` module that reads
only `ctx.memory_views[store]`. Atomspace-backed: rebuild a private read-only space from a sealed
snapshot; `SpaceRef.copy()` is not a fork. Attach the store to an arm through its `memory:`
section; no core change is involved.

### 5.8 Caps, writers, consumers

Prefer reusing `residual_point_cap` / `residual_quantile_cap` with a new writer or consumer
plugin (both are pure functions over `xp`, selected by the cap's `writer:` / `consumer:` config
keys). A new writer must state `is_baseline_preserving_at_zero()` truthfully and pass
`WriterContractMixin`. If you need a new cap kind (e.g. categorical tilt), implement kind `cap`
with `apply` and one output port of the matching forecast kind and coordinate schema.

### 5.9 Tasks and datasets

Task: implement `PredictionTask` (`allowed_roles`, `permitted_view`, `prediction_space`,
`make_query`, `score`, `loss_terms(pred, y, mask, xp)`). The loss must be traceable in `xp` and
return `(numerator, denominator)`.

Dataset: implement `ScenarioProvider` (`observation_fields` with truthful roles, `iter_requests`
that yields role-tagged bundles + queries and *no targets*, `outcome(request_id, query_id)` with
`available_at` after the origin and coordinate masks, `grouping_metadata`). Optionally
`matured_episodes(before_origin, namespace)` for memory seeding. Run `ScenarioContractMixin`.

## 6. Assemble, validate, draw, run, report

```yaml
schema_version: ness.experiment/3
protocol_id: my_instance_v1
runtime: {backend: auto, platform: gpu, devices: {platform: gpu, selection: all, fallback: cpu}}
scenario: {plugin: my_energy_market_dataset, config: {path: data/market.parquet, ...}}
tasks: [{id: day_ahead, plugin: quantile_forecast_task, config: {horizons: 24, channels: [load], levels: [0.05, 0.5, 0.95]}}]
protocol: {train_split: train, test_split: test, batch_size: 32, updates: 500, seeds: [0, 1, 2], restore_check_requests: 32}
arms:
  native:       {seed: 0, composition: {nodes: [<substrate>], output: {task: day_ahead, from: substrate://fm/forecast/quantiles}}}
  context_cap:  {seed: 0, learning: {rule: bp_direct, optimizer: {kind: adam, lr: 0.001}}, composition: {...}}   # same-fact control
  symbolic_cap: {...}
substitutions: [...]
```

`examples/quickstart/C_bring_your_own_models.yaml` is a complete template of exactly this shape
with placeholder `my_*` plugin ids. Then:

```bash
ness validate configs/my_study.yaml                     # every arm compiles; resolved wiring printed; fails closed
ness graph    configs/my_study.yaml --out graphs/       # composition per arm, sandwich view, arms matrix, data access
ness run      configs/my_study.yaml --out runs/my_study/<id>
ness evaluate runs/my_study/<id>
ness report   runs/my_study/<id>
ness inspect  runs/my_study/<id>/checkpoints <manifest-id>
```

Put the whole study in one experiment file (`scenario`, `tasks`, `protocol` with `seeds`,
`arms`, `substitutions`). The runner records raw artifacts (predictions, evidence traces,
training history, timing, checkpoint-restore differences, the substitution matrix, manifests,
environment, core source hash) and `ness report` turns them into figures and a Markdown report
without touching models. Substitutions are complete arms built by configuration; if one of
yours needs a core edit, that is an interface gap to report (section 8), not something to patch.

## 7. Definition of done

Your instance is done when all of the following hold, and your CI checks them:

* `ness audit --strict` exits 0 in the environment you run experiments in, and `pip show ness`
  shows an installed (non-editable) package.
* `git diff` of any NESS checkout you touched shows no change under `src/ness/`, `tests/`,
  `pyproject.toml`; ideally you have no NESS checkout at all.
* Every plugin has contract tests built on the test kit, including a checkpoint round trip, and
  they pass.
* Every config passes `ness validate`; `ness graph` draws what you intended; `ness plugins`
  shows your ids provided by your package with no missing dependency in the target environment.
* The study has a native control arm and a same-fact context-only control; the paired
  differences the runner reports are what you interpret.
* Every arm and seed restored from its checkpoint reproduces the recorded predictions exactly
  (the runner's restore check), and the causal check passes.
* Capabilities you could not verify (hidden-state access, a differentiable path, a license) are
  declared `unverified` / `unsupported` and therefore block launch; nothing is labelled verified
  by assumption.
* The dependency lock (`ness audit`), model revisions and weight digests are recorded in module
  `meta` and in your README.
* Nothing in your package monkeypatches, subclasses-to-override, or catches-to-continue the
  core behaviours listed in section 1.

## 8. When you really need a core change

Some needs cannot be met by a plugin or a configuration. They are **core proposals**, made to
the NESS maintainers as an issue (or a pull request against the NESS repository that follows
`CONTRIBUTING.md`), never as a change inside your instance:

| need | why it is core | what to include in the proposal |
|---|---|---|
| a new module kind, selector scheme, boundary transform, merge operator, port kind or evidence type | vocabulary the compiler and every plugin rely on | the contract, shape and gradient rules, an ADR draft, tests |
| a new learning rule or inference profile name | the platform vocabulary with verified/experimental/unsupported status | the algorithm spec, parity tests against an existing rule, status |
| a new numerical backend (torch, ...) or a FabricPC compat module for a new version | bootstrap, capability probes, version routing, credit map | region, rule, bootstrap hook, probes, compatibility-matrix entry, import-order tests |
| a composition-level recurrent workspace solver, caches, separate task views | executor and transaction semantics | the seam is `composition.InferenceWorkspaceSpec`, `runtime/executor.py`, `NessSystem._policy_for` |
| a change to manifests, checkpoints, artifact layout or the CLI | whole-system identity and reproducibility | migration path and tests |

While a proposal is pending, keep your instance honest: declare the missing capability
`unsupported` in your plugin's capabilities, keep the arm out of your study or mark it
experimental, and never emulate the capability with a workaround that hides the gap.
