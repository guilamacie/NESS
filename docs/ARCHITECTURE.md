# NESS platform: architecture, interfaces and how to use it

## 1. Summary

NESS is a research platform for *predictive systems built from typed evidence*: one or more
lower "substrates" (frozen or trainable models) produce representations and native forecasts;
semantic adapters, typed symbolic programs, probabilistic reasoners and pinned memory queries
turn those into explicit, provenance-carrying evidence; a trainable upper module and a "cap"
(consumer + baseline-preserving writer) turn everything into a forecast in a declared
prediction space; a task scores it only after the outcome is revealed.

This codebase implements the NESS/FabricPC cross-domain design specification (an external document, cited in these docs as "PDF §n") and the
collaborator-facing extensions of its coding-agent handoff addendum (external, cited as "addendum §n"; neither document is distributed in this repository, both are available from the project lead) as a **plugin
platform**. The scientific question is visible in configuration and manifests: which
substrates ran, which earlier ports each module read, which symbolic and probabilistic
computations executed, which memory snapshot was pinned, which cap wrote the output, which
learning rule owned which parameters, and which checkpoint produced every prediction.

What is delivered and verified in this cycle (see `IMPLEMENTATION_STATUS.md` for the
implemented / verified / experimental / design-only split):

* a typed **contracts** package (evidence, forecasts, coordinates, availability roles,
  requests/outcomes, ports, capabilities, state snapshots, manifests) with no heavyweight
  dependencies;
* a **macro composition graph** (`ness.composition/1`): named typed ports, `scheme://node/port`
  source selectors, a small fixed vocabulary of boundary transforms and merges, a compiler that
  resolves and validates every edge (DAG-ness, access policy, shapes, coordinates, gradient
  boundaries) and records the resolved graph;
* a **plugin API** with entry-point discovery, lazy imports, declared optional dependencies, and
  a contributor **test kit** of contract-test mixins;
* host-side subsystems: a typed **program IR** and reference interpreter, an immutable
  **memory snapshot store** with fork/append/seal and pinned causal views, an **exact finite
  probabilistic reasoner** plus an importance-sampling comparator;
* two optional numerical backends behind one contract layer: the **NESS JAX backend**
  (`ness.backends.jax`: differentiable region, `bp_direct`, NESS-owned batched/sharded
  gradients) and the **FabricPC backend** (`ness.backends.fabricpc`: version-routed adapter for
  FabricPC 0.6.x, a FabricPC predictive-coding workspace cap, PC-local and BP-through-inference
  learning rules, probe-based capabilities); a credit map that refuses cross-runtime derivative
  paths and supports hybrid ownership; numpy optimizers with restorable state;
* a **single runtime bootstrap owner** (`ness.runtimes.bootstrap`) with a device specification
  (requested / visible / selected / realized), `ness doctor`, and subprocess import-order tests;
* the **prediction transaction** (pin → validate → execute → record → reveal → learn), whole
  system **manifests and checkpoints** (content-addressed blobs, staged atomic publication,
  restore that re-instantiates plugins from the manifest and treats state as data);
* an **experiment runner** with paired arms on identical request streams, a **CLI**
  (`validate / run / evaluate / inspect / audit / plugins`), the synthetic **vertical slice**
  (six arms) and an **external example plugin package** that registers without editing core code.

Not delivered (declared, fail explicitly): Hyperon-backed memory, TimesFM or vision/text
providers, composition-level recurrent `InferenceWorkspace` regions (a recurrent FabricPC
workspace *node* exists, experimental), centered-nudge / equilibrium-propagation rules, model
parallelism, multi-host execution, the reference44 numerical port. Their names exist in the
capability vocabulary with status `unsupported`/`experimental`, so selecting them is a launch
error or an explicit opt-in, never a fallback.

## 2. Package layout and dependency direction

```
src/ness/
  contracts/        typed data contracts (numpy only)                         <- everything
  plugin_api/       MacroModule/PortValue/RuntimeContext, BaseModule, registry, testkit
  composition/      CompositionGraphSpec, selectors, YAML/JSON parser, compiler
  runtimes/         ops (boundary/merge vocabulary, xp-parametrised); bootstrap (THE runtime owner);
                    devices (DeviceSpec/resolution); doctor; jax_runtime (compat shim)
  backends/jax/     NESS JAX backend: DifferentiableRegion, BPDirectRule, batched/sharded objective
  backends/fabricpc/ FabricPC backend: version routing, translate (AlgorithmSpec), compat/v0_6 (only
                    FabricPC importer), module (workspace cap), rules, capabilities (probes), checkpoint
  runtime/          GraphExecutor, input assembly, NessSystem, PredictionTransaction
  learning/         CreditMap, optimizers, learning-profile vocabulary
  inference/        inference-profile vocabulary
  symbolic/         ProgramIR, primitives, RelationWorld, ReferenceInterpreter
  memory/           MemoryStore contract, InMemorySnapshotStore, QueryPlan/QueryResult
  probabilistic/    FiniteLatentModel, Exact / ImportanceSampling reasoners
  scenarios/        ScenarioProvider helpers, ToyTemporalDataset
  tasks/            access policies, PointForecastTask, QuantileForecastTask
  checkpoint/       LocalBlobStore, CheckpointStore, SystemSnapshot
  experiments/      ExperimentSpec parsing, runner and reports
  observability/    PredictionRecord, ExperienceEvent
  reference_plugins/ toy substrates, upper modules, semantic adapter, program node, memory query, caps, writers/consumers, descriptors
  config/           YAML/JSON loading
  cli/              `ness` command
```

```
                    NESS contracts
                          |
           +--------------+--------------+
           |                             |
      JAX backend                 FabricPC backend
           |                             |
          JAX                        FabricPC
                                         |
                                        JAX
```

Rules enforced by construction and tests:

* `contracts` imports neither FabricPC, Hyperon, JAX nor a modality framework. Only
  `ness.backends.jax.*` and `ness.backends.fabricpc.compat.*` import JAX/FabricPC, and only
  plugins that declare `runtime = "jax"` / `"fabricpc"` import those modules. `import ness`,
  `ness plugins`, `ness validate --schema-only`, `ness audit` and `ness doctor --backend numpy`
  never import or initialise JAX (subprocess tests: `tests/test_bootstrap_import_order.py`).
* Exactly one place initialises a numerical backend: `ness.runtimes.bootstrap.bootstrap`,
  called by `NessSystem.build`, the runner and the CLI before any module state exists. It calls
  FabricPC's `setup_jax` before the first JAX computation, respects the user's `JAX_PLATFORMS`
  and `XLA_FLAGS`, resolves the experiment's `runtime.devices` spec and records a report.
* The core knows *kinds, ports and capabilities*, not domains. There is no `if modality ==`
  anywhere; test `T17` runs a mock provider with an unseen modality through the unchanged
  executor.
* Backends depend on contracts, never the reverse. Reference plugins are registered through the
  same entry-point group (`ness.plugins`) that third-party packages use.

## 3. Core concepts (contracts)

| concept | class | essential rule |
|---|---|---|
| observation | `Observation`, `ObservationBundle`, `ObservationField` | role-tagged (`observed`, `known_future`, `derived`, `outcome_only`, `evaluation_oracle`), named axes, `available_at` |
| information boundary | `AccessPolicy`, `AvailabilityCut` | a policy can never allow outcome/oracle roles; `bundle.select(policy)` is the only way to obtain a prediction-time view |
| forecast | `PointForecast`, `QuantileForecast`, `CategoricalForecast`, `SampleForecast` + `ForecastCapabilities` | each implements only the operations it truly supports; anything else raises `UnsupportedCapability` |
| prediction space | `PredictionSpace` | validates forecast type, coordinate schema and quantile grid before scoring or combination |
| evidence | `FeatureEvidence` (value + knownness), `HypothesisEvidence` (alternatives + weights + weight semantics), `PredictiveEvidence`, `RetrievalEvidence`, `ExecutionTrace`, `ConstraintEvidence`, `BindingEvidence`, `QueryStatus` | distinct types; padding and absence never become facts |
| evidence graph | `EvidenceGraph` | immutable, append-only via `with_fragment`, dependency-checked, versioned by content hash |
| request / outcome | `PredictionRequest`, `TaskQuery`, `TrainingOutcome` | a request never contains its withheld target; outcomes carry coordinate-wise masks and release sequence |
| ports | `PortSpec`, `PortSchema`, `ModuleDescriptor`, `ParameterGroupSpec` | payload kind, shape (None = variable, () = any), semantic type, availability role, differentiability, coordinate schema |
| capabilities | `SubstrateCapabilities`, `ProbabilisticReasonerCapabilities`, `BackendCapabilities`, `Capability{Set,Status}` | `verified / unsupported / experimental`; only `verified` may be launched |
| state | `ModuleState` (params by group, buffers, meta), `StateSnapshot` (arrays + JSON, no object dtype) | restore is fail-closed on plugin id, major version, schema id and config hash; `migrate_state` is the only escape |
| identity | `PredictorManifest`, `ComponentRef` | every scored prediction resolves to one manifest id (a content hash) naming every component, its version, config hash and state hash |

## 4. The macro composition graph

A composition is a DAG of *scientifically meaningful* macro modules. Internal tensor
operations stay inside module implementations; the graph is not a tensor IR.

```yaml
nodes:
  - id: base_ts
    plugin: toy_frozen_transformer
    config: {...}
    inputs: {history: observation://target_history}
  - id: upper
    plugin: tiny_upper_transformer
    inputs:
      main:
        merge: gated_add                      # explicit merge when >1 source
        sources:
          - substrate://base_ts/state/final
          - {from: substrate://base_ts/state/early, boundary: [{kind: linear, out_dim: 16}]}
        post: [{kind: layer_norm}]            # transforms after the merge
  - id: cap
    plugin: residual_point_cap
    inputs: {baseline: substrate://base_ts/forecast/point, neural: module://upper/hidden}
output: {task: future_value, from: module://cap/forecast}
```

**Source selectors** are `scheme://node/port`. The scheme must agree with the producer's kind:
`observation://field`, `substrate://`, `module://` (upper modules and caps), `semantic://`,
`program://`, `reasoner://`, `memory://`. Free-form strings are rejected.

**Boundary transforms**: `identity`, `linear` (learned), `layer_norm`/`rms_norm` (affine
learned by default), `standardize` (declared constants), `mean_pool`, `last_step`, `flatten`.
**Merges**: `concat`, `add`, `gated_add` (learned gate), `weighted_sum` (learned softmax
weights), `select`, `attention_pool`, `cross_attention`. Shape rules run at compile time; learned
transforms are only legal into a differentiable destination; `add`-like merges require identical
shapes and compatible coordinates unless an explicit projection is present. Edge parameters are
their own manifest component (`composition:boundaries`).

**Compiler guarantees** (`ness.composition.compile_graph`): every selector resolves to a typed
port of the right kind; every required input is wired; an accidental cycle is rejected while a
cycle inside a declared `recurrent_regions` entry is accepted (execution of such a region raises
`UnsupportedCapability` until a solver is verified); every edge is a read under the task access
policy (`observation://target_future` is an `AccessPolicyViolation`); the differentiable region
and every stop edge are computed and recorded; the output port kind and coordinates match the
task's prediction space; `composition_hash` covers the spec, every descriptor and the
observation schema.

**Execution** (`GraphExecutor`): nodes run in topological order under one `RuntimeContext`.
A single-source, transform-free input is passed through *as typed evidence* (so a cap sees a
`PointForecast`, a reasoner sees `FeatureEvidence` with knownness). Any merge/transform lowers to
a dense array via the documented `dense()` rule, is recorded as derived evidence, and keeps a
knownness mask only for a plain `concat` of feature evidence. Every output becomes a
`PortValue` with provenance (producer, dependencies, roles, availability, resolved selectors).

## 5. Runtimes, learning and inference

*Runtime is a declared per-module property.* Frozen providers and host-side modules run in
`numpy`/`host`; trainable numeric modules run in the NESS JAX backend (`jax`) and implement a
pure `apply(params, dense_inputs, state, xp)` next to `forward`; FabricPC workspaces run in
`fabricpc` (see `docs/FABRICPC_BACKEND.md`). A learning rule declares `required_runtime()` and
may only own parameter groups of modules in that runtime; several rules can co-own a graph
(hybrid credit, PDF §11.6), each computing gradients from the same pre-update snapshot. Both merges and
module code are written against an array namespace `xp`, so the eager path (numpy) and the
traced path (jax.numpy) execute the same code (`test_T05_traced_region_matches_eager_forward`).

*Gradients never cross runtimes.* `DifferentiableRegion` re-runs only jax nodes as a function of
the trainable parameters; every input from a host node, frozen provider or non-differentiable
port is fed as a constant recorded in the eager pass. `build_credit_map` raises
`GradientBoundaryError` if a differentiable rule is asked to own a host-runtime parameter group.
Stop edges are listed in the credit map and prediction diagnostics.

*Inference and learning are separate vocabularies.* `ness.inference.INFERENCE_PROFILES`:
`direct`, `fabricpc_feedforward`, `fabricpc_spc`, `fabricpc_epc` verified; `fabricpc_spc_recurrent`
experimental (explicit opt-in); `reliability_settling`, `recurrent_workspace`, `query_settle`
unsupported. `ness.learning.LEARNING_PROFILES`: `bp_direct`/`ff_bp`, `fabricpc_pc_local`,
`fabricpc_bp_through_inference` and the PDF names they implement (`workspace_pc_local`,
`workspace_epc_local`, `workspace_bp_unroll`) verified; `reliability_bp_unroll`,
`workspace_ep_centered` unsupported. Names resolve to plugins; the plugin's `AlgorithmSpec`
in the manifest is the identity (`manifest.extra["algorithm_specs"]`).

*Losses* are owned by tasks (`loss_terms(pred, y, mask, xp)`), reduced as
sum-of-numerators / sum-of-denominators once per task, weighted across tasks; never a mean of
means. Optimizers (`adam`, `sgd`) skip frozen groups and snapshot their moments for learner
checkpoints.

## 6. Host-side subsystems

**Symbolic** (`ness.symbolic`): `ProgramIR` (`ness.program/1`; ops `input, const, follow,
intersect, union, exists, count, call`), canonical hash invariant to input renaming, primitive
registry (`window_slope`, `window_mean`, `last_value`) with typed signatures, and a
`ReferenceInterpreter` with fuel/rows/depth budgets, witness traces and orthogonal
`truth_status` / `completeness` / `status`. The trusted-helper program of the PDF is a golden test.

**Memory** (`ness.memory`): `MemoryStore` protocol (`open_view, query, fork, append, seal`) and
`InMemorySnapshotStore`. Snapshots are content-addressed; `fork` copies; `append` requires the
expected head and skips duplicate event ids; a `MemoryView` is pinned with an `AvailabilityCut`
and namespace filter *before* any ranking; analogue queries return `RetrievalEvidence` +
`FeatureEvidence` whose knownness is all-false when nothing matched. Hyperon would implement the
same protocol behind an adapter.

**Probabilistic** (`ness.probabilistic`): `ProbabilisticReasoner` contract (`describe_reasoner,
compile, infer`) implemented by `ExactFiniteRegimeReasoner` (exact enumeration, weight semantics
`exact_posterior`) and `ImportanceSamplingRegimeReasoner` (`approximate_posterior`, effective
sample size in diagnostics). Both are also `MacroModule`s consuming declared feature ports.

## 7. Transaction, identity, checkpoints, experiments

`NessSystem.build(exp, arm, registry)` compiles the arm, initialises module states and edge
parameters from the arm seed, resolves inference and learning profiles, builds the credit map,
and attaches memory (seeding a snapshot from matured scenario episodes if requested).

`PredictionTransaction` implements the transaction: `predict()` pins the manifest and memory
views, applies the task's `permitted_view`, executes, and returns an immutable
`PredictionRecord`; `reveal()` scores and stores an idempotent `ExperienceEvent`; in
`prequential` mode `learn_step()` performs one complete optimizer update and then publishes
pending memory to a *new* snapshot (in-flight predictions keep their pinned views). `frozen`
mode performs no updates of any kind.

`NessSystem.manifest()` returns the `PredictorManifest`; `snapshot("serving"|"learner")`
produces a `SystemSnapshot`; `CheckpointStore.stage_and_publish` writes content-addressed `.npy`
blobs and an index, validates, and publishes with an expected-parent compare-and-swap per arm.
`NessSystem.from_snapshot` recompiles the composition from the stored spec, refuses a
composition-hash or manifest-id mismatch, and restores each component through its plugin's
`restore_state` (fail-closed on version/schema/config).

`run_experiment` shares one scenario across arms so every arm sees identical request streams
(the training phase feeds every arm, frozen or trainable, the same `batch_size x updates`
requests), trains prequentially (per model seed in `protocol.seeds`), publishes a learner checkpoint and
re-predicts test requests from the restored system, evaluates frozen, runs a causal check, and
writes structured raw artifacts (`config/ manifests/ checkpoints/ data/ predictions/ metrics/
traces/ environment/ logs/`). Configuration-only `substitutions` are checked (validate, predict,
causal, train, restore) and recorded with resolved plugin ids and the core source hash.
`ness report <run_dir>` regenerates metrics, twelve figures and a Markdown report from those
artifacts alone (`ness.experiments.toy_report`).

## 8. How to use

```bash
pip install -e ".[jax]"                              # core + NESS JAX backend
pip install -e ".[fabricpc]"                         # + FabricPC 0.6.x backend (GPU: see docs/INSTALL.md)
pip install -e examples/external_plugin_example      # optional: third-party plugin example
ness doctor --backend fabricpc --json doctor.json    # versions, bootstrap order, devices, probed capabilities
ness plugins                                         # discovered plugins, missing dependencies flagged
ness validate examples/vertical_slice_timeseries/configs/vertical_slice.yaml   # compile every arm, print resolved wiring
ness run examples/vertical_slice_timeseries/configs/vertical_slice.yaml --out runs/vs
ness evaluate runs/vs/report.json
ness inspect runs/vs/checkpoints                     # manifests; add an id for components + wiring
ness audit                                           # dependency lock + capability status
ness run examples/fabricpc_workspace/configs/fabricpc_arms.yaml --out runs/fpc   # PC / ePC / hybrid arms
ness run examples/vertical_slice_timeseries/configs/toy_vertical_slice.yaml --out runs/toy/<id>   # A0-A5 x seeds + substitutions
ness report runs/toy/<id>                            # metrics/, plots/fig01..12, report/*.md from saved artifacts only
pytest -q                                            # 204 tests; backend tests skip when the extra is absent
```

Programmatic use:

```python
from ness.config import load_experiment
from ness.plugin_api import default_registry
from ness.runtime.system import NessSystem
from ness.runtime.transaction import PredictionTransaction

exp = load_experiment("examples/vertical_slice_timeseries/configs/vertical_slice.yaml")
reg = default_registry()
system = NessSystem.build(exp, exp.arm("neuro_symbolic_probabilistic_cap"), reg)
scenario = system.scenario
tx = PredictionTransaction(system, mode="prequential")
for req in scenario.iter_requests("train", tuple(system.tasks.values()), {"max_requests": 16}):
    rec = tx.predict(req)                                      # immutable, target-free
    tx.reveal(req.request_id, {q.task_id: scenario.outcome(req.request_id, q.query_id) for q in req.queries})
    if tx.pending_batch_size() == 8:
        print(tx.learn_step())                                 # one complete BP update
print(system.manifest().manifest_id, tx.aggregate_scores())
```

## 9. Extension points

| to add | implement | register | see |
|---|---|---|---|
| lower substrate | `MacroModule` (kind `substrate`) exposing `state/*` and `forecast/*` ports, frozen group, `SubstrateCapabilities` | entry point `ness.plugins` | `reference_plugins/frozen_substrates.py`, example plugin `ar_substrate.py` |
| upper module | `MacroModule` + `apply` (kind `upper_module`, runtime `jax`) with a trainable group | entry point | `reference_plugins/upper_modules.py` |
| semantic / neuro-symbolic module | kind `semantic_adapter`; declare which ports it reads; return typed evidence with `used_inputs` | entry point | `reference_plugins/semantic.py` |
| symbolic program | a `ness.program/1` IR in config for the generic `typed_program` node; new primitives via `PrimitiveRegistry` | config / registry | `symbolic/operators.py` |
| probabilistic reasoner | kind `reasoner`; `ProbabilisticReasonerCapabilities`; outputs `HypothesisEvidence` with honest weight semantics | entry point | `probabilistic/reasoners.py`, example plugin `reasoner.py` |
| memory backend | `MemoryStore` protocol (kind `memory_store`) | entry point | `memory/store.py` |
| memory query | kind `memory_query`, reads `ctx.memory_views[store]` only | entry point | `reference_plugins/memory_query.py` |
| cap / writer / consumer | kind `cap` (or reuse `residual_*_cap` with a new `writer`/`consumer` plugin) | entry point | `reference_plugins/caps.py`, `writers_consumers.py` |
| task | `PredictionTask` protocol (kind `task`) | entry point | `tasks/forecast_tasks.py` |
| dataset / scenario | `ScenarioProvider` protocol (kind `scenario`) | entry point | `scenarios/toy_temporal.py` |
| learning rule | `LearningRule` (kind `learning_rule`, `required_runtime()`, `gradients(ctx, batch, owned)`) + profile entry with status | entry point + `learning/profiles.py` | `backends/jax/region.py`, `backends/fabricpc/rules.py` |
| numerical backend | `ness/backends/<name>/` with a bootstrap hook, capability probes, version routing and compat modules | `runtimes/bootstrap.py` | `backends/fabricpc/` |

Contract-test mixins in `ness.plugin_api.testkit` cover each family; the example plugin's tests
show the intended usage.

## 10. Test map (PDF T-ids and addendum T33-T44)

| id | where |
|---|---|
| T01 locks / unsupported capabilities | `test_runtime_system.py::test_unsupported_profiles_fail_explicitly`, `test_learning.py::TestProfiles`, `ness audit` |
| T02 / T03 interpreter parity, witness & unknown semantics | `test_symbolic.py` |
| T04 / T38 causal noninterference, target isolation | `test_runtime_system.py` (`assert_target_free`), `test_probabilistic.py` |
| T05 / T06 forward and gradient parity | `test_learning.py::TestBPDirect`, `DifferentiableModuleMixin` |
| T07 / T23 reductions | `test_learning.py::test_T07_*`, `test_tasks_scenario.py` |
| T08 resume / publication CAS | `test_checkpoint.py` |
| T09 / T39 memory isolation | `test_memory.py`, `test_runtime_system.py::TestMemoryArm` |
| T16 fail-closed programs | `test_symbolic.py::TestFailClosed` |
| T17 unknown modality through core | `test_runtime_system.py::test_T17_*` |
| T18 forecast algebra, writer neutrality | `test_contracts.py`, `test_tasks_scenario.py`, `test_runtime_system.py::test_zero_init_*` |
| T19 / T21 / T44 availability, target roles | `test_contracts.py::TestObservationsAndAccess`, `test_composition.py::test_T44_*`, `ScenarioContractMixin` |
| T22 / T42 gradient boundary | `test_learning.py::TestGradientBoundary` |
| T24 frozen substrate state | `FrozenModuleMixin`, `test_learning.py::test_update_reduces_*` |
| T27 / T40 manifest restore, plugin round trip | `test_checkpoint.py`, `ModuleContractMixin` |
| T30 / T41 scoped dependencies, third-party isolation | `test_plugins_external.py` |
| T31 / T33 provider & module substitution | `test_vertical_slice.py::TestSubstitutionProof` |
| T34 arbitrary earlier-port wiring | `test_runtime_system.py::test_T34_*` |
| T35 residual / normalization semantics | `test_composition.py`, `test_vertical_slice.py::test_residual_merge_*` |
| T36 explicit recurrence boundary | `test_composition.py::test_T36_*` |
| T37 reasoner substitution | `test_probabilistic.py`, `test_vertical_slice.py::test_swap_probabilistic_reasoner` |
| T43 composition provenance | `test_composition.py::test_T43_*`, `test_runtime_system.py::test_T43_*` |
| T13 PC update audit (FabricPC) | `tests/fabricpc_backend/test_fabricpc_integration.py` (clamped energy falls under `fabricpc_pc_local`; algorithm specs recorded), `test_fabricpc_parity.py` |
| T14 distributed reduction | `tests/fabricpc_backend/test_fabricpc_multidevice.py` (2 forced CPU devices; 2 GPUs), `test_jax_backend_sharded.py` |
| T01/T30 locks and gates (backend) | `tests/test_bootstrap_import_order.py`, `tests/test_devices.py`, `ness.backends.fabricpc.version` |
| T22 runtime boundary (FabricPC) | `test_fabricpc_integration.py::test_cross_runtime_ownership_is_refused`, `::test_learned_boundary_transform_into_fabricpc_node_is_refused` |

Not exercised (no implementation to test): T10 caches, T11 paired optimizer state (partially:
identical request streams), T12 sparse probability interface, T15 accounting beyond per-node
costs, T20 alignment, T25/T26/T28/T29 provider-specific geometry and modality tests, T32
search utility. T13/T14 are exercised for the FabricPC backend at toy scale only.
