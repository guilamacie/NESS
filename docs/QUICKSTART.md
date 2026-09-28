# NESS quick start

This guide takes you from an empty environment to running, modifying and extending a NESS
experiment. Everything in NESS is driven by one YAML file per experiment: it names the data,
the tasks, the training protocol, and one or more *arms*, each of which composes plugins into
a predictor. You configure; you do not edit core code. The runnable files for every example
below live in [`examples/quickstart/`](../examples/quickstart/).

## 1. Install (pick a level)

```bash
python -m venv .venv && . .venv/bin/activate
pip install "ness[jax] @ git+https://github.com/guilamacie/NESS.git"        # core + NESS JAX backend (CPU); enough for examples 01-05, 07
pip install "ness[fabricpc] @ git+https://github.com/guilamacie/NESS.git"   # + FabricPC 0.6.x backend; needed for example 06
pip install "ness[report] @ git+https://github.com/guilamacie/NESS.git"     # + matplotlib/pandas for `ness report`
```

From a clone, `pip install -e ".[dev]"` gives all of the above plus pytest, and
`pip install -e examples/external_plugin_example` installs the third-party plugin package used by
example 05. GPU and version details: [`INSTALL.md`](INSTALL.md).

Check the environment before anything else:

```bash
ness doctor --backend jax --platform cpu     # versions, bootstrap order, devices
ness plugins                                 # every plugin discovered, missing dependencies flagged
```

## 2. The five commands

```bash
ness validate <config.yaml>                  # parse, compile every arm, print the resolved wiring; no computation
ness run      <config.yaml> --out <run_dir>  # train/evaluate every arm (and substitution proofs), write raw artifacts
ness evaluate <run_dir>                      # summary table (test loss/MAE, updates, restore diff, causal check)
ness report   <run_dir>                      # figures + Markdown report regenerated from the saved artifacts
ness inspect  <run_dir>/checkpoints [<manifest-id>]   # whole-system manifests; components and wiring of one
```

`ness run --arms a b` runs a subset; `--no-substitutions` skips the substitution proofs;
`--only-substitutions` refreshes just those in an existing run directory.

Try it on the shipped study (six arms, three seeds, ten substitutions, about five minutes on CPU):

```bash
ness run examples/vertical_slice_timeseries/configs/toy_vertical_slice.yaml --out runs/toy/first
ness evaluate runs/toy/first
ness report runs/toy/first        # runs/toy/first/plots/fig01..fig12.png, report/toy_vertical_slice_report.md
```

## 3. Anatomy of an experiment file

```yaml
schema_version: ness.experiment/3
protocol_id: my_study_v1                     # part of every manifest and artifact

runtime:                                     # optional; auto = least capable backend the arms need
  backend: auto                              # numpy | jax | fabricpc | auto
  platform: auto                             # cpu | gpu | tpu | auto  (a pinned gpu fails closed unless devices.fallback: cpu)
  x64: true
  profile: development                       # scientific = unverified backend versions fail closed

scenario:                                    # the data: any plugin of kind `scenario`
  plugin: toy_temporal_dataset
  config: {n_steps: 400, context: 32, horizon: 4, seed: 1234, train_fraction: 0.7}

tasks:                                       # what is predicted and how it is scored
  - id: future_value
    plugin: point_forecast_task              # or quantile_forecast_task
    config: {horizons: 4, channels: [y0, y1], loss: mse}

protocol:                                    # the paired training/evaluation protocol shared by all arms
  train_split: train
  test_split: test
  batch_size: 8                              # requests per optimizer update
  updates: 8                                 # every arm (frozen or trainable) sees batch_size x updates train requests
  seeds: [0, 1, 2]                           # model-initialisation seeds (optional; default = each arm's own seed)
  max_test_requests: 48                      # optional cap on test origins
  restore_check_requests: 8                  # test requests re-predicted after checkpoint restore (default 8)
  causal_check: true                         # perturb withheld outcomes and re-predict (default true)

arms:                                        # each arm is a complete predictor
  my_arm:
    description: free text
    seed: 0
    learning:  {rule: bp_direct, optimizer: {kind: adam, lr: 0.003, clip_norm: 5.0}}   # required if any node is trainable
    inference: {profile: direct}             # optional; direct is the default
    memory: {...}                            # optional; see section 6
    composition:
      nodes: [ ... ]                         # see section 4
      output: {task: future_value, from: module://cap/forecast}

substitutions: [ ... ]                       # optional: arms that must validate/predict/train/restore (section 5)
```

YAML anchors (`x-base: &base` ... `*base`) are the intended way to share node fragments between
arms; the parser reads only the keys named above, so top-level fragment holders (by convention
prefixed `x-`) are ignored.

## 4. Composing an arm: nodes and wiring

A node is a plugin instance plus its inputs:

```yaml
- id: upper
  plugin: tiny_upper_transformer            # `ness plugins` lists ids and kinds
  config: {in_dim: 16, model_dim: 16, heads: 2}
  inputs:
    main: substrate://base_ts/state/final    # one source: a selector string
```

Selectors are `scheme://node/port`: `observation://<field>` (a scenario field),
`substrate://<node>/<port>`, `module://<node>/<port>`, `semantic://<node>/<port>`,
`program://<node>/<port>`, `reasoner://<node>/<port>`, `memory://<node>/<port>`.
`ness validate` prints the ports every node exposes, and the compiler rejects a wire whose
shape, semantic type, access role or gradient boundary does not fit.

An input can transform its source or merge several sources:

```yaml
inputs:
  neural: {from: substrate://base_ts/state/final, boundary: [mean_pool]}          # parameter-free reduction
  main:
    merge: gated_add                                                              # concat | add | gated_add | weighted_sum | select | attention_pool | cross_attention
    sources:
      - substrate://base_ts/state/final
      - {from: substrate://base_ts/state/early, boundary: [{kind: linear, out_dim: 16}]}   # learned edge parameters
    post: [{kind: layer_norm}]                                                    # identity | linear | layer_norm | rms_norm | standardize | mean_pool | last_step | flatten
```

Learned boundaries (`linear`, gated merges) are trainable edge parameters owned by the arm's
learning rule. They may not feed a host-runtime or FabricPC node: gradients do not cross
runtimes, and the compiler says so at validate time.

The canonical stack, bottom to top (each layer is optional except the output):

| layer | kind | reference plugins | reads | emits |
|---|---|---|---|---|
| lower substrate | `substrate` | `toy_frozen_transformer`, `toy_frozen_conv`, external `toy_linear_ar_substrate` | observation fields | `state/early`, `state/final`, `forecast/point` (+ `forecast/quantiles`) |
| upper module | `upper_module` | `tiny_upper_transformer`, `tiny_upper_mlp` | substrate states | `hidden` (trainable) |
| semantic adapter | `semantic_adapter` | `simple_temporal_semantics` | raw fields, any earlier port | `features` (typed FeatureEvidence) |
| program | `program` | `typed_program` (+ a `ness.program/1` IR in config) | bound ports | `value` |
| reasoner | `reasoner` | `exact_finite_regime_reasoner`, `importance_sampling_regime_reasoner`, external `deterministic_threshold_reasoner` | features | `posterior` (HypothesisEvidence) |
| memory query | `memory_query` | `analogue_memory_query` | a pinned memory view | `evidence`, `retrieval` |
| cap | `cap` | `residual_point_cap`, `residual_quantile_cap`, `jax_dense_workspace_cap`, `fabricpc_residual_cap` | a baseline forecast + named feature ports | `forecast`, `correction` |

The cap's `features:` map declares every feature port and its dimension; the compiler checks
each against the producing port. A cap reproduces its baseline exactly at initialisation.

## 5. Worked examples

Each file runs with `ness run examples/quickstart/<file> --out runs/qs/<name>` and validates
with `ness validate`.

| file | shows | needs |
|---|---|---|
| `01_baseline_only.yaml` | the frozen provider's own forecast, no learning: the control arm | core |
| `02_neural_cap.yaml` | substrate -> trainable upper module (gated residual merge) -> residual cap; `learning:` | jax |
| `03_symbolic_probabilistic.yaml` | + semantic adapter, typed program, exact two-state reasoner feeding the cap | jax |
| `04_with_memory.yaml` | + a memory store attached to the arm and an analogue query lowered to evidence | jax |
| `05_external_plugin_substitution.yaml` | third-party substrate, reasoner and writer; `substitutions:` proofs | jax + example plugin |
| `06_fabricpc_workspace.yaml` | the cap on FabricPC: feedforward BP vs. sPC settling + local PC rule | fabricpc |
| `07_quantiles.yaml` | a different prediction space: native quantiles, pinball loss, monotone quantile cap | jax |

These protocols are deliberately tiny (a few hundred steps, 6-8 optimizer updates) so each
file runs in under a minute on CPU. Expect the learned arms *not* to beat the frozen baseline on
the test split at that budget; the paired control is exactly what tells you so. The shipped
study in `examples/vertical_slice_timeseries/` is the properly sized version.

**Reading the results.** `<run_dir>/summary.txt` is the table `ness evaluate` prints;
`metrics/summary.json|csv` hold per arm and seed test loss, MAE, RMSE, updates, restore
difference (must be exactly 0) and the causal-check verdict; `predictions/*.csv` hold every
prediction and truth; `traces/*_evidence.csv` hold the typed evidence each node produced;
`metrics/substitution_matrix.*` hold the proofs; `manifests/` and `checkpoints/` hold the
whole-system checkpoints; `environment/` records versions and the core source hash.

## 6. Memory: when and how

Memory is **optional**. An arm without a `memory:` section instantiates no store and no view;
the design requires "no-memory" arms to be valid, and every example except 04 is one.

Attach memory only when a node consumes it:

```yaml
tasks:
  - id: future_value
    plugin: point_forecast_task
    config: {horizons: 4, channels: [y0, y1], memory_namespaces: [episodes]}   # task access policy (default: [episodes])

arms:
  with_memory:
    memory:
      name: episodes                          # what memory-query nodes name in `store:`
      store: {plugin: in_memory_snapshot_store}   # any plugin of kind `memory_store` (a MemoryStore implementation)
      namespace: episodes
      seed_from_scenario: true                # initial snapshot: episodes matured before the first train origin
      learner_appends: true                   # revealed episodes are appended on the learner branch and published between updates
      episode_window_field: target_history    # which observation field forms the episode window
    composition:
      nodes:
        - {id: analogues, plugin: analogue_memory_query,
           config: {store: episodes, namespace: episodes, k: 3, horizon: 4, channels: 2}, inputs: {query: observation://target_history}}
        - {id: cap, plugin: residual_point_cap, config: {..., features: {neural: 16, memory: 8}},
           inputs: {..., memory: memory://analogues/evidence}}
```

What happens at run time: for every request the system opens a view of the current snapshot
cut at the request origin (only records whose `available_at` is at or before the origin are
visible), the query node runs one bounded query against that pinned view, and the result is
lowered to `FeatureEvidence` (mean matured continuation of the top-k analogues, with
knownness false when nothing eligible exists) plus `RetrievalEvidence` (returned ids, candidate
count, completeness, truncation, view id). Nothing writes to memory inside a prediction. With
`learner_appends`, revealed episodes are appended to a learner branch and sealed into a new
content-addressed snapshot between optimizer updates; in-flight predictions keep their pinned
views. The snapshot id is part of the whole-system manifest, so a restored system queries the
same memory.

A memory-query node wired without a `memory:` section compiles but fails at the first prediction
with a clear message; remove the node and its cap feature to go memory-free.

## 7. Bring your own data

Implement the `ScenarioProvider` protocol (`ness.plugin_api.protocols`): `observation_fields()`
declares each field with its role (`observed`, `known_future`, `derived`, `outcome_only`,
`evaluation_oracle`), `iter_requests(split, tasks, protocol)` yields target-free prediction
requests, `outcome(request_id, query_id)` reveals outcomes after the fact, and
`grouping_metadata` tags requests for diagnostics. Optionally add `matured_episodes(before_origin,
namespace)` to let memory seed itself. Register the class through the `ness.plugins` entry point
(section 8) and point `scenario.plugin` at it. `src/ness/scenarios/toy_regime.py` is a complete,
small example including an oracle field that predictors cannot read.

## 8. Bring your own models (plugins)

A plugin is a class with a `plugin_id`, a `module_kind`, a `runtime` (`numpy`, `host`, `jax`,
`fabricpc`), `validate_config`, `describe()` (its ports, parameter groups and capabilities),
`initialize(rng)` and `forward(inputs, state, ctx)`; trainable modules also implement
`apply(params, dense_inputs, state, xp)` so the same code runs eagerly and under tracing.
State is snapshotted and restored as restricted data. Registration is one entry point:

```toml
[project.entry-points."ness.plugins"]
my_substrate = "my_package.descriptors:MY_SUBSTRATE"    # a PluginDescriptor(id, kind, version, "module:Class", deps, doc)
```

Start from `examples/external_plugin_example/` (a substrate, a reasoner, a writer and a
heavy-dependency plugin, with tests built on `ness.plugin_api.testkit`), then read
[`AGENT_HANDOFF.md`](AGENT_HANDOFF.md), which walks through one plugin per layer with the
contracts each must honour, and [`CONTRIBUTING_PLUGINS.md`](CONTRIBUTING_PLUGINS.md) for
lifecycle and state rules. Wrapping a real frozen model (a TimesFM checkpoint, a text or
vision encoder) means implementing a `substrate` that exposes `state/*` and `forecast/*` ports
with `Differentiability.STOP` and a frozen parameter group; the port contract, not the library,
is what the rest of the system sees.

## 9. Programmatic use

```python
from ness.config import load_experiment
from ness.plugin_api import default_registry
from ness.runtime.system import NessSystem
from ness.runtime.transaction import PredictionTransaction

exp = load_experiment("examples/quickstart/02_neural_cap.yaml")
system = NessSystem.build(exp, exp.arm("neural_cap"), default_registry())
scenario, tasks = system.scenario, tuple(system.tasks.values())
tx = PredictionTransaction(system, mode="prequential")
for req in scenario.iter_requests("train", tasks, {"max_requests": 16}):
    rec = tx.predict(req)                                            # immutable, target-free
    tx.reveal(req.request_id, {q.task_id: scenario.outcome(req.request_id, q.query_id) for q in req.queries})
    if tx.pending_batch_size() == 8:
        print(tx.learn_step())                                       # one complete update
print(system.manifest().manifest_id, tx.aggregate_scores())
```

## 10. Errors you will meet first

| message | meaning | fix |
|---|---|---|
| `has trainable parameter groups [...] but no learning.rule` | a node or edge is trainable | add `learning:` to the arm, or use a frozen provider only |
| `feature port X delivered (n,), declared (m,)` | cap `features:` dimension differs from the producer | set the declared dimension to what `ness validate` prints for the port |
| `memory query node ... requires memory store ... attached` | a `memory://` consumer without `memory:` | add the memory section or remove the node |
| `plugin 'X' requires ['jax'] which are not installed` | the plugin's runtime extra is missing | `pip install "ness[jax]"` (or `[fabricpc]`) |
| `learned boundary transform 'linear' feeds a non-differentiable runtime` | a `linear`/gated boundary feeds a host or FabricPC node | use a parameter-free boundary (`mean_pool`, `flatten`, ...) |
| `runtime already bootstrapped ... cannot re-initialise JAX` | two incompatible runtime requests in one process | run the FabricPC arm in a fresh process, or bootstrap it first |
| `AccessPolicyViolation` | a wire reads a field the task view forbids (an outcome, an oracle, a namespace) | it is the design working; read only permitted fields |

## 11. Where next

`ARCHITECTURE.md` (interfaces and layout), `COMPOSITION_GRAPH.md` (the wiring language in
full), `PROBABILISTIC_REASONERS.md`, `CHECKPOINT_CONTRACT.md`, `FABRICPC_BACKEND.md`, and
`IMPLEMENTATION_STATUS.md` for what is verified, experimental or design-only.
