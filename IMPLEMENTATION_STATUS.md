# IMPLEMENTATION_STATUS.md

Status vocabulary: **verified** = implemented and covered by passing tests in this repository;
**implemented** = present, exercised by the vertical slice but with thinner tests;
**experimental** = present but not to be relied on scientifically; **design-only** = declared
name/contract exists, execution raises `UnsupportedCapability`.

Test suite: 204 tests (`pytest -q`; backend-specific tests skip when an extra is absent: 3 skips
on CPU with FabricPC installed, more without JAX). Both wheels build
(`dist/ness-0.1.0-py3-none-any.whl`, `dist/ness_example_plugin-0.1.0-py3-none-any.whl`) and the
core wheel installs into a fresh venv without JAX/FabricPC (`ness doctor --backend numpy`,
schema-only validation, core tests). Exact environments exercised: CPU jax 0.10.2 + fabricpc 0.6.0;
CPU jax 0.7.0 (range minimum) + fabricpc 0.6.0; 2 x RTX 3090 jax 0.10.2 (cuda12) + fabricpc 0.6.0.

## Verified

* Typed contracts: observations/roles/availability, access policies, forecasts with capability
  algebra, prediction spaces, typed evidence + immutable evidence graph, requests/outcomes,
  ports/descriptors, capabilities, restricted state snapshots, manifests, canonical hashing.
* Composition graph: parser (YAML/JSON sugar), selectors, compiler (DAG, declared recurrent
  regions, access policy per edge, shapes, coordinates, learned-op placement, gradient
  boundaries, output/task compatibility, resolved wiring, composition hash).
* Plugin API: `MacroModule`/`BaseModule`, registry with entry-point discovery, lazy factory
  import, declared optional dependencies (fail closed), contributor test kit (7 mixins +
  harness + causality helpers); external example plugin registers and composes with core
  modules without core edits (T41), heavy-dependency plugin blocks only itself (T30).
* Symbolic: `ness.program/1` IR, canonical identity, reference interpreter with budgets,
  witnesses, truth/completeness/error orthogonality, fail-closed opcodes/primitives (T02/T03/T16).
* Memory: in-memory snapshot store (fork isolation, expected-head CAS, idempotent events,
  content-addressed seals, causal views filtered before ranking, export/import) (T09/T39).
* Probabilistic: exact finite reasoner and importance-sampling comparator with honest weight
  semantics, missing-feature marginalisation, request-RNG determinism (T37/T38).
* Runtime: executor with typed pass-through, provenance on every port value, evidence graph per
  request; `NessSystem` build/predict/learn/manifest/snapshot/restore; transaction with
  frozen/prequential modes, idempotent experience events, memory learner branch + publication.
* Learning: NESS JAX backend differentiable region (forward parity with eager path < 1e-9;
  gradients match finite differences), `bp_direct`/`ff_bp` in per-item, batched (vmap) and
  data-parallel (sharded `("data",)` mesh) modes with identical objective, credit map refusing
  cross-runtime ownership and supporting hybrid multi-rule ownership (T22/T42, PDF §11.6),
  sum-then-divide reductions (T07), frozen groups untouched (T24), Adam/SGD with restorable state.
* Runtime bootstrap and devices: single owner (`ness.runtimes.bootstrap`), `DeviceSpec` with
  requested/visible/selected/realized sets, explicit fallbacks only, `ness doctor` (text + JSON),
  subprocess import-order tests (`import ness` imports no JAX/FabricPC; backend import does not
  initialise; late `setup_jax` detected; scientific profile fails closed).
* FabricPC backend (FabricPC 0.6.0, adapter `0.6-adapter/1.0.0`): version routing (verified /
  qualified / unsupported / experimental_override), `fabricpc_residual_cap` (dense PC graph,
  target-free readout, zero-init readout => baseline parity), profiles `fabricpc_feedforward`,
  `fabricpc_spc`, `fabricpc_epc`; rules `fabricpc_pc_local` (PDF workspace_pc_local /
  workspace_epc_local) and `fabricpc_bp_through_inference` (PDF ff_bp / workspace_bp_unroll);
  forward + gradient + one-update parity with the NESS JAX twin (1e-8); clamped energy falls
  under the PC rule and task loss falls under the BP rule; complete `AlgorithmSpec` per node and
  rule in manifests; restricted-array checkpoints with digest/family fail-closed restore;
  probe-based capabilities; data parallelism verified on 2 forced CPU devices and on 2 GPUs
  (gradients/loss/predictions identical to < 1e-9, odd batches refused); single GPU verified.
* Tasks/scenario: point (mse/mae) and quantile (pinball, coverage/width/monotone diagnostics)
  tasks with as-of-origin policies; two synthetic scenarios (`toy_temporal_dataset`,
  `toy_regime_dataset` with known-future events and recorded change points) with role-tagged
  fields incl. outcome-only/oracle fields removed by the policy (T44); rolling-origin splits.
* Experiment pipeline: seeds, structured raw artifacts (config/manifests/checkpoints/data/
  predictions/metrics/traces/environment/logs), in-run checkpoint-restore comparison, causal
  checks, substitution matrix with core-source hash, `ness report` (metrics + 12 figures +
  Markdown from artifacts only), `ness evaluate <run_dir>`.
* Checkpoints: content-addressed pickle-free blobs, staged atomic publication with
  compare-and-swap, load with checksum/hash verification, whole-system restore reproducing
  predictions bitwise (serving and learner, with and without memory) (T08 subset, T27, T40).
* Experiments/CLI: paired arms on identical request streams (T11 subset), reports with
  per-request scores, `ness validate/run/evaluate/inspect/audit/plugins`.
* Vertical slice: six arms + external-plugin arm; zero-init caps reproduce the native baseline
  bitwise (T18); every addendum §10.3 substitution is configuration-only (T31/T33-T35, T43).

Reference run (`ness run examples/vertical_slice_timeseries/configs/vertical_slice.yaml`,
CPU, ~55 s; engineering demonstration, not a research result; 16 updates of batch 8):

```
arm                               test loss   train loss  updates  n_test
native_baseline                   0.34936     0.29563      0        96
neural_only_cap                   0.36564     0.45796     16        96
neuro_symbolic_cap                0.36653     0.46070     16        96
neuro_symbolic_probabilistic_cap  0.36799     0.46175     16        96
with_memory                       0.34567     0.45817     16        96
provider_substitution             0.28120     0.36615     16        96
```

Reference run of the FabricPC arms (`examples/fabricpc_workspace`, CPU, 12 updates of batch 8, 64 test requests):

```
arm                               test loss   train loss  updates
native_baseline                   0.36587     -            0
jax_dense_bp                      0.31916     0.38166     12
fabricpc_feedforward_bp           0.31457     0.37585     12
fabricpc_spc_pc_local             0.31356     0.37418     12
fabricpc_spc_bp_unroll            0.31441     0.37581     12
fabricpc_epc_pc_local             0.31781     0.38057     12
hybrid_jax_upper_fabricpc_cap     0.34580     0.41270     12
```
(engineering demonstration; `fabricpc_feedforward_bp` == `fabricpc_spc_bp_unroll` because a
target-free settle on an acyclic workspace at feedforward init is the feedforward pass.)

## Toy vertical slice reference run (2026-09-26, `runs/toy_vertical_slice/20260926-220302-final`)

Executed end to end per the toy-slice requirements document: `toy_regime_dataset` (target `y`,
auxiliary `x`, known events `e`, hidden two-state regime with recorded change points), arms A0-A5
over three model seeds, ten configuration-only substitution proofs (12/12 rows incl. sub-variants
passed validate/predict/causal/train/restore), in-run checkpoint publish + restore (max
|Δprediction| = 0 for every arm/seed), causal checks (all pass), twelve figures and a Markdown
report regenerated from saved CSV/JSON by `ness report`, plus a LaTeX/PDF write-up in `reports/`.
Test MAE (mean over 3 seeds): A0 0.646, A1 0.624, A2 0.608, A3 0.607, A4 0.594, A5 0.524. The
symbolic/probabilistic/memory arms improve consistently but slightly over the neural-only cap;
the same-fact neural control A5 is the best arm: explicit symbolic composition did not add
predictive value beyond information access on this toy problem (null result, reported as such).
Regime posterior (A3): accuracy 0.81, Brier 0.14, NLL 0.43 against the evaluation-only hidden regime; event predicate agrees with ground truth on 100% of test origins.
Protocol note: every arm, frozen or trainable, is fed the same training request stream
(`batch_size x updates` requests), so train-phase metrics are comparable across arms and the
per-request streams are identical in both phases; earlier runs under `runs/toy_vertical_slice/`
(`20260926-214028-v2`, `20260926-214605-clean`) predate that rule and a substitution-restore fix
and are kept as evidence, not as the reference. Test metrics are bitwise identical across the runs.
The recorded core source hash (`environment/core_source_hash.json`, `cbf471e80120…`) is the tree at run
time; the only core edit after the run is a tick-label change in `experiments/toy_report.py` (fig 11).

## Implemented (thinner tests)

* `attention_pool` and `cross_attention` merges (shape rules and ops exist; compiled in tests
  only for the simpler merges).
* `residual_quantile_cap` + `monotone_quantile_writer` + native quantile grid from the frozen
  transformer (parity and monotonicity tested; no trained quantile arm in the shipped config).
* Cost ledger per node (realised costs recorded; no budget enforcement beyond interpreter/query
  budgets).
* Reasoner `compile()` path (tested indirectly through `forward`).

## Experimental

* `fabricpc_spc_recurrent`: cyclic hidden FabricPC graph (`unroll: U`) with sPC settling; runs,
  lowers energy, explicit opt-in (`allow_experimental: true`); unrolled-cycle semantics are FabricPC's.
* FabricPC version override (`NESS_FABRICPC_ALLOW_UNVERIFIED=1`): runs an untested family with
  the newest adapter, never as verified.
* Masked objectives on FabricPC: honoured by `fabricpc_bp_through_inference`, refused by
  `fabricpc_pc_local` (per-sample energies).
* FabricPC `InferenceSchedule` (composed ePC -> sPC) is not exposed by NESS plugins.
* Training speed: per-request tracing on the jax backend unless `batched`/`data_parallel`;
  FabricPC rules retrace per call (no persistent jit cache across updates).
* Toy frozen providers' self-pretraining (deterministic; a stand-in for external weights).

## Design-only (explicit `UnsupportedCapability`)

* Inference profiles `reliability_settling`, `recurrent_workspace` (composition-level regions),
  `query_settle`; learning profiles `reliability_bp_unroll`, `workspace_ep_centered` (no nudge
  phase in FabricPC 0.6, none in NESS).
* Model/tensor parallelism and multi-host execution (represented separately in
  `BackendCapabilities`; FabricPC 0.6 reserves a `model` mesh axis but ships no sharding).
* Cross-runtime gradient bridges (jax <-> fabricpc <-> torch).
* Hyperon memory backend, TimesFM / vision / text substrates, reference44 port, separate task
  views per request, caches (T10), sparse vocabulary interface (T12), structural search (§15),
  alignment records (§8.5) beyond the `BindingEvidence` type, `adaptive` test-time inference mode.

## Known simplifications to revisit

* Reference caps ignore knownness masks (unknown -> 0; declared in capabilities).
* Frozen providers reject contexts longer than `max_context` instead of bucketing.
* The dependency lock records `fabricpc`/`hyperon` as `unresolved`; a scientific launch of any
  profile needing them must fail on that (T01) once such profiles exist.
