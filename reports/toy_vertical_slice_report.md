# NESS toy vertical slice - run report

Run directory: `runs/toy_vertical_slice/20260926-220302-final`  Protocol: `toy_vertical_slice_v2` (`b7261445860b`)

## 1. Executive summary

* End-to-end prediction -> record -> outcome -> learning -> checkpoint -> frozen evaluation ran for 18 arm/seed combinations with no failures.
* Substitution proofs: 12/12 configuration-only variants passed every applicable check (validate, predict, causal, train, restore); core source hash `cbf471e80120` for all.
* Checkpoint restore: max |Δprediction| over all arms = 0.000 (0 = bitwise).
* Baseline parity at neutral initialisation holds (zero-initialised writers); see substitution matrix and §13.
* Predictive results are an engineering smoke comparison on a toy problem; no superiority claim is made (§7).
* Lowest mean test MAE: `A5_same_fact_neural` = 0.5243 vs frozen baseline 0.6458 (3 seeds).
* Regime posterior (A3): accuracy 0.811, Brier 0.143 against the hidden regime (evaluation only).

## 2. Purpose and status

This run is an **engineering/reference demonstration** of the NESS interfaces and controls. It is not evidence that the neuro-symbolic or probabilistic system is superior to a Transformer; null or negative differences are valid outcomes and are reported as such.

## 3. Toy dataset

Generator (deterministic, seed 2026):

```
x_t = 0.8 x_(t-1) + 0.3 eps_x
NORMAL: y_t = 0.6 y_(t-1) + 0.4 x_(t-1) + 0.5 e_t + 0.1 eps
SHIFT : y_t = 0.6 y_(t-1) + -0.4 x_(t-2) + 1.5 e_t + 0.2 eps
e_t: 1 for event_length steps every event_period steps;  r_t: alternates every segment_length steps starting NORMAL
```

Change points: [90, 180, 270, 360, 450, 540, 630]. Context 32 steps, horizon 4. Rolling-origin splits (origin first/last/count): {'train': [32, 442, 411], 'dev': [446, 544, 99], 'test': [548, 716, 169]}.
Access policy: the predictor sees `series_history` (y, x) and `event_history` (observed) plus `event_future` (known_future). `target_future` is outcome-only and `true_regime` is an evaluation oracle; both are removed by every task view before prediction.

![Figure 1](plots/fig01_dataset.png)

*Figure 1. Representative segment: target, auxiliary channel, events; shaded regions mark the hidden SHIFT regime (evaluation only).*

## 4. Instantiated architecture

Resolved composition of `A4_full_with_memory@s2` (manifest `f19b4ceafe5d2dc9`):

| node | plugin | runtime | inputs |
|---|---|---|---|
| `analogues` | `analogue_memory_query@1.0.0` | host | `query` <- `observation://series_history` |
| `base_ts` | `toy_frozen_transformer@1.0.0` | numpy | `history` <- `observation://series_history` |
| `event_program` | `typed_program@1.0.0` | host | `events` <- `observation://event_future` |
| `slope_program` | `typed_program@1.0.0` | host | `series` <- `observation://series_history` |
| `upper` | `tiny_upper_transformer@1.0.0` | jax | `main` <- `substrate://base_ts/state/final`, `substrate://base_ts/state/early` |
| `semantic` | `simple_temporal_semantics@1.0.0` | numpy | `raw` <- `observation://series_history`; `neural` <- `module://upper/hidden`; `events` <- `observation://event_future` |
| `regime` | `exact_finite_regime_reasoner@1.0.0` | numpy | `semantic` <- `semantic://semantic/features` |
| `cap` | `residual_point_cap@1.0.0` | jax | `baseline` <- `substrate://base_ts/forecast/point`; `neural` <- `module://upper/hidden`; `semantic` <- `semantic://semantic/features`; `slope` <- `program://slope_program/value`; `event` <- `program://event_program/value`; `posterior` <- `reasoner://regime/posterior`; `memory` <- `memory://analogues/evidence` |

The upper module merges `state/final` with a learned linear projection of `state/early` through `gated_add`, followed by `layer_norm` (edge parameters are a separate manifest component `composition:boundaries`). The frozen substrate group is `frozen`; trainable groups are `upper/upper`, `cap/consumer` and the boundary parameters, all owned by `bp_direct`. Every edge into the cap from a host node (semantic, programs, reasoner, memory) is a stop-gradient leaf. Symbolic programs run through the reference interpreter; the reasoner is exact enumeration over two states; memory is a pinned snapshot view queried once per prediction.

![Figure 2](plots/fig02_architecture.png)

*Figure 2. Architecture/data-flow diagram generated from the resolved manifest of the full arm.*

## 5. Experiment arms

| arm | description |
|---|---|
| `A0_frozen_baseline` | frozen lower transformer's native point forecast; no learned cap |
| `A1_neural_cap` | frozen base + trainable upper transformer (final + projected early residual) + residual point cap |
| `A2_neural_symbolic` | A1 + semantic adapter (raw + upper state + events) + typed slope and event programs |
| `A3_neural_symbolic_probabilistic` | A2 + exact two-state regime reasoner (posterior HypothesisEvidence) |
| `A4_full_with_memory` | A3 + pinned analogue memory (matured episodes only; learner appends between updates) |
| `A5_same_fact_neural` | same permitted raw facts as A2/A3 (window, known-future events) fed directly to the cap; no symbolic composition |

## 6. Training and inference protocol

* Model seeds: [0, 1, 2]; dataset seed locked (2026).
* Prequential training on the train split: batch 8 requests per update, 24 updates (Adam, lr 0.003, clip 5); frozen evaluation on the test split (no updates, no memory writes).
* Inference profile `direct` (no settling); learning rule `bp_direct` (BP through the deployed direct computation). Frozen: substrate weights. Trainable: upper module, cap consumer, boundary transforms.
* Whole-system learner checkpoint published after training; the restored predictor re-predicts the first 16 test requests (Figure 11). Causal check: predictions are re-computed with outcome fields perturbed and removed (must be identical).
* Cost accounting: wall-clock per phase and process peak RSS (`metrics/timing.csv`); per-node realised costs in prediction records.

## 7. Forecasting results

| arm | MAE (mean±sd over seeds) | RMSE | MAE stable | MAE near transition (≤8) | n scored | failed |
|---|---|---|---|---|---|---|
| `A0_frozen_baseline` | 0.6458 ± 0.0000 | 0.8529 | 0.6762 | 0.3914 | 169 | 0 |
| `A1_neural_cap` | 0.6235 ± 0.0170 | 0.8361 | 0.6546 | 0.3629 | 169 | 0 |
| `A2_neural_symbolic` | 0.6082 ± 0.0171 | 0.8178 | 0.6397 | 0.3444 | 169 | 0 |
| `A3_neural_symbolic_probabilistic` | 0.6073 ± 0.0172 | 0.8167 | 0.6386 | 0.3450 | 169 | 0 |
| `A4_full_with_memory` | 0.5936 ± 0.0167 | 0.8019 | 0.6257 | 0.3243 | 169 | 0 |
| `A5_same_fact_neural` | 0.5243 ± 0.0069 | 0.7170 | 0.5558 | 0.2602 | 169 | 0 |

Per-seed values: `metrics/per_seed.csv`. Paired per-request differences vs the reference arm: `metrics/summary.json` (`paired`).

![Figure 3](plots/fig03_trajectories.png)

![Figure 4](plots/fig04_metric_by_arm.png)

![Figure 5](plots/fig05_error_vs_transition.png)

*Figures 3-5: trajectories on held-out intervals, aggregate error by arm with seed dispersion, error versus distance to regime change.* No superiority claim: differences are within a toy, single-dataset setting.

## 8. Probabilistic reasoning results

| arm | seed | n | Brier | NLL | accuracy | semantics |
|---|---|---|---|---|---|---|
| `A3_neural_symbolic_probabilistic` | 0 | 169 | 0.143 | 0.431 | 0.811 | exact_posterior |
| `A3_neural_symbolic_probabilistic` | 1 | 169 | 0.143 | 0.431 | 0.811 | exact_posterior |
| `A3_neural_symbolic_probabilistic` | 2 | 169 | 0.143 | 0.431 | 0.811 | exact_posterior |
| `A4_full_with_memory` | 0 | 169 | 0.143 | 0.431 | 0.811 | exact_posterior |
| `A4_full_with_memory` | 1 | 169 | 0.143 | 0.431 | 0.811 | exact_posterior |
| `A4_full_with_memory` | 2 | 169 | 0.143 | 0.431 | 0.811 | exact_posterior |

The reasoner consumed only the eight semantic features (slopes, volatilities, cross-lag correlations, neural cue, known-future event cue) computed from permitted evidence; emission parameters were fitted on the training split (`fit_regime_emissions.py`) using the hidden regime as a fitting label on allowed data. The hidden regime is never a predictor input.

![Figure 6](plots/fig06_regime_posterior.png)

## 9. Symbolic pathway results

```json
{
 "event_program": {
  "n": 169,
  "agreement_with_ground_truth": 1.0,
  "known_fraction": 1.0
 },
 "slope_program": {
  "n": 169,
  "mean_abs_slope_y_normal": 0.10387418000101922,
  "mean_abs_slope_y_shift": 0.1800140949891405
 },
 "semantic_features_by_regime": {
  "slope_0": {
   "NORMAL": 0.010520918750041328,
   "SHIFT": -0.012696656621482524
  },
  "slope_1": {
   "NORMAL": 0.0014147783157622098,
   "SHIFT": 0.01206326240708966
  },
  "volatility_0": {
   "NORMAL": 0.2833106637904408,
   "SHIFT": 0.4358862239760088
  },
  "volatility_1": {
   "NORMAL": 0.3208223751571537,
   "SHIFT": 0.2920354389658638
  },
  "xcorr_lag1": {
   "NORMAL": 0.5839930984168991,
   "SHIFT": -0.519951206765412
  },
  "xcorr_lag2": {
   "NORMAL": 0.6635569671260301,
   "SHIFT": -0.4670810164084324
  },
  "neural_cue": {
   "NORMAL": -0.000797722022632291,
   "SHIFT": 0.0008435627275274963
  },
  "event_cue": {
   "NORMAL": 0.14156626506024098,
   "SHIFT": 0.10465116279069768
  }
 }
}
```

![Figure 8](plots/fig08_symbolic.png)

Programs (`window_slope_8`, `event_active_at_origin_h4`) run through the reference interpreter with fuel budgets; outputs carry knownness (missing windows are unknown, never 0) and `ExecutionTrace`s (`traces/*_evidence.csv`, kind `execution_trace`). Whether the program changed the cap's prediction is visible as the A1→A2 paired difference in `metrics/summary.json`.

## 10. Memory results

```json
{
 "retrievals": 1083,
 "fraction_no_analogue": 0.0110803324099723,
 "mean_age": 229.63483146067415,
 "median_age": 127.0,
 "min_age": 4,
 "truncated": 0,
 "phases": [
  "test",
  "train"
 ]
}
```

![Figure 9](plots/fig09_memory_retrieval.png)

Retrieval operates on a pinned view filtered by availability *before* ranking; the learner appends matured episodes only between updates and publishes a new snapshot; in-flight predictions keep their view (T39). No benefit is claimed where the A3→A4 difference is within seed dispersion.

## 11. Training dynamics

![Figure 7](plots/fig07_training_curves.png)

Non-finite states, clipping fallbacks or failed runs: none recorded.

## 12. Modularity / substitution proof

| id | description | validate | predict | causal | train | restore | resolved plugins |
|---|---|---|---|---|---|---|---|
| `S01_lower_provider_conv` | lower provider A (transformer) -> provider B (causal conv) | pass | pass | pass | pass | pass | base_ts: toy_frozen_conv@1.0.0, upper: tiny_upper_transformer@1.0.0, cap: residual_point_cap@1.0.0 |
| `S02_upper_mlp` | upper transformer -> MLP upper module | pass | pass | pass | pass | pass | base_ts: toy_frozen_transformer@1.0.0, upper: tiny_upper_mlp@1.0.0, cap: residual_point_cap@1.0.0 |
| `S03a_semantic_raw_only` | semantic adapter reads raw input only | pass | pass | pass | pass | pass | base_ts: toy_frozen_transformer@1.0.0, semantic: simple_temporal_semantics@1.0.0, upper: tiny_upper_transformer@1.0.0, cap: residual_point_cap@1.0.0 |
| `S03b_semantic_base_final_only` | semantic adapter reads raw (required) + base final state instead of the upper state | pass | pass | pass | pass | pass | base_ts: toy_frozen_transformer@1.0.0, semantic: simple_temporal_semantics@1.0.0, upper: tiny_upper_transformer@1.0.0, cap: residual_point_cap@1.0.0 |
| `S03c_semantic_raw_plus_final` | semantic adapter reads raw + base final | pass | pass | pass | pass | pass | base_ts: toy_frozen_transformer@1.0.0, semantic: simple_temporal_semantics@1.0.0, upper: tiny_upper_transformer@1.0.0, cap: residual_point_cap@1.0.0 |
| `S03d_semantic_raw_early_final` | semantic adapter reads raw + early (neural port) + final (base_context port) | pass | pass | pass | pass | pass | base_ts: toy_frozen_transformer@1.0.0, semantic: simple_temporal_semantics@1.0.0, upper: tiny_upper_transformer@1.0.0, cap: residual_point_cap@1.0.0 |
| `S04_residual_off` | early-state residual into the upper module disabled | pass | pass | pass | pass | pass | base_ts: toy_frozen_transformer@1.0.0, upper: tiny_upper_transformer@1.0.0, cap: residual_point_cap@1.0.0 |
| `S05_rms_norm` | boundary normalisation layer_norm -> rms_norm | pass | pass | pass | pass | pass | base_ts: toy_frozen_transformer@1.0.0, upper: tiny_upper_transformer@1.0.0, cap: residual_point_cap@1.0.0 |
| `S06_programs_disabled` | symbolic programs removed (semantic adapter kept) | pass | pass | pass | pass | pass | base_ts: toy_frozen_transformer@1.0.0, upper: tiny_upper_transformer@1.0.0, semantic: simple_temporal_semantics@1.0.0, cap: residual_point_cap@1.0.0 |
| `S07_trivial_reasoner` | exact reasoner -> deterministic threshold reasoner (external plugin) | pass | pass | pass | pass | pass | base_ts: toy_frozen_transformer@1.0.0, event_program: typed_program@1.0.0, slope_program: typed_program@1.0.0, upper: tiny_upper_transformer@1.0.0, semantic: simple_temporal_semantics@1.0.0, regime: deterministic_threshold_reasoner@0.1.0, cap: residual_point_cap@1.0.0 |
| `S08_memory_disabled` | memory removed from the full cap | pass | pass | pass | pass | pass | base_ts: toy_frozen_transformer@1.0.0, event_program: typed_program@1.0.0, slope_program: typed_program@1.0.0, upper: tiny_upper_transformer@1.0.0, semantic: simple_temporal_semantics@1.0.0, regime: exact_finite_regime_reasoner@1.0.0, cap: residual_point_cap@1.0.0 |
| `S09_writer_consumer_changed` | bounded writer (external plugin) + MLP consumer | pass | pass | pass | pass | pass | base_ts: toy_frozen_transformer@1.0.0, upper: tiny_upper_transformer@1.0.0, cap: residual_point_cap@1.0.0 |

No substitution required editing core scheduler/runtime code: all rows ran against core source hash `cbf471e80120cf9e64c15d1b0725456acb0ef0d3e4af3d0448722d47d067d378` (`environment/core_source_hash.json`).

![Figure 10](plots/fig10_substitution_matrix.png)

## 13. Checkpoint / reproducibility proof

| arm@seed | manifest | checkpoint | restore n | max abs diff | manifest match |
|---|---|---|---|---|---|
| `A0_frozen_baseline@s0` | `3b62b8aff9ba` | `3b62b8aff9ba` | 16 | 0.000 | True |
| `A0_frozen_baseline@s1` | `1c466c2d7c7f` | `1c466c2d7c7f` | 16 | 0.000 | True |
| `A0_frozen_baseline@s2` | `765b82f8425a` | `765b82f8425a` | 16 | 0.000 | True |
| `A1_neural_cap@s0` | `9aa59f1dbe53` | `9aa59f1dbe53` | 16 | 0.000 | True |
| `A1_neural_cap@s1` | `e5f507d01da2` | `e5f507d01da2` | 16 | 0.000 | True |
| `A1_neural_cap@s2` | `6f621c206e5e` | `6f621c206e5e` | 16 | 0.000 | True |
| `A2_neural_symbolic@s0` | `82662e6a37cb` | `82662e6a37cb` | 16 | 0.000 | True |
| `A2_neural_symbolic@s1` | `c9632ff72766` | `c9632ff72766` | 16 | 0.000 | True |
| `A2_neural_symbolic@s2` | `7851a7c14a0b` | `7851a7c14a0b` | 16 | 0.000 | True |
| `A3_neural_symbolic_probabilistic@s0` | `73dd98d198b0` | `73dd98d198b0` | 16 | 0.000 | True |
| `A3_neural_symbolic_probabilistic@s1` | `a7626a9f26a3` | `a7626a9f26a3` | 16 | 0.000 | True |
| `A3_neural_symbolic_probabilistic@s2` | `8ca748459e27` | `8ca748459e27` | 16 | 0.000 | True |
| `A4_full_with_memory@s0` | `b16cc54125b0` | `b16cc54125b0` | 16 | 0.000 | True |
| `A4_full_with_memory@s1` | `bd10a0b38a94` | `bd10a0b38a94` | 16 | 0.000 | True |
| `A4_full_with_memory@s2` | `f19b4ceafe5d` | `f19b4ceafe5d` | 16 | 0.000 | True |
| `A5_same_fact_neural@s0` | `014e141c827a` | `014e141c827a` | 16 | 0.000 | True |
| `A5_same_fact_neural@s1` | `650fdf4e2a9d` | `650fdf4e2a9d` | 16 | 0.000 | True |
| `A5_same_fact_neural@s2` | `5b6dc76f1ae2` | `5b6dc76f1ae2` | 16 | 0.000 | True |

![Figure 11](plots/fig11_checkpoint_restore.png)

Reproduce: see §17 commands; checkpoints under `checkpoints/` (content-addressed blobs + manifests), inspect with `ness inspect <run>/checkpoints <manifest>`.

## 14. Cost and scaling observations

![Figure 12](plots/fig12_cost.png)

Toy-only components: the frozen providers (self-pretrained tiny transformer/conv), the in-memory snapshot store, the exact two-state reasoner. Intended to scale through the same interfaces: substrate ports (external frozen providers), the differentiable region (batched/sharded), FabricPC workspaces, plugin memory stores. No large-scale performance is extrapolated.

## 15. Tests and acceptance status

See `docs/ARCHITECTURE.md` §10 for the T01-T44 map; this run additionally exercises T04/T38 (causal check per arm), T08/T27/T40 (restore), T31/T33-T35 (substitutions), T39 (memory isolation in A4), T43 (resolved wiring in manifests). Not applicable at toy scale: T10, T12, T14, T20, T25-T29, T32.

## 16. Problems discovered and architecture decisions

* Emission parameters of the exact reasoner are a fitted asset (train split) and are recorded in the config; a fitted-state contract (ADR) would make this a first-class plugin state rather than configuration.
* The cap lowers unknown feature entries to 0 and ignores knownness masks (declared capability); a mask-aware consumer is a plugin change.
* Per-request tracing under `jax.value_and_grad` dominates training time at this scale; the batched/data-parallel path exists for uniform shapes.

## 17. Reproduction and next steps

```bash
ness validate examples/vertical_slice_timeseries/configs/toy_vertical_slice.yaml
ness run examples/vertical_slice_timeseries/configs/toy_vertical_slice.yaml --out runs/toy_vertical_slice/20260926-220302-final
ness evaluate runs/toy_vertical_slice/20260926-220302-final
ness report runs/toy_vertical_slice/20260926-220302-final
ness inspect runs/toy_vertical_slice/20260926-220302-final/checkpoints <manifest-id>
```

Next integrations (smallest steps): a TimesFM-3 forecast-only substrate behind the same `forecast/point|quantiles` ports; a frozen text+vision substrate pair with alignment evidence; a Hyperon-backed `MemoryStore`; a NumPyro reasoner plugin compared against `exact_finite_regime_reasoner`; larger frozen-provider runs on the FabricPC backend (see `docs/FABRICPC_BACKEND.md`).

## Artifact index

| artifact | path |
|---|---|
| configuration | `config/experiment.yaml` |
| dataset series | `data/series.csv` |
| generation manifest | `data/generation_manifest.json` |
| splits | `data/splits.json` |
| ground truth per origin | `data/ground_truth.csv` |
| predictions per arm/seed | `predictions/*.csv` |
| evidence traces | `traces/*_evidence.csv` |
| metrics summary | `metrics/summary.json, metrics/summary.csv, metrics/aggregate_by_arm.csv, metrics/per_seed.csv` |
| training history | `metrics/train_history.csv` |
| timing | `metrics/timing.csv` |
| checkpoint restore | `metrics/checkpoint_restore_*.csv` |
| substitution matrix | `metrics/substitution_matrix.json|csv` |
| regime/memory/symbolic diagnostics | `metrics/regime_diagnostics.csv, metrics/memory_diagnostics.json, metrics/symbolic_diagnostics.json` |
| manifests | `manifests/*.json` |
| checkpoints | `checkpoints/` |
| environment | `environment/*.json` |
| log | `logs/run.log` |
| plots | `plots/*.png|svg` |