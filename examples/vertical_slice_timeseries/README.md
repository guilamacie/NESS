# Toy vertical slice

Two configurations live here:

* `configs/toy_vertical_slice.yaml` - **the reference run** required by the toy-slice
  requirements: interpretable regime dataset (`toy_regime_dataset`: target `y`, auxiliary `x`,
  known event `e`, hidden two-state regime `r` with recorded change points), arms A0-A5, three
  model seeds, ten configuration-only substitution proofs, in-run checkpoint publish + restore
  comparison, causal checks, structured raw artifacts and a generated report.
* `configs/vertical_slice.yaml` - the first-cycle six-arm slice on `toy_temporal_dataset`
  (kept for the test suite and as a second example).

| arm | what changes |
|---|---|
| `A0_frozen_baseline` | frozen tiny transformer's native point forecast only |
| `A1_neural_cap` | + trainable upper transformer (final + projected early residual, gated add, layer norm) + residual point cap |
| `A2_neural_symbolic` | + semantic adapter (raw series, upper state, known-future events) + typed programs `window_slope_8`, `event_active_at_origin_h4` |
| `A3_neural_symbolic_probabilistic` | + exact two-state regime reasoner (posterior `HypothesisEvidence`) |
| `A4_full_with_memory` | + pinned analogue memory (matured episodes only; learner appends between updates) |
| `A5_same_fact_neural` | same permitted raw facts (32x2 window, events) fed directly to the cap; no symbolic computation |

## Reproduce

```bash
pip install -e ".[dev]"                       # jax + matplotlib + pandas
pip install -e examples/external_plugin_example   # S07/S09 use its reasoner and writer
ness validate examples/vertical_slice_timeseries/configs/toy_vertical_slice.yaml
ness run examples/vertical_slice_timeseries/configs/toy_vertical_slice.yaml --out runs/toy_vertical_slice/<run_id>
ness evaluate runs/toy_vertical_slice/<run_id>          # summary table
ness report   runs/toy_vertical_slice/<run_id>          # metrics/, plots/fig01..fig12, report/toy_vertical_slice_report.md
ness inspect  runs/toy_vertical_slice/<run_id>/checkpoints <manifest-id>
python examples/vertical_slice_timeseries/fit_regime_emissions.py   # re-derive the reasoner emission parameters (train split)
python reports/fill_report.py runs/toy_vertical_slice/<run_id> --pdf --test-status "<pytest result>"   # LaTeX/PDF write-up from the saved artifacts
```

Run directory layout: `config/ manifests/ checkpoints/ data/ predictions/ metrics/ traces/ plots/
logs/ environment/ report/` (see `report/toy_vertical_slice_report.md` §Artifact index for the
meaning of every file). Every figure is regenerated from the saved CSV/JSON by `ness report`.
The LaTeX/PDF write-up of the reference run is in `reports/` (`toy_vertical_slice_report.tex`/`.pdf`,
filled from `reports/toy_vertical_slice_report.tex.in` by `reports/fill_report.py`; `reports/SOURCE_RUN.txt`
names the run it was filled from). Every arm, frozen or trainable, is fed the same
`batch_size x updates` training requests, so train-phase metrics are comparable across arms.
