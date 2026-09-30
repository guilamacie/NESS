# ADR-0017: Runner, artifacts and checks generic over forecast types; retention

**Decision**
* Restore, causal and substitution checks compare `forecast.dense()` (every forecast type has a
  documented dense lowering), not `point()`.
* Per-request rows keep the v0.2 element-wise columns (`pred_i`, `truth_i`, `abs_err_i`, `mae`,
  `mse`) when the forecast has a point of the outcome's size. Otherwise (logits over classes
  against integer ids, categorical forecasts) a row carries `forecast_type`, `score_num`,
  `score_den`, `score` (the task's `ScoreResult.value`) and the scalars of the task's optional
  `row_metrics(forecast, outcome)` hook.
* Paired contrasts use per-request `mse` when both arms have it, else `score` (recorded as
  `metric: score`).
* `protocol.artifacts.ground_truth: false` skips the ground-truth dump.
* `PredictionTransaction(..., retention="outputs")` (runner: `protocol.retention: outputs`) drops
  a frozen prediction's execution record once its outcome is revealed, so frozen evaluation keeps
  O(1) execution records; the prediction record and scores are kept. Evidence traces of the test
  phase are then not written. `full` (default) is unchanged.

**Not decided here** Preserving producer dtypes instead of casting port payloads to float64 (the
second half of N-07) changes numerics and is deferred.

**Tests** `tests/test_v030_runner_generic.py`; the instance's tiny token configuration runs
through `ness run` with every arm ok (it failed with `IndexError` in v0.2.1).
