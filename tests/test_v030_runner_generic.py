"""v0.3.0 SHOULD/COULD items: runner and artifacts generic over forecast types (N-03/N-08,
ADR-0017), opt-in retention for frozen evaluation (N-07, first half), and the categorical
forecast fixes (B-1, dtype-aware normalisation)."""

import numpy as np
import pytest

from ness.contracts import CategoricalForecast, ContractViolation, CoordinateSchema, PointForecast, ScoreResult, TrainingOutcome
from ness.experiments.artifacts import ArtifactWriter

from ness_test_helpers import BASE, TASKS, arm, experiment, needs_jax

POS = CoordinateSchema("pos[2]", ("position",))
LOGITS = CoordinateSchema("pos[2]xvocab[5]", ("position", "class"))


class _Req:
    request_id, origin, group = "r@1", 1, {"stream": "task"}
    queries = (type("Q", (), {"query_id": "q/tok", "task_id": "tok"})(),)


class _TokTask:
    def row_metrics(self, forecast, outcome):
        lp = forecast.values - np.log(np.sum(np.exp(forecast.values), axis=-1, keepdims=True))
        ids = outcome.values.astype(int)
        return {"nll": float(-np.mean(lp[np.arange(len(ids)), ids])), "top1": float(np.mean(np.argmax(lp, -1) == ids))}


def test_N03_non_elementwise_forecasts_get_score_and_task_metrics_not_index_errors():
    logits = PointForecast(np.array([[2.0, 0, 0, 0, 0], [0, 0, 3.0, 0, 0]]), LOGITS, functional="logits")
    oc = TrainingOutcome("r@1", "q/tok", np.array([0.0, 1.0]), np.array([True, True]), 1, 2, POS)
    row = ArtifactWriter.prediction_row("A", 0, "test", _Req(), {"q/tok": logits}, {"tok": oc}, {"tok": ScoreResult(3.0, 2.0)}, {"tok": _TokTask()})
    assert row["tok/score"] == 1.5 and row["tok/score_num"] == 3.0 and row["tok/forecast_type"] == "point"
    assert set(row) >= {"tok/nll", "tok/top1"} and not any("/pred_" in k or "/mse" in k for k in row)
    cat = CategoricalForecast(np.log(np.full((2, 5), 0.2)), LOGITS, "v5")
    row = ArtifactWriter.prediction_row("A", 0, "test", _Req(), {"q/tok": cat}, {"tok": oc}, {"tok": ScoreResult(1.0, 2.0)})
    assert row["tok/score"] == 0.5 and row["tok/forecast_type"] == "categorical"


def test_N03_point_rows_keep_their_v02_columns():
    fc = PointForecast(np.array([[1.0, 2.0]]), CoordinateSchema("h", ("horizon", "channel")))
    oc = TrainingOutcome("r@1", "q/tok", np.array([[1.5, 2.0]]), np.array([[True, True]]), 1, 2, CoordinateSchema("h", ("horizon", "channel")))
    row = ArtifactWriter.prediction_row("A", 0, "test", _Req(), {"q/tok": fc}, {"tok": oc}, {"tok": ScoreResult(0.25, 2.0)})
    assert [k for k in row if k.startswith("tok/")] == ["tok/pred_0", "tok/pred_1", "tok/truth_0", "tok/abs_err_0", "tok/truth_1", "tok/abs_err_1",
                                                         "tok/mae", "tok/mse", "tok/score_num", "tok/score_den"]


def test_categorical_sample_draws_every_row_and_tolerance_follows_dtype():
    lp = np.log(np.array([[1.0, 0, 0], [0, 0, 1.0]]) + 1e-300)
    lp = lp - np.log(np.sum(np.exp(lp), -1, keepdims=True))
    fc = CategoricalForecast(lp, CoordinateSchema("pos[2]xv3", ("position", "class")), "v3")
    s = fc.sample(4, np.random.default_rng(0))
    assert s.shape == (4, 2) and (s[:, 0] == 0).all() and (s[:, 1] == 2).all()      # B-1: the second row is sampled too
    one = CategoricalForecast(np.log(np.array([0.5, 0.5])), CoordinateSchema("v2", ("class",)), "v2")
    assert one.sample(3, np.random.default_rng(1)).shape == (3,)
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 50257)).astype(np.float32)
    lp32 = x - np.log(np.sum(np.exp(x), -1, keepdims=True))                            # float32 log-softmax, 50k classes
    CategoricalForecast(lp32.astype(np.float32), CoordinateSchema("big", ("position", "class")), "big")
    with pytest.raises(ContractViolation, match="normalised"):
        CategoricalForecast(np.log(np.array([0.5, 0.49])), CoordinateSchema("v2", ("class",)), "v2")


@needs_jax
def test_N07_outputs_retention_keeps_frozen_evaluation_flat(registry, scenario):
    from ness.runtime.system import NessSystem
    from ness.runtime.transaction import PredictionTransaction

    exp = experiment({"a": arm([BASE], output="substrate://base_ts/forecast/point", learning=None)})
    sysm = NessSystem.build(exp, exp.arm("a"), registry, scenario=scenario)
    full, flat = PredictionTransaction(sysm, "frozen"), PredictionTransaction(sysm, "frozen", retention="outputs")
    reqs = list(scenario.iter_requests("test", tuple(sysm.tasks.values()), {"max_requests": 6}))
    for req in reqs:
        for tx in (full, flat):
            rec = tx.predict(req)
            tx.reveal(req.request_id, {q.task_id: scenario.outcome(req.request_id, q.query_id) for q in req.queries})
    assert len(full._pending) == len(reqs) and len(flat._pending) == 0                    # E1: O(1) execution records
    assert full.aggregate_scores() == flat.aggregate_scores()                             # E3: scores unchanged
    for req in reqs:
        np.testing.assert_array_equal(full.records[req.request_id].outputs[req.queries[0].query_id].dense(),
                                      flat.records[req.request_id].outputs[req.queries[0].query_id].dense())
    with pytest.raises(Exception, match="retention"):
        PredictionTransaction(sysm, "frozen", retention="none")


@needs_jax
def test_N07_runner_retention_and_optional_ground_truth(registry, tmp_path):
    from ness.experiments import run_experiment

    exp = experiment({"base": arm([BASE], output="substrate://base_ts/forecast/point", learning=None)},
                     protocol={"train_split": "train", "test_split": "test", "batch_size": 4, "updates": 1, "max_train_requests": 4, "max_test_requests": 4,
                               "retention": "outputs", "artifacts": {"ground_truth": False}, "restore_check_requests": 2})
    rep = run_experiment(exp, registry, tmp_path)
    r = next(iter(rep.arms.values()))
    assert r.status == "ok" and r.restore_check["max_abs_diff"] == 0.0 and r.causal_check == "pass"   # E2: checks still work
    assert not (tmp_path / "data" / "ground_truth.csv").exists() and (tmp_path / "data" / "splits.json").exists()
