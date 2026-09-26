import numpy as np
import pytest

from ness.contracts import ContractViolation, QuantileForecast, TrainingOutcome, quantile_schema
from ness.plugin_api.testkit import ScenarioContractMixin, WriterContractMixin
from ness.reference_plugins.writers_consumers import MonotoneQuantileWriter, PointResidualWriter
from ness.scenarios import ToyTemporalDataset, chronological_split, rolling_origins
from ness.tasks import PointForecastTask, QuantileForecastTask


class TestToyScenarioContract(ScenarioContractMixin):  # T44
    def make_scenario(self):
        return ToyTemporalDataset({"n_steps": 200, "context": 16, "horizon": 4, "seed": 1})

    def make_tasks(self):
        return (PointForecastTask({"task_id": "future_value", "horizons": 4, "channels": ["y0", "y1"]}),)


class TestPointWriter(WriterContractMixin):
    def make_writer(self):
        return PointResidualWriter()


class TestMonotoneQuantileWriter(WriterContractMixin):  # T18
    def make_writer(self):
        return MonotoneQuantileWriter()

    def test_zero_native_gap_cannot_be_opened(self):
        w = MonotoneQuantileWriter()
        b = np.zeros((1, 1, 3))
        out = w.write(b, np.array([0.0, 5.0, 5.0]), np)
        np.testing.assert_allclose(out, 0.0)


def test_rolling_origin_split_has_gap():
    o = rolling_origins(100, 10, 4)
    sp = chronological_split(o, 0.7, 4)
    assert sp["train"][-1] + 4 <= sp["test"][0]
    assert not set(sp["train"]) & set(sp["test"])


def test_T07_point_loss_masks_and_padding():
    t = PointForecastTask({"task_id": "t", "horizons": 4, "channels": ["y0", "y1"]})
    y = np.arange(8.0).reshape(4, 2)
    pred = np.zeros((4, 2))
    mask = np.ones((4, 2)); mask[2] = 0  # a delayed/padded coordinate contributes nothing
    n, d = t.loss_terms(pred, y, mask, np)
    assert d == 6 and n == float((y**2 * mask).sum())
    oc = TrainingOutcome("r", "q", y, mask.astype(bool), 1, 1, t.prediction_space().target)
    from ness.contracts import PointForecast
    sc = t.score(PointForecast(pred, t.prediction_space().target), oc)
    assert np.isclose(sc.value, n / d) and sc.diagnostics["n_valid"] == 6


def test_pinball_matches_manual_and_diagnostics():
    t = QuantileForecastTask({"task_id": "q", "horizons": 2, "channels": ["y0"], "levels": [0.1, 0.5, 0.9]})
    q = np.array([[[-1.0, 0.0, 1.0]], [[0.0, 1.0, 2.0]]])
    y = np.array([[0.5], [3.0]])
    n, d = t.loss_terms(q, y, np.ones((2, 1)), np)
    taus = np.array([0.1, 0.5, 0.9])
    u = y[..., None] - q
    manual = (u * (taus - (u < 0))).sum()
    assert np.isclose(n, manual) and d == 6
    fc = QuantileForecast(q, t.prediction_space().target, (0.1, 0.5, 0.9))
    sc = t.score(fc, TrainingOutcome("r", "q", y, np.ones((2, 1), bool), 1, 1, t.prediction_space().target))
    assert sc.diagnostics["coverage"] == 0.5 and sc.diagnostics["monotone"] == 1.0


def test_quantile_task_rejects_mismatched_levels():
    t = QuantileForecastTask({"task_id": "q", "horizons": 2, "channels": ["y0"], "levels": [0.1, 0.5, 0.9]})
    other = quantile_schema(("y0",), (1, 2), (0.25, 0.75))
    with pytest.raises(ContractViolation):
        t.prediction_space().validate(QuantileForecast(np.zeros((2, 1, 2)), other, (0.25, 0.75)))
