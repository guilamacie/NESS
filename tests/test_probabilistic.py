import numpy as np
import pytest

from ness.contracts import ApproximationStatus, ContractViolation
from ness.plugin_api.testkit import ReasonerContractMixin, make_context, port_value
from ness.probabilistic import ExactFiniteRegimeReasoner, FiniteLatentModel, ImportanceSamplingRegimeReasoner

CFG = {"model": {"states": ["a", "b"], "prior": [0.4, 0.6], "emission": {"means": [[0, 0, 0], [1, 1, 1]], "scales": [[1, 1, 1], [1, 1, 1]]}},
       "feature_ports": {"x": 2, "y": 1}}


class TestExactReasonerContract(ReasonerContractMixin):
    def make_module(self):
        return ExactFiniteRegimeReasoner(CFG)


class TestISReasonerContract(ReasonerContractMixin):
    def make_module(self):
        return ImportanceSamplingRegimeReasoner({**CFG, "samples": 2000})


def _inputs():
    return {"x": port_value(np.array([0.7, 0.4]), "x"), "y": port_value(np.array([0.9]), "y")}


def test_T37_reasoner_substitution_agrees_within_tolerance_with_explicit_diagnostics():
    ex = ExactFiniteRegimeReasoner(CFG)
    im = ImportanceSamplingRegimeReasoner({**CFG, "samples": 20000})
    o1 = ex.forward(_inputs(), ex.initialize(np.random.default_rng(0)), make_context(seed=0))
    o2 = im.forward(_inputs(), im.initialize(np.random.default_rng(0)), make_context(seed=0))
    p1, p2 = o1.ports["posterior"], o2.ports["posterior"]
    assert p1.weight_semantics == "exact_posterior" and p1.approximation == ApproximationStatus.EXACT
    assert p2.weight_semantics == "approximate_posterior" and p2.approximation == ApproximationStatus.APPROXIMATE
    np.testing.assert_allclose(p1.weights, p2.weights, atol=0.02)
    assert "effective_sample_size" in o2.diagnostics and "effective_sample_size" not in o1.diagnostics
    # analytic check of the exact posterior
    m = FiniteLatentModel.from_config(CFG["model"])
    x = np.array([0.7, 0.4, 0.9])
    lj = np.log(m.prior) + m.log_likelihood(x, np.ones(3, bool))
    np.testing.assert_allclose(p1.weights, np.exp(lj - lj.max()) / np.exp(lj - lj.max()).sum())


def test_missing_features_are_marginalised_not_zero_filled():
    ex = ExactFiniteRegimeReasoner(CFG)
    st = ex.initialize(np.random.default_rng(0))
    inp = _inputs()
    inp["y"] = port_value(np.array([123.0]), "y", mask=np.array([False]))  # unknown -> dropped from likelihood
    post = ex.forward(inp, st, make_context()).ports["posterior"].weights
    m = ex.model
    lj = np.log(m.prior) + m.log_likelihood(np.array([0.7, 0.4, 0.0]), np.array([True, True, False]))
    np.testing.assert_allclose(post, np.exp(lj - lj.max()) / np.exp(lj - lj.max()).sum())
    assert ex.forward(inp, st, make_context()).evidence[0].details["known_features"] == 2


def test_importance_sampler_is_deterministic_under_request_rng():
    im = ImportanceSamplingRegimeReasoner({**CFG, "samples": 500})
    st = im.initialize(np.random.default_rng(0))
    a = im.forward(_inputs(), st, make_context(seed=5)).ports["posterior"].weights
    b = im.forward(_inputs(), st, make_context(seed=5)).ports["posterior"].weights
    c = im.forward(_inputs(), st, make_context(seed=6)).ports["posterior"].weights
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_reasoner_config_validation():
    with pytest.raises(ContractViolation):
        ExactFiniteRegimeReasoner({**CFG, "feature_ports": {"x": 1}})  # dims do not match emission
    with pytest.raises(ContractViolation):
        FiniteLatentModel("z", ("a", "b"), np.array([0.5, 0.6]), np.zeros((2, 1)), np.ones((2, 1)))
