import numpy as np
import pytest

from ness.contracts import (
    AccessPolicy, AvailabilityCut, CategoricalForecast, ContractViolation, CoordinateSchema, EvidenceGraph, FeatureEvidence, FieldRole,
    HypothesisEvidence, Observation, ObservationBundle, PointForecast, PredictionSpace, ProducerRef, Provenance, QuantileForecast,
    StateSnapshot, UnsupportedCapability, AccessPolicyViolation, content_hash, horizon_schema, quantile_schema, series_schema,
)

H = horizon_schema(("y0", "y1"), (1, 2, 3, 4))
Q = quantile_schema(("y0", "y1"), (1, 2, 3, 4), (0.1, 0.5, 0.9))
PROV = Provenance(ProducerRef("n", "p", "1.0.0"))


class TestForecastAlgebra:  # T18 forecast capability algebra
    def test_point_has_no_quantiles_or_density(self):
        pf = PointForecast(np.zeros((4, 2)), H)
        with pytest.raises(UnsupportedCapability):
            pf.quantiles((0.5,))
        with pytest.raises(UnsupportedCapability):
            pf.log_prob(np.zeros((4, 2)))

    def test_quantile_has_no_log_prob_and_no_interpolation(self):
        qf = QuantileForecast(np.sort(np.random.default_rng(0).normal(size=(4, 2, 3)), axis=-1), Q, (0.1, 0.5, 0.9))
        with pytest.raises(UnsupportedCapability):
            qf.log_prob(np.zeros((4, 2)))
        with pytest.raises(UnsupportedCapability):
            qf.quantiles((0.25,))
        assert qf.quantiles((0.5,)).shape == (4, 2, 1)
        assert qf.point().shape == (4, 2)

    def test_quantile_levels_must_increase_and_match(self):
        with pytest.raises(ContractViolation):
            QuantileForecast(np.zeros((4, 2, 3)), Q, (0.9, 0.5, 0.1))
        with pytest.raises(ContractViolation):
            QuantileForecast(np.zeros((4, 2, 2)), Q, (0.1, 0.9))

    def test_categorical_requires_normalisation(self):
        cs = CoordinateSchema("vocab[4]", ("class",))
        with pytest.raises(ContractViolation):
            CategoricalForecast(np.log(np.array([0.5, 0.5, 0.5, 0.5])), cs, "v1")
        CategoricalForecast(np.log(np.full(4, 0.25)), cs, "v1")

    def test_prediction_space_rejects_wrong_type_and_coordinates(self):
        space = PredictionSpace("s", "point", H, "mse")
        space.validate(PointForecast(np.zeros((4, 2)), H))
        with pytest.raises(ContractViolation):
            space.validate(QuantileForecast(np.zeros((4, 2, 3)), Q, (0.1, 0.5, 0.9)))
        other = horizon_schema(("a", "b"), (1, 2, 3, 4))
        with pytest.raises(ContractViolation):
            space.validate(PointForecast(np.zeros((4, 2)), other))

    def test_non_finite_forecast_rejected(self):
        with pytest.raises(ContractViolation):
            PointForecast(np.full((4, 2), np.nan), H)


class TestEvidence:
    def test_hypothesis_semantics_validation(self):
        HypothesisEvidence("h", PROV, "x", alternatives=("a", "b"), weights=np.array([0.3, 0.7]), weight_semantics="exact_posterior")
        with pytest.raises(ContractViolation):
            HypothesisEvidence("h", PROV, "x", alternatives=("a", "b"), weights=np.array([0.3, 0.3]), weight_semantics="exact_posterior")
        with pytest.raises(ContractViolation):  # exact posterior cannot declare omitted mass
            HypothesisEvidence("h", PROV, "x", alternatives=("a", "b"), weights=np.array([0.5, 0.5]), weight_semantics="exact_posterior", omitted_mass=0.1)
        HypothesisEvidence("h", PROV, "x", alternatives=("a", "b"), weights=np.array([2.0, 5.0]), weight_semantics="score")

    def test_feature_knownness_is_separate_from_value(self):
        fe = FeatureEvidence("f", PROV, "x", value=np.array([1.0, np.nan]), knownness=np.array([True, False]))
        np.testing.assert_array_equal(fe.dense(), [1.0, 0.0])
        with pytest.raises(ContractViolation):
            FeatureEvidence("f", PROV, "x", value=np.array([1.0, np.nan]))  # unknown NaN marked known

    def test_evidence_graph_is_append_only_and_checks_dependencies(self):
        g = EvidenceGraph("g")
        a = FeatureEvidence("a", PROV, "x", value=np.zeros(1))
        b = FeatureEvidence("b", Provenance(ProducerRef("n", "p", "1"), ("a",)), "x", value=np.zeros(1))
        g2 = g.with_fragment((a,))
        assert g.ids() == () and g2.ids() == ("a",)
        with pytest.raises(ContractViolation):
            g.with_fragment((b,))  # dependency 'a' unknown in g
        g3 = g2.with_fragment((b,))
        assert g3.edges() == (("a", "b"),)
        assert g2.version_hash() != g3.version_hash()


class TestObservationsAndAccess:
    def _bundle(self):
        s = series_schema(("y0", "y1"))
        obs = (
            Observation("o1", "history", "ts", FieldRole.OBSERVED, s, np.zeros((8, 2)), 8),
            Observation("o2", "future", "ts", FieldRole.OUTCOME_ONLY, H, np.zeros((4, 2)), 12),
            Observation("o3", "late", "ts", FieldRole.OBSERVED, s, np.zeros((8, 2)), 9),  # arrives after origin
            Observation("o4", "cal_future", "cal", FieldRole.KNOWN_FUTURE, H, np.zeros((4, 2)), 8),
        )
        return ObservationBundle("b", obs, 8)

    def test_policy_cannot_allow_outcome_roles(self):
        with pytest.raises(AccessPolicyViolation):
            AccessPolicy("p", frozenset({FieldRole.OBSERVED, FieldRole.OUTCOME_ONLY}), AvailabilityCut(8))

    def test_select_removes_forbidden_and_unavailable(self):  # T19/T44 core semantics
        b = self._bundle()
        pol = AccessPolicy("p", frozenset({FieldRole.OBSERVED, FieldRole.KNOWN_FUTURE}), AvailabilityCut(8))
        view = b.select(pol)
        assert set(view.field_names()) == {"history", "cal_future"}
        view.assert_prediction_safe()
        with pytest.raises(AccessPolicyViolation):
            b.assert_prediction_safe()

    def test_observation_rank_must_match_axes(self):
        with pytest.raises(ContractViolation):
            Observation("o", "f", "ts", FieldRole.OBSERVED, series_schema(("y0",)), np.zeros(8), 8)


class TestHashingAndState:
    def test_content_hash_is_canonical(self):
        assert content_hash({"b": 1, "a": [1, 2]}) == content_hash({"a": [1, 2], "b": 1})
        assert content_hash(np.zeros(3)) != content_hash(np.zeros(4))

    def test_state_snapshot_rejects_object_arrays(self):
        with pytest.raises(ContractViolation):
            StateSnapshot("p", "1.0.0", "s", {"x": np.array([object()])})
