import numpy as np
import pytest

from ness.contracts import BudgetExceeded, Completeness, ContractViolation, ExecutionBudget, Knownness, TruthStatus
from ness.symbolic import Fact, ProgramIR, ReferenceInterpreter, RelationWorld

BODY = {"op": "exists", "value": {"op": "intersect", "items": [
    {"op": "follow", "root": "actor", "path": ["trusts"]},
    {"op": "follow", "root": "actor", "path": ["inverse_helps"]},
    {"op": "follow", "root": "object", "path": ["inside", "inverse_opens", "inverse_holds"]}]}}
TRUSTED_HELPER = ProgramIR.from_dict({"schema_version": "ness.program/1", "name": "trusted_helpful_key_holder", "historical_alias": "v43:681",
                                      "semantics": "typed_set_exists_v1", "inputs": {"actor": "person", "object": "object"}, "output": "truth_result", "body": BODY})
ALL = frozenset({"trusts", "helps", "holds", "opens", "inside"})
FACTS = (Fact("trusts", "ivo", "hana"), Fact("helps", "hana", "ivo"), Fact("holds", "hana", "key1"), Fact("opens", "key1", "chest"), Fact("inside", "gem", "chest"))


def run(facts, closed=ALL, budget=None):
    return ReferenceInterpreter().execute(TRUSTED_HELPER, {"actor": "ivo", "object": "gem"}, RelationWorld(facts, closed), budget)


class TestExactSemantics:  # T02 / T03
    def test_golden_shared_witness_true(self):
        r = run(FACTS)
        assert r.truth_status == TruthStatus.TRUE and r.completeness == Completeness.COMPLETE and r.status == "ok"
        assert any(f.relation == "holds" for f in r.witnesses)

    def test_swapping_shared_witness_flips_feature(self):
        facts = (Fact("trusts", "ivo", "hana"), Fact("helps", "ben", "ivo"), Fact("holds", "hana", "key1"), Fact("opens", "key1", "chest"), Fact("inside", "gem", "chest"))
        # every marginal relation still holds for *someone*, but no single H satisfies all
        assert run(facts).truth_status == TruthStatus.FALSE

    def test_incomplete_no_match_is_unknown_not_false(self):
        facts = (Fact("trusts", "ivo", "hana"), Fact("helps", "ben", "ivo"), Fact("holds", "hana", "key1"), Fact("opens", "key1", "chest"), Fact("inside", "gem", "chest"))
        r = run(facts, closed=frozenset({"trusts"}))
        assert r.truth_status == TruthStatus.UNKNOWN and r.completeness == Completeness.PARTIAL

    def test_witness_under_truncation_still_true(self):
        r = run(FACTS, closed=frozenset())  # nothing declared complete, yet a witness exists
        assert r.truth_status == TruthStatus.TRUE and r.completeness == Completeness.PARTIAL

    def test_identity_is_invariant_to_variable_renaming(self):
        renamed = ProgramIR("x", "typed_set_exists_v1", {"a": "person", "o": "object"}, "truth_result",
                            {"op": "exists", "value": {"op": "intersect", "items": [
                                {"op": "follow", "root": "a", "path": ["trusts"]}, {"op": "follow", "root": "a", "path": ["inverse_helps"]},
                                {"op": "follow", "root": "o", "path": ["inside", "inverse_opens", "inverse_holds"]}]}})
        assert renamed.program_hash == TRUSTED_HELPER.program_hash
        different = ProgramIR("y", "typed_set_exists_v1", {"a": "person", "o": "object"}, "truth_result",
                              {"op": "exists", "value": {"op": "follow", "root": "a", "path": ["trusts"]}})
        assert different.program_hash != TRUSTED_HELPER.program_hash


class TestFailClosed:  # T16
    def test_unknown_opcode_rejected_at_construction(self):
        with pytest.raises(ContractViolation):
            ProgramIR("bad", "s", {"a": "person"}, "truth_result", {"op": "eval_python", "code": "1"})

    def test_unknown_primitive_rejected(self):
        p = ProgramIR("bad", "s", {"s": "numeric_series"}, "numeric_vector", {"op": "call", "primitive": "os_system", "args": {}})
        with pytest.raises(ContractViolation):
            ReferenceInterpreter().execute(p, {"s": np.zeros((4, 1))})

    def test_undeclared_input_rejected(self):
        with pytest.raises(ContractViolation):
            ProgramIR("bad", "s", {"a": "person"}, "entity_set", {"op": "follow", "root": "zzz", "path": ["r"]})

    def test_fuel_exhaustion_is_explicit_censored_status(self):
        r = run(FACTS, budget=ExecutionBudget(fuel=3))
        assert r.status == "budget_exhausted" and r.truth_status == TruthStatus.ERROR and r.knownness == Knownness.UNAVAILABLE_BUDGET


class TestPrimitives:
    def test_window_slope_semantics_and_missingness(self):
        p = ProgramIR("slope", "typed_numeric_v1", {"series": "numeric_series", "window": "int"}, "numeric_vector",
                      {"op": "call", "primitive": "window_slope", "args": {"series": {"op": "input", "name": "series"}, "window": {"op": "input", "name": "window"}}})
        series = np.stack([2.0 * np.arange(10), -1.0 * np.arange(10)], axis=1)
        r = ReferenceInterpreter().execute(p, {"series": series, "window": 5})
        np.testing.assert_allclose(r.value, [2.0, -1.0])
        short = ReferenceInterpreter().execute(p, {"series": series[:3], "window": 5})
        assert short.knownness == Knownness.MISSING and short.status == "ok"
