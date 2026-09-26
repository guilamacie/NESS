"""FabricPC through NESS's public runtime path: build -> predict (target-free) -> record ->
reveal -> learn, for every FabricPC profile, plus hybrid credit and explicit failures."""

import copy
import sys

import numpy as np
import pytest

from ness_test_helpers import BASE, BASELINE_ARM, arm, build, experiment, first_requests  # noqa: E402

from ness.contracts import GradientBoundaryError, UnsupportedCapability, ValidationError
from ness.plugin_api.testkit import assert_target_free
from ness.runtime.transaction import PredictionTransaction

SEM = {"id": "semantic", "plugin": "simple_temporal_semantics", "config": {"window": 8, "channels": 2, "use_neural": False}, "inputs": {"raw": "observation://target_history"}}
INPUTS = {"baseline": "substrate://base_ts/forecast/point", "neural": {"from": "substrate://base_ts/state/final", "boundary": ["mean_pool"]}, "semantic": "semantic://semantic/features"}


def fcap(profile, **inf):
    return {"id": "cap", "plugin": "fabricpc_residual_cap", "config": {"channels": ["y0", "y1"], "horizons": 4, "features": {"neural": 16, "semantic": 4}, "hidden": [8, 8],
                                                                     "inference": {"profile": profile, **inf}}, "inputs": INPUTS}


def farm(profile, rule, **inf):
    a = arm([BASE, SEM, fcap(profile, **inf)], learning={"rule": rule, "optimizer": {"kind": "adam", "lr": 0.01}})
    a["inference"] = {"profile": profile, "allow_experimental": profile == "fabricpc_spc_recurrent"}
    return a


PROFILES = [("fabricpc_feedforward", "fabricpc_bp_through_inference", {}),
            ("fabricpc_spc", "workspace_pc_local", {"eta_infer": 0.05, "infer_steps": 10}),
            ("fabricpc_spc", "workspace_bp_unroll", {"eta_infer": 0.05, "infer_steps": 10}),
            ("fabricpc_epc", "workspace_epc_local", {"eta_infer": 0.05, "infer_steps": 4}),
            ("fabricpc_spc_recurrent", "fabricpc_pc_local", {"eta_infer": 0.05, "infer_steps": 10, "unroll": 2})]


@pytest.mark.parametrize("profile,rule,inf", PROFILES, ids=[f"{p}+{r}" for p, r, _ in PROFILES])
def test_profile_predicts_target_free_learns_and_records_algorithm_spec(registry, scenario, profile, rule, inf):
    sysm = build(registry, farm(profile, rule, **inf), scenario)
    reqs = first_requests(scenario, sysm, "train", 4)
    # zero-initialised readout => exact baseline parity before learning
    base = build(registry, BASELINE_ARM, scenario)
    r0 = reqs[0]
    np.testing.assert_array_equal(sysm.predict(r0)[0].outputs[r0.queries[0].query_id].values, base.predict(r0)[0].outputs[r0.queries[0].query_id].values)
    assert_target_free(lambda req: {k: v.point() for k, v in sysm.predict(req)[0].outputs.items()}, r0)
    m = sysm.manifest()
    spec = m.extra["algorithm_specs"]["cap"]
    assert spec["inference"]["profile"] == profile and spec["clamps_prediction"] == ["input"] and spec["readout"].startswith("z_mu")
    assert m.extra["runtime"]["backend"] == "fabricpc" and m.dependency_lock["fabricpc"].startswith("0.6")
    tx = PredictionTransaction(sysm, "prequential")
    for req in reqs:
        tx.predict(req)
        tx.reveal(req.request_id, {q.task_id: scenario.outcome(req.request_id, q.query_id) for q in req.queries})
    before = {k: v.copy() for k, v in sysm.node_states["cap"].params["workspace"].items()}
    diag = tx.learn_step()
    assert diag["update"] == 1 and np.isfinite(diag["grad_norm"]) and diag["grad_norm"] > 0
    changed = any(not np.array_equal(before[k], sysm.node_states["cap"].params["workspace"][k]) for k in before)
    assert changed
    rule_spec = sysm.manifest().extra["algorithm_specs"][f"rule:{sysm.learning_profile}"]
    assert rule_spec["cap"]["learning_rule"].startswith(sysm.learning_profile)
    # prediction after learning is still target-free and finite
    assert_target_free(lambda req: {k: v.point() for k, v in sysm.predict(req)[0].outputs.items()}, reqs[0])


def test_pc_local_reduces_clamped_energy_and_bp_reduces_task_loss(registry, scenario):
    for rule, key in (("workspace_pc_local", "clamped_energy_per_prediction"), ("workspace_bp_unroll", "task_loss")):
        sysm = build(registry, farm("fabricpc_spc", rule, eta_infer=0.05, infer_steps=10), scenario)
        tx = PredictionTransaction(sysm, "prequential")
        reqs = first_requests(scenario, sysm, "train", 6)
        for req in reqs:
            tx.predict(req)
            tx.reveal(req.request_id, {q.task_id: scenario.outcome(req.request_id, q.query_id) for q in req.queries})
        from ness.learning import BatchItem
        batch = [BatchItem(tx._pending[r.request_id].execution, tx._pending[r.request_id].outcomes) for r in reqs]
        rid = sysm.learning_profile
        l0 = sysm.rules[rid].gradients(sysm.learning_context(), batch, sysm.owned_groups())[2]["cap"][key]
        for _ in range(8):
            sysm.learn(batch)
        l1 = sysm.rules[rid].gradients(sysm.learning_context(), batch, sysm.owned_groups())[2]["cap"][key]
        assert l1 < l0, (rule, l0, l1)


def test_hybrid_credit_jax_upper_bp_plus_fabricpc_cap_pc(registry, scenario):
    upper = {"id": "upper", "plugin": "tiny_upper_mlp", "config": {"in_dim": 16, "model_dim": 8}, "inputs": {"main": "substrate://base_ts/state/final"}}
    cap = {"id": "cap", "plugin": "fabricpc_residual_cap", "config": {"channels": ["y0", "y1"], "horizons": 4, "features": {"neural": 8, "semantic": 4}, "hidden": [8],
                                                                    "inference": {"profile": "fabricpc_spc", "eta_infer": 0.05, "infer_steps": 5}},
           "inputs": {"baseline": "substrate://base_ts/forecast/point", "neural": "module://upper/hidden", "semantic": "semantic://semantic/features"}}
    a = arm([BASE, upper, SEM, cap], learning={"rule": "bp_direct", "optimizer": {"kind": "adam", "lr": 0.01}, "credit": {"overrides": {"cap/workspace": "fabricpc_pc_local"}}})
    a["inference"] = {"profile": "fabricpc_spc"}
    sysm = build(registry, a, scenario)
    assert set(sysm.rules) == {"bp_direct", "fabricpc_pc_local"}
    assert sysm.credit.owners["upper/upper"] == "bp_direct" and sysm.credit.owners["cap/workspace"] == "fabricpc_pc_local"
    assert any(e.startswith("module://upper/hidden->cap") for e in sysm.credit.stop_boundaries)  # no cross-runtime gradient
    tx = PredictionTransaction(sysm, "prequential")
    for req in first_requests(scenario, sysm, "train", 4):
        tx.predict(req)
        tx.reveal(req.request_id, {q.task_id: scenario.outcome(req.request_id, q.query_id) for q in req.queries})
    d = tx.learn_step()
    assert set(d["rules"]) == {"bp_direct", "fabricpc_pc_local"}
    # bp_direct's gradient for the upper module is zero: the FabricPC cap is a stop-gradient consumer of upper/hidden
    assert d["rules"]["bp_direct"]["grad_norm"] == 0.0


def test_cross_runtime_ownership_is_refused(registry, scenario):
    with pytest.raises(GradientBoundaryError):
        build(registry, farm("fabricpc_feedforward", "bp_direct"), scenario)
    upper = {"id": "upper", "plugin": "tiny_upper_mlp", "config": {"in_dim": 16, "model_dim": 8}, "inputs": {"main": "substrate://base_ts/state/final"}}
    cap = copy.deepcopy(fcap("fabricpc_feedforward"))
    cap["config"]["features"] = {"neural": 8, "semantic": 4}
    cap["inputs"]["neural"] = "module://upper/hidden"
    with pytest.raises(GradientBoundaryError):
        build(registry, arm([BASE, upper, SEM, cap], learning={"rule": "fabricpc_pc_local"}), scenario)


def test_learned_boundary_transform_into_fabricpc_node_is_refused(registry, scenario):
    from ness.contracts import CompositionError
    cap = copy.deepcopy(fcap("fabricpc_feedforward"))
    cap["inputs"]["neural"] = {"from": "substrate://base_ts/state/final", "boundary": ["mean_pool", {"kind": "linear", "out_dim": 16}]}
    with pytest.raises(CompositionError, match="learned boundary transform"):
        build(registry, arm([BASE, SEM, cap], learning={"rule": "fabricpc_bp_through_inference"}), scenario)


def test_experimental_recurrent_profile_requires_explicit_opt_in(registry, scenario):
    a = farm("fabricpc_spc_recurrent", "fabricpc_pc_local", eta_infer=0.05, infer_steps=5, unroll=2)
    a["inference"] = {"profile": "fabricpc_spc_recurrent"}
    with pytest.raises(UnsupportedCapability, match="experimental"):
        build(registry, a, scenario)
    with pytest.raises(ValidationError, match="unroll"):
        build(registry, farm("fabricpc_spc_recurrent", "fabricpc_pc_local", eta_infer=0.05, infer_steps=5), scenario)


def test_masked_outcome_refused_by_pc_local_but_honoured_by_bp(registry, scenario):
    from ness.contracts import TrainingOutcome
    from ness.learning import BatchItem
    sysm = build(registry, farm("fabricpc_spc", "workspace_pc_local", eta_infer=0.05, infer_steps=5), scenario)
    req = first_requests(scenario, sysm, "train", 1)[0]
    _, rec, _ = sysm.predict(req, mode="train")
    oc = scenario.outcome(req.request_id, req.queries[0].query_id)
    mask = oc.mask.copy(); mask[0, 0] = False
    partial = TrainingOutcome(oc.request_id, oc.query_id, oc.values, mask, oc.release_sequence, oc.available_at, oc.target)
    with pytest.raises(UnsupportedCapability, match="masked"):
        sysm.learn([BatchItem(rec, {"future_value": partial})])
    sysm2 = build(registry, farm("fabricpc_spc", "workspace_bp_unroll", eta_infer=0.05, infer_steps=5), scenario)
    _, rec2, _ = sysm2.predict(req, mode="train")
    d = sysm2.learn([BatchItem(rec2, {"future_value": partial})])
    assert np.isfinite(d["loss"])
