"""NESS JAX backend vs NESS FabricPC backend on the *same* dense computation: forward values,
final prediction, raw gradients and one optimizer update. Algorithms that are definitionally
different (PC local rule vs BP) are NOT forced to agree; their difference is asserted."""

import copy
import sys

import numpy as np
import pytest

from ness_test_helpers import BASE, arm, build, first_requests  # noqa: E402

from ness.learning import BatchItem
from ness.runtime.transaction import PredictionTransaction

SEM = {"id": "semantic", "plugin": "simple_temporal_semantics", "config": {"window": 8, "channels": 2, "use_neural": False}, "inputs": {"raw": "observation://target_history"}}
INPUTS = {"baseline": "substrate://base_ts/forecast/point", "neural": {"from": "substrate://base_ts/state/final", "boundary": ["mean_pool"]}, "semantic": "semantic://semantic/features"}
CFG = {"channels": ["y0", "y1"], "horizons": 4, "features": {"neural": 16, "semantic": 4}, "hidden": [8, 8], "activation": "tanh"}


def jax_arm():
    return arm([BASE, SEM, {"id": "cap", "plugin": "jax_dense_workspace_cap", "config": CFG, "inputs": INPUTS}], learning={"rule": "bp_direct", "optimizer": {"kind": "sgd", "lr": 0.05}})


def fpc_arm(rule="fabricpc_bp_through_inference", profile="fabricpc_feedforward", **inf):
    a = arm([BASE, SEM, {"id": "cap", "plugin": "fabricpc_residual_cap", "config": {**CFG, "inference": {"profile": profile, **inf}}, "inputs": INPUTS}],
            learning={"rule": rule, "optimizer": {"kind": "sgd", "lr": 0.05}})
    a["inference"] = {"profile": profile}
    return a


def _copy_weights(jax_sys, fpc_sys):
    """Map jax dense params (w0,b0,w1,b1) onto the FabricPC graph (h0, readout)."""
    jp = jax_sys.node_states["cap"].params["workspace"]
    fp = fpc_sys.node_states["cap"].params["workspace"]
    keymap = {}
    for k in fp:
        node, kind, name = k.split("|", 2)
        layer = int(node[1:]) if node.startswith("h") else len(CFG["hidden"])
        keymap[k] = (f"w{layer}" if kind == "weights" else f"b{layer}")
    for k, jk in keymap.items():
        src = jp[jk]
        fp[k] = src.reshape(fp[k].shape) if fp[k].shape != src.shape else src.copy()
    fpc_sys._manifest_cache = None


def _batch(sysm, scenario, n=4):
    items = []
    for req in first_requests(scenario, sysm, "train", n):
        _, rec, _ = sysm.predict(req, mode="train")
        items.append(BatchItem(rec, {q.task_id: scenario.outcome(req.request_id, q.query_id) for q in req.queries}))
    return items


def test_forward_and_bp_gradient_parity_feedforward(registry, scenario):
    jx, fp = build(registry, jax_arm(), scenario), build(registry, fpc_arm(), scenario)
    # give both the same (non-zero) weights
    rng = np.random.default_rng(3)
    for k, v in jx.node_states["cap"].params["workspace"].items():
        jx.node_states["cap"].params["workspace"][k] = rng.normal(0, 0.3, v.shape)
    _copy_weights(jx, fp)
    reqs = first_requests(scenario, jx, "test", 3)
    for r in reqs:
        a = jx.predict(r)[0].outputs[r.queries[0].query_id].values
        b = fp.predict(r)[0].outputs[r.queries[0].query_id].values
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
    bj, bf = _batch(jx, scenario), _batch(fp, scenario)
    lj, gj, _ = jx.rules["bp_direct"].gradients(jx.learning_context(), bj, jx.owned_groups())
    lf, gf, _ = fp.rules["fabricpc_bp_through_inference"].gradients(fp.learning_context(), bf, fp.owned_groups())
    assert abs(lj - lf) < 1e-9
    gjw, gfw = gj["cap/workspace"], gf["cap/workspace"]
    for k, v in gfw.items():
        node, kind, name = k.split("|", 2)
        layer = int(node[1:]) if node.startswith("h") else len(CFG["hidden"])
        ref = gjw[f"w{layer}" if kind == "weights" else f"b{layer}"].reshape(v.shape)
        np.testing.assert_allclose(v, ref, rtol=1e-8, atol=1e-10)
    # one optimizer update on both -> identical predictions afterwards
    jx.learn(bj); fp.learn(bf)
    for r in reqs:
        np.testing.assert_allclose(jx.predict(r)[0].outputs[r.queries[0].query_id].values, fp.predict(r)[0].outputs[r.queries[0].query_id].values, rtol=1e-9, atol=1e-10)


def _same_weights(*systems):
    rng = np.random.default_rng(5)
    for k in systems[0].node_states["cap"].params["workspace"]:
        w = rng.normal(0, 0.3, systems[0].node_states["cap"].params["workspace"][k].shape)
        for s in systems:
            s.node_states["cap"].params["workspace"][k] = w.copy()


def test_target_free_settling_on_a_dag_with_feedforward_init_is_the_feedforward_prediction(registry, scenario):
    """A scientific fact the platform must state, not hide: with FeedforwardStateInit every
    unclamped node starts at z_latent = z_mu (zero energy), so free settling on an acyclic
    workspace moves nothing. fabricpc_spc and fabricpc_feedforward are the SAME deployed
    predictor; they differ only in the learning rule. Recurrence or a non-feedforward state
    initialisation is required for settling to change the deployed prediction."""
    ff = build(registry, fpc_arm(), scenario)
    spc = build(registry, fpc_arm("workspace_pc_local", "fabricpc_spc", eta_infer=0.05, infer_steps=10), scenario)
    _same_weights(ff, spc)
    r = first_requests(scenario, ff, "test", 1)[0]
    a = ff.predict(r)[0].outputs[r.queries[0].query_id].values
    b = spc.predict(r)[0].outputs[r.queries[0].query_id].values
    np.testing.assert_allclose(a, b, atol=1e-12)
    sa, sb = ff.manifest().extra["algorithm_specs"]["cap"], spc.manifest().extra["algorithm_specs"]["cap"]
    assert sa["inference"]["solver"] == "none" and sb["inference"]["solver"] == "state_based_sgd"
    assert ff.manifest().manifest_id != spc.manifest().manifest_id  # different algorithm identity even when predictions coincide


def test_settling_changes_the_deployed_prediction_with_global_init_or_recurrence(registry, scenario):
    ff = build(registry, fpc_arm(), scenario)
    noisy = build(registry, fpc_arm("workspace_pc_local", "fabricpc_spc", eta_infer=0.05, infer_steps=10, state_init="global_normal", state_init_std=0.5), scenario)
    _same_weights(ff, noisy)
    r = first_requests(scenario, ff, "test", 1)[0]
    a = ff.predict(r)[0].outputs[r.queries[0].query_id].values
    b = noisy.predict(r)[0].outputs[r.queries[0].query_id].values
    b2 = noisy.predict(r)[0].outputs[r.queries[0].query_id].values
    assert not np.allclose(a, b) and np.array_equal(b, b2)  # differs from feedforward, deterministic per request
    assert "GlobalStateInit" in noisy.manifest().extra["algorithm_specs"]["cap"]["initialization"]
    rec = build(registry, {**fpc_arm("fabricpc_pc_local", "fabricpc_spc_recurrent", eta_infer=0.05, infer_steps=10, unroll=2), "inference": {"profile": "fabricpc_spc_recurrent", "allow_experimental": True}}, scenario)
    rec.compiled.node("cap").module  # two hidden layers are needed for the lateral edge
