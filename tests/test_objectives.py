"""N-04 / ADR-0012: auxiliary objectives, training-only nodes, revealed auxiliary targets and
parameter anchors (acceptance A1-A5 of the v0.3.0 core handoff) plus fail-closed cases."""

import copy

import numpy as np
import pytest

from ness.composition import compile_graph, parse_composition
from ness.contracts import AccessPolicyViolation, CompositionError, ContractViolation, GradientBoundaryError, UnsupportedCapability, ValidationError
from ness.learning import BatchItem
from ness.learning.objectives import loss_registry, parse_objectives
from ness.plugin_api.testkit import assert_target_free
from ness.runtime.system import NessSystem
from ness.runtime.transaction import PredictionTransaction
from ness.tasks import AS_OF_ORIGIN_ROLES

from ness_test_helpers import BASE, TASKS, arm, experiment, needs_jax

pytestmark = needs_jax

UPPER1 = {"id": "upper", "plugin": "tiny_upper_transformer", "config": {"in_dim": 16, "model_dim": 16, "heads": 2},
          "inputs": {"main": "substrate://base_ts/state/final"}}
CAP = {"id": "cap", "plugin": "residual_point_cap", "config": {"channels": ["y0", "y1"], "horizons": 4, "features": {"neural": 16}},
       "inputs": {"baseline": "substrate://base_ts/forecast/point", "neural": "module://upper/hidden"}}
TEACHER = {"id": "teacher", "plugin": "toy_frozen_transformer", "training_only": True,
           "config": {"width": 16, "heads": 2, "layers": 2, "channels": ["y0", "y1"], "horizons": 4, "seed": 11, "pretrain_steps": 60},
           "inputs": {"history": "observation://target_history"}}
PRESERVE = {"id": "preserve", "kind": "port_target", "source": "module://upper/sequence", "target": "substrate://teacher/state/final", "loss": "mse"}
SGD = {"rule": "bp_direct", "optimizer": {"kind": "sgd", "lr": 0.05}}


def _system(registry, scenario, nodes, objectives=None, learning=SGD, arm_id="a", output="module://cap/forecast"):
    lr = copy.deepcopy(learning)
    if objectives is not None:
        lr["objectives"] = copy.deepcopy(objectives)
    exp = experiment({arm_id: arm(nodes, output=output, learning=lr)})
    return NessSystem.build(exp, exp.arm(arm_id), registry, scenario=scenario)


def _batch(system, scenario, n=4, split="train"):
    tx = PredictionTransaction(system, "prequential")
    for req in scenario.iter_requests(split, tuple(system.tasks.values()), {"max_requests": n}):
        tx.predict(req)
        tx.reveal(req.request_id, {q.task_id: scenario.outcome(req.request_id, q.query_id) for q in req.queries})
    return tx, [BatchItem(it.execution, it.outcomes, it.request, dict(it.auxiliary)) for it in tx._batch]


def _grads(system, batch):
    if system.objectives is not None:
        system._prepare_objectives(batch)
    ctx = system.learning_context()
    return system.rule.gradients(ctx, batch, system.owned_groups())


# ------------------------------------------------------------------ A1
def test_A1_auxiliary_port_objective_matches_hand_written_jax_gradient(registry, scenario):
    import jax
    import jax.numpy as jnp

    w = 0.37
    sysm = _system(registry, scenario, [BASE, UPPER1, CAP, TEACHER], [{**PRESERVE, "weight": w}])
    _, batch = _batch(sysm, scenario)
    loss, grads, diag = _grads(sysm, batch)
    assert diag["objectives"]["preserve"]["items"] == len(batch) and diag["objectives"]["task:future_value"]["items"] == len(batch)

    # the teacher target computed independently: an ordinary (non training-only) prediction of the same provider
    teacher_sys = _system(registry, scenario, [dict(TEACHER, id="base_ts", training_only=False)], learning=None, output="substrate://base_ts/forecast/point")
    targets = [np.asarray(teacher_sys.predict(it.request)[1].port_values[("base_ts", "state/final")].dense()) for it in batch]

    upper, capm = sysm.compiled.node("upper").module, sysm.compiled.node("cap").module
    task = sysm.tasks["future_value"]
    st = sysm.node_states
    owned = sysm.owned_groups()
    tp0 = {gid: {n: jnp.asarray(a) for n, a in st[gid.split("/")[0]].params[gid.split("/", 1)[1]].items()} for gid in owned}

    def f(tp):
        up_p = {g.split("/", 1)[1]: v for g, v in tp.items() if g.startswith("upper/")}
        cap_p = {g.split("/", 1)[1]: v for g, v in tp.items() if g.startswith("cap/")}
        tn = td = an = ad = 0.0
        for it, tgt in zip(batch, targets):
            rec = it.record
            x = jnp.asarray(rec.port_values[("base_ts", "state/final")].dense())
            base = jnp.asarray(rec.port_values[("base_ts", "forecast/point")].dense())
            up = upper.apply(up_p, {"main": x}, st["upper"], jnp)
            out = capm.apply(cap_p, {"baseline": base, "neural": up["hidden"]}, st["cap"], jnp)
            oc = it.outcomes["future_value"]
            n, d = task.loss_terms(out["forecast"], jnp.asarray(oc.values), jnp.asarray(oc.mask, dtype=jnp.float64), jnp)
            tn, td = tn + n, td + d
            an, ad = an + jnp.sum((up["sequence"] - tgt) ** 2), ad + tgt.size
        return tn / td + w * an / ad

    ref_loss, ref = jax.value_and_grad(f)(tp0)
    assert abs(float(ref_loss) - loss) <= 1e-9
    for gid in owned:
        for n in ref[gid]:
            np.testing.assert_allclose(grads[gid][n], np.asarray(ref[gid][n]), rtol=0, atol=1e-9)
    assert max(float(np.max(np.abs(g))) for g in grads["upper/" + next(iter(k.split("/", 1)[1] for k in owned if k.startswith("upper/")))].values()) > 0


def test_A1_weight_zero_is_bitwise_task_only_training(registry, scenario):
    plain = _system(registry, scenario, [BASE, UPPER1, CAP])                                   # task-only, no teacher
    zero = _system(registry, scenario, [BASE, UPPER1, CAP, TEACHER], [{**PRESERVE, "weight": 0.0}])
    for sysm in (plain, zero):
        tx, _ = _batch(sysm, scenario)
        sysm._diag = tx.learn_step()
    for nid in ("upper", "cap"):
        for g, grp in plain.node_states[nid].params.items():
            for n, a in grp.items():
                np.testing.assert_array_equal(a, zero.node_states[nid].params[g][n])
    assert zero._diag["loss"] == plain._diag["loss"]
    assert zero._diag["objectives"]["preserve"]["weight"] == 0.0 and zero._diag["objectives"]["preserve"]["value"] > 0


# ------------------------------------------------------------------ A2
def _compile(registry, scenario, nodes, output="module://cap/forecast"):
    spec = parse_composition({"nodes": copy.deepcopy(nodes), "output": {"task": "future_value", "from": output}})
    task = registry.create("point_forecast_task", {"task_id": "future_value", "horizons": 4, "channels": ["y0", "y1"]})
    return compile_graph(spec, registry, scenario.observation_fields(), {"future_value": task.prediction_space()}, AS_OF_ORIGIN_ROLES)


def test_A2_training_only_nodes_can_never_reach_an_output(registry, scenario):
    g = _compile(registry, scenario, [BASE, UPPER1, CAP, TEACHER])
    assert g.training_only_nodes == ("teacher",) and not g.node("teacher").differentiable
    feeds = copy.deepcopy(UPPER1)
    feeds["inputs"]["main"] = "substrate://teacher/state/final"                                  # a predicting node reads it
    with pytest.raises(CompositionError, match="training-only"):
        _compile(registry, scenario, [BASE, feeds, CAP, TEACHER])
    with pytest.raises(CompositionError, match="training-only"):
        _compile(registry, scenario, [BASE, TEACHER], output="substrate://teacher/forecast/point")  # it is the output
    trainable = dict(UPPER1, id="t2", training_only=True)
    with pytest.raises(CompositionError, match="frozen"):
        _compile(registry, scenario, [BASE, UPPER1, CAP, trainable])
    chained = dict(TEACHER, id="t3", inputs={"history": "observation://target_history"})
    g2 = _compile(registry, scenario, [BASE, UPPER1, CAP, TEACHER, chained])                     # training-only may feed training-only
    assert set(g2.training_only_nodes) == {"teacher", "t3"}


def test_A2_predictions_identical_with_and_without_training_only_node(registry, scenario):
    a = _system(registry, scenario, [BASE, UPPER1, CAP])
    b = _system(registry, scenario, [BASE, UPPER1, CAP, TEACHER], [{**PRESERVE, "weight": 0.5}])
    reqs = list(scenario.iter_requests("test", tuple(a.tasks.values()), {"max_requests": 3}))
    for req in reqs:
        ra, reca, _ = a.predict(req)
        rb, recb, _ = b.predict(req)
        for q in req.queries:
            np.testing.assert_array_equal(ra.outputs[q.query_id].point(), rb.outputs[q.query_id].point())
        assert not any(k[0] == "teacher" for k in recb.port_values)                              # never executed for a prediction
    assert b.manifest().extra["training_only_nodes"] == ["teacher"] and "training_only_nodes" not in a.manifest().extra
    assert "objectives" in b.manifest().extra["algorithm_specs"] and "objectives" not in a.manifest().extra["algorithm_specs"]
    assert_target_free(lambda r: {k: v.dense() for k, v in b.predict(r)[0].outputs.items()}, reqs[0])


# ------------------------------------------------------------------ A3
def test_A3_request_filter_contributes_nothing_for_non_matching_items(registry, scenario):
    none = _system(registry, scenario, [BASE, UPPER1, CAP, TEACHER], [{**PRESERVE, "weight": 0.8, "applies_to": {"group": {"series_id": "never"}}}])
    zero = _system(registry, scenario, [BASE, UPPER1, CAP, TEACHER], [{**PRESERVE, "weight": 0.0}])
    _, b1 = _batch(none, scenario)
    _, b2 = _batch(zero, scenario)
    l1, g1, d1 = _grads(none, b1)
    l2, g2, _ = _grads(zero, b2)
    o = d1["objectives"]["preserve"]
    assert (o["weight"], o["numerator"], o["denominator"], o["items"]) == (0.8, 0.0, 0.0, 0) and np.isnan(o["value"])
    assert l1 == l2
    for gid in g2:
        for n in g2[gid]:
            np.testing.assert_array_equal(g1[gid][n], g2[gid][n])
    # a partial filter: only items whose hidden-regime-free group key matches take part, and exactly their terms are summed
    origins = [r.origin for r in scenario.iter_requests("train", tuple(zero.tasks.values()), {"max_requests": 6})]
    chosen = [origins[0], origins[3]]
    part = _system(registry, scenario, [BASE, UPPER1, CAP, TEACHER], [{**PRESERVE, "applies_to": {"group": {"origin": chosen}}}])
    _, b3 = _batch(part, scenario, n=6)
    _, _, d3 = _grads(part, b3)
    match = [it for it in b3 if it.request.group["origin"] in chosen]
    assert d3["objectives"]["preserve"]["items"] == len(match) and len(match) >= 1
    num = den = 0.0
    for it in match:
        seq = np.asarray(it.record.port_values[("upper", "sequence")].dense())
        tgt = it.objective_inputs["preserve"][0]
        num += float(np.sum((seq - tgt) ** 2))
        den += tgt.size
    assert abs(d3["objectives"]["preserve"]["numerator"] - num) <= 1e-9 * max(1.0, num) and d3["objectives"]["preserve"]["denominator"] == den


# ------------------------------------------------------------------ A4
def test_A4_parameter_anchor_gradient_and_checkpoint_round_trip(registry, scenario, tmp_path):
    from ness.checkpoint import CheckpointStore

    w = 0.25
    anch = _system(registry, scenario, [BASE, UPPER1, CAP], [{"id": "keep", "kind": "parameter_anchor", "groups": ["upper/*"], "weight": w}])
    plain = _system(registry, scenario, [BASE, UPPER1, CAP])
    assert any(a[1].startswith("l2_to_reference:initial:keep") for a in anch.credit.anchors) and not plain.credit.anchors
    for sysm in (anch, plain):                                    # move theta away from the reference, identically
        for g, grp in sysm.node_states["upper"].params.items():
            sysm.node_states["upper"].params[g] = {n: a + 0.01 * np.random.default_rng(len(g) * 7 + len(n)).normal(size=a.shape) for n, a in grp.items()}
    tx_a, _ = _batch(anch, scenario)
    tx_p, _ = _batch(plain, scenario)
    before = {g: {n: a.copy() for n, a in grp.items()} for g, grp in anch.node_states["upper"].params.items()}
    da, dp = tx_a.learn_step(), tx_p.learn_step()
    lr = 0.05
    for g, grp in before.items():
        for n, theta in grp.items():
            ref = anch.anchor_refs["keep"][f"upper/{g}"][n]
            g_task = (theta - plain.node_states["upper"].params[g][n]) / lr        # plain SGD: theta' = theta - lr * g_task
            expected = theta - lr * (g_task + w * (theta - ref))
            np.testing.assert_allclose(anch.node_states["upper"].params[g][n], expected, rtol=0, atol=1e-12)
    assert da["objectives"]["keep"]["value"] > 0 and da["objectives"]["keep"]["items"] == 0
    # the reference survives a learner checkpoint; training then continues identically
    store = CheckpointStore(tmp_path / "ck")
    mid = store.stage_and_publish(anch.snapshot("learner"), "a", store.head("a"))
    restored = NessSystem.from_snapshot(store.load(mid), registry, scenario)
    for gid, grp in anch.anchor_refs["keep"].items():
        for n, a in grp.items():
            np.testing.assert_array_equal(restored.anchor_refs["keep"][gid][n], a)
    for sysm in (anch, restored):
        tx = PredictionTransaction(sysm, "prequential")
        for req in list(scenario.iter_requests("train", tuple(sysm.tasks.values()), {"max_requests": 8}))[4:]:
            tx.predict(req)
            tx.reveal(req.request_id, {q.task_id: scenario.outcome(req.request_id, q.query_id) for q in req.queries})
        tx.learn_step()
    for g, grp in anch.node_states["upper"].params.items():
        for n, a in grp.items():
            np.testing.assert_array_equal(a, restored.node_states["upper"].params[g][n])
    # a serving checkpoint carries no reference: learning fails closed instead of anchoring to the wrong weights
    mid_s = store.stage_and_publish(anch.snapshot("serving"), "s", store.head("s"))
    served = NessSystem.from_snapshot(store.load(mid_s), registry, scenario)
    tx = PredictionTransaction(served, "prequential")
    req = next(iter(scenario.iter_requests("train", tuple(served.tasks.values()), {"max_requests": 1})))
    tx.predict(req)
    tx.reveal(req.request_id, {q.task_id: scenario.outcome(req.request_id, q.query_id) for q in req.queries})
    with pytest.raises(ContractViolation, match="anchor"):
        tx.learn_step()


# ------------------------------------------------------------------ A5
def test_A5_auxiliary_targets_are_learning_only(registry, scenario):
    aux = [{"id": "task_off", "kind": "task", "task": "future_value", "weight": 0.0},
           {"id": "aux", "kind": "port_target", "source": "module://cap/forecast", "target": {"auxiliary": "target_future"}, "loss": "mse"}]
    sysm = _system(registry, scenario, [BASE, UPPER1, CAP], aux)
    plain = _system(registry, scenario, [BASE, UPPER1, CAP])
    assert sysm.auxiliary_fields() == frozenset({"target_future"})
    req = next(iter(scenario.iter_requests("train", tuple(sysm.tasks.values()), {"max_requests": 1})))
    _, _, ctx = sysm.predict(req)
    assert not ctx.permitted_bundle.has("target_future")                          # never in a permitted view
    assert_target_free(lambda r: {k: v.dense() for k, v in sysm.predict(r)[0].outputs.items()}, req)
    _, b1 = _batch(sysm, scenario)
    assert all("target_future" in it.auxiliary for it in b1)                      # delivered by reveal, to learning only
    _, b2 = _batch(plain, scenario)
    l1, g1, d1 = _grads(sysm, b1)
    l2, g2, _ = _grads(plain, b2)
    # the revealed target equals the task outcome here, so the auxiliary mse reproduces the task gradient
    assert abs(l1 - l2) <= 1e-12 and d1["objectives"]["task_off"]["weight"] == 0.0
    for gid in g2:
        for n in g2[gid]:
            np.testing.assert_allclose(g1[gid][n], g2[gid][n], rtol=0, atol=1e-12)
    wired = copy.deepcopy(CAP)
    wired["config"]["features"]["leak"] = 8
    wired["inputs"]["leak"] = {"from": "observation://target_future", "boundary": ["flatten"]}
    with pytest.raises(AccessPolicyViolation):
        _compile(registry, scenario, [BASE, UPPER1, wired])


# ------------------------------------------------------------------ fail closed
@pytest.mark.parametrize("objective, error, match", [
    ({"id": "x", "kind": "port_target", "source": "substrate://base_ts/state/final", "target": "substrate://teacher/state/final", "loss": "mse"}, GradientBoundaryError, "not differentiable"),
    ({"id": "x", "kind": "port_target", "source": "module://upper/sequence", "target": "module://upper/hidden", "loss": "mse"}, GradientBoundaryError, "constants"),
    ({"id": "x", "kind": "port_target", "source": "module://upper/sequence", "target": "substrate://teacher/state/final", "loss": "nope"}, ValidationError, "unknown loss"),
    ({"id": "x", "kind": "port_target", "source": "module://upper/sequence", "target": {"auxiliary": "target_history"}, "loss": "mse"}, ValidationError, "outcome_only"),
    ({"id": "x", "kind": "port_target", "source": "module://upper/sequence", "target": "substrate://teacher/state/final", "loss": "mse", "loss_config": {"t": 2}}, ValidationError, "no options"),
    ({"id": "x", "kind": "parameter_anchor", "groups": ["nothing/*"]}, ValidationError, "matches no owned"),
    ({"id": "x", "kind": "task", "task": "other"}, ValidationError, "unknown task"),
    ({"id": "x", "kind": "port_target", "source": "module://upper/sequence", "target": "substrate://teacher/state/final", "loss": "mse", "extra": 1}, ValidationError, "unknown keys"),
    ({"id": "x", "kind": "weird"}, ValidationError, "unknown kind"),
])
def test_objectives_fail_closed(registry, scenario, objective, error, match):
    with pytest.raises(error, match=match):
        _system(registry, scenario, [BASE, UPPER1, CAP, TEACHER], [objective])


def test_objectives_unsupported_in_batched_mode_and_duplicate_ids(registry, scenario):
    with pytest.raises(UnsupportedCapability, match="per-item"):
        _system(registry, scenario, [BASE, UPPER1, CAP, TEACHER], [PRESERVE], learning={**SGD, "rule_config": {"batched": True}})
    with pytest.raises(ValidationError, match="duplicate"):
        parse_objectives([PRESERVE, PRESERVE])


def test_builtin_losses_match_closed_forms():
    reg = loss_registry()
    rng = np.random.default_rng(0)
    s, t = rng.normal(size=(3, 5)), rng.normal(size=(3, 5))
    mask = np.array([1.0, 0.0, 1.0])
    lsm = lambda x: x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))  # noqa: E731
    n, d = reg["kl_last_axis"].fn(s, t, mask, np)
    kl = np.sum(np.exp(lsm(t)) * (lsm(t) - lsm(s)), axis=-1)
    assert abs(n - np.sum(mask * kl)) < 1e-12 and d == 2.0
    n, d = reg["kl_last_axis"].fn(s, t, None, np, temperature=2.0)
    kl2 = np.sum(np.exp(lsm(t / 2)) * (lsm(t / 2) - lsm(s / 2)), axis=-1)
    assert abs(n - kl2.sum()) < 1e-12 and d == 3.0
    ids = np.array([4.0, 0.0, 2.0])
    n, d = reg["cross_entropy"].fn(s, ids, mask, np)
    assert abs(n - (-lsm(s)[0, 4] - lsm(s)[2, 2])) < 1e-12 and d == 2.0
    n, d = reg["mse"].fn(s, t, None, np)
    assert abs(n - np.sum((s - t) ** 2)) < 1e-12 and d == 15.0
    n, d = reg["mse"].fn(s, t, mask, np)                                          # mask over leading positions broadcasts
    assert abs(n - np.sum(mask[:, None] * (s - t) ** 2)) < 1e-12 and d == 10.0
