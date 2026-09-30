"""N-02 / ADR-0013: optimizer parameter groups and learning-rate schedules (acceptance B1-B3;
B4, the vertical slice bitwise, is checked against v0.2.1 outside the unit suite)."""

import copy
import math

import numpy as np
import pytest

from ness.contracts import ValidationError
from ness.learning.optimizers import GroupRule, Optimizer, ScheduleSpec

from ness_test_helpers import BASE, arm, experiment, needs_jax


def _params(seed=0):
    rng = np.random.default_rng(seed)
    return {"upper/w": {"a": rng.normal(size=(3, 2)), "b": rng.normal(size=4)}, "cap/c": {"k": rng.normal(size=5)}}


def _v021_update(opt_state, params, grads):
    """The v0.2.1 Optimizer.update, verbatim, as the bitwise reference for the default path."""
    s = opt_state
    s["step"] += 1
    if s["clip_norm"] is not None:
        total = float(np.sqrt(sum(float(np.sum(np.asarray(g) ** 2)) for grp in grads.values() for g in grp.values())))
        scale = min(1.0, s["clip_norm"] / (total + 1e-12))
        grads = {gid: {n: np.asarray(g) * scale for n, g in grp.items()} for gid, grp in grads.items()}
    out = {gid: {n: np.array(a, copy=True) for n, a in grp.items()} for gid, grp in params.items()}
    for gid, grp in grads.items():
        for n, g in grp.items():
            g = np.asarray(g, dtype=np.float64)
            p = out[gid][n]
            if s["wd"]:
                p = p * (1.0 - s["lr"] * s["wd"])
            if s["kind"] == "sgd":
                out[gid][n] = p - s["lr"] * g
                continue
            m = s["m"].setdefault(gid, {}).get(n, np.zeros_like(g))
            v = s["v"].setdefault(gid, {}).get(n, np.zeros_like(g))
            m = 0.9 * m + (1 - 0.9) * g
            v = 0.999 * v + (1 - 0.999) * g**2
            s["m"][gid][n], s["v"][gid][n] = m, v
            mhat = m / (1 - 0.9 ** s["step"])
            vhat = v / (1 - 0.999 ** s["step"])
            out[gid][n] = p - s["lr"] * mhat / (np.sqrt(vhat) + 1e-8)
    return out


@pytest.mark.parametrize("kind", ["adam", "sgd"])
def test_default_path_is_bitwise_v021(kind):
    opt = Optimizer.from_config({"kind": kind, "lr": 0.01, "weight_decay": 0.1, "clip_norm": 0.5})
    ref = {"step": 0, "clip_norm": 0.5, "wd": 0.1, "lr": 0.01, "kind": kind, "m": {}, "v": {}}
    p1 = p2 = _params()
    for step in range(3):
        grads = _params(10 + step)
        p1 = opt.update(p1, grads)
        p2 = _v021_update(ref, p2, grads)
        for gid in p1:
            for n in p1[gid]:
                np.testing.assert_array_equal(p1[gid][n], p2[gid][n])
    assert not opt.configured and opt.last_stats == {}


def test_B1_group_learning_rates_sgd_exact_and_adam_first_step_ratio():
    rules = [{"match": "upper/*", "lr_scale": 0.1}]
    sgd = Optimizer.from_config({"kind": "sgd", "lr": 0.2, "param_groups": rules})
    sgd.bind_groups(["upper/w", "cap/c"])
    p, g = _params(), _params(1)
    out = sgd.update(p, g)
    for n in p["upper/w"]:
        np.testing.assert_array_equal(out["upper/w"][n], p["upper/w"][n] - 0.2 * 0.1 * g["upper/w"][n])
    np.testing.assert_array_equal(out["cap/c"]["k"], p["cap/c"]["k"] - 0.2 * g["cap/c"]["k"])
    assert sgd.last_stats["upper/w"]["lr"] == 0.2 * 0.1 and sgd.last_stats["cap/c"]["lr"] == 0.2
    adam = Optimizer.from_config({"kind": "adam", "lr": 0.01, "param_groups": [{"match": "upper/*", "lr_scale": 0.1}, {"match": "cap/*", "lr_scale": 1.0}]})
    adam.bind_groups(["upper/w", "cap/c"])
    same = {"upper/w": {"a": np.ones((2,)) * 0.3}, "cap/c": {"a": np.ones((2,)) * 0.3}}
    zero = {k: {n: np.zeros_like(a) for n, a in v.items()} for k, v in same.items()}
    out = adam.update(zero, same)
    ratio = out["upper/w"]["a"] / out["cap/c"]["a"]
    np.testing.assert_allclose(ratio, 0.1, rtol=1e-12)
    absolute = Optimizer.from_config({"kind": "sgd", "lr": 1.0, "param_groups": [{"match": "cap/*", "lr": 0.003}]})
    absolute.bind_groups(["upper/w", "cap/c"])
    assert absolute.group_lr("cap/c") == 0.003 and absolute.group_lr("upper/w") == 1.0


@pytest.mark.parametrize("sched, step, expected", [
    ({"kind": "constant", "warmup_steps": 4}, 1, 0.25),
    ({"kind": "constant", "warmup_steps": 4}, 4, 1.0),
    ({"kind": "constant", "warmup_steps": 4}, 99, 1.0),
    ({"kind": "linear", "warmup_steps": 2, "total_steps": 12, "min_lr_ratio": 0.1}, 7, 1 - 0.9 * 0.5),
    ({"kind": "linear", "warmup_steps": 2, "total_steps": 12, "min_lr_ratio": 0.1}, 40, 0.1),
    ({"kind": "cosine", "warmup_steps": 0, "total_steps": 10}, 5, 0.5 * (1 + math.cos(math.pi * 0.5))),
    ({"kind": "cosine", "warmup_steps": 10, "total_steps": 20, "min_lr_ratio": 0.2}, 15, 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * 0.5))),
    ({"kind": "cosine", "warmup_steps": 10, "total_steps": 20, "min_lr_ratio": 0.2}, 20, 0.2),
])
def test_B2_schedule_closed_forms(sched, step, expected):
    assert abs(ScheduleSpec.from_config(sched).factor(step) - expected) < 1e-15
    opt = Optimizer.from_config({"kind": "sgd", "lr": 0.5, "schedule": sched})
    opt.bind_groups(["g/x"])
    assert abs(opt.group_lr("g/x", step) - 0.5 * expected) < 1e-15


def test_clipping_order_multiplier_then_group_then_global():
    opt = Optimizer.from_config({"kind": "sgd", "lr": 1.0, "clip_norm": 1.0,
                                 "param_groups": [{"match": "a/*", "grad_multiplier": 10.0, "clip_norm": 2.0}]})
    opt.bind_groups(["a/x", "b/y"])
    p = {"a/x": {"w": np.zeros(1)}, "b/y": {"w": np.zeros(1)}}
    g = {"a/x": {"w": np.array([1.0])}, "b/y": {"w": np.array([1.0])}}
    out = opt.update(p, g)
    # a: 1*10 = 10 -> per-group clip to 2; b: 1. Global norm sqrt(4 + 1) -> scale 1/sqrt(5)
    np.testing.assert_allclose(out["a/x"]["w"], [-2.0 / math.sqrt(5)], rtol=1e-12)
    np.testing.assert_allclose(out["b/y"]["w"], [-1.0 / math.sqrt(5)], rtol=1e-12)
    st = opt.last_stats["a/x"]
    assert st["grad_multiplier"] == 10.0 and st["grad_norm_raw"] == 1.0 and abs(st["grad_norm_applied"] - 2 / math.sqrt(5)) < 1e-12
    assert abs(st["update_norm"] - 2 / math.sqrt(5)) < 1e-12


def test_group_weight_decay_is_decoupled_and_per_group():
    opt = Optimizer.from_config({"kind": "sgd", "lr": 0.1, "weight_decay": 0.0, "param_groups": [{"match": "a/*", "weight_decay": 0.5}]})
    opt.bind_groups(["a/x", "b/y"])
    p = {"a/x": {"w": np.array([2.0])}, "b/y": {"w": np.array([2.0])}}
    g = {"a/x": {"w": np.array([1.0])}, "b/y": {"w": np.array([1.0])}}
    out = opt.update(p, g)
    assert out["a/x"]["w"][0] == 2.0 * (1 - 0.1 * 0.5) - 0.1 and out["b/y"]["w"][0] == 2.0 - 0.1


@pytest.mark.parametrize("cfg, match", [
    ({"kind": "adam", "lr": 0.1, "param_groups": [{"match": "missing/*"}]}, "match no owned"),
    ({"kind": "adam", "lr": 0.1, "param_groups": [{"match": "a/*", "lr": 1.0, "lr_scale": 0.1}]}, "not both"),
    ({"kind": "adam", "lr": 0.1, "param_groups": [{"match": "a/*", "momentum": 0.9}]}, "unknown keys"),
    ({"kind": "adam", "lr": 0.1, "schedule": {"kind": "cosine", "warmup_steps": 5}}, "total_steps"),
    ({"kind": "adam", "lr": 0.1, "schedule": {"kind": "exp"}}, "schedule.kind"),
    ({"kind": "adam", "lr": 0.1, "betas": [0.9, 0.99]}, "unknown keys"),
])
def test_param_group_rules_fail_closed(cfg, match):
    with pytest.raises(ValidationError, match=match):
        opt = Optimizer.from_config(cfg)
        opt.bind_groups(["a/x", "b/y"])


def test_first_match_wins():
    opt = Optimizer.from_config({"kind": "sgd", "lr": 1.0, "param_groups": [{"match": "a/*", "lr_scale": 0.5}, {"match": "*", "lr_scale": 0.1}]})
    opt.bind_groups(["a/x", "b/y"])
    assert opt.group_lr("a/x") == 0.5 and opt.group_lr("b/y") == 0.1


@needs_jax
def test_B3_mid_schedule_checkpoint_restore_continues_bitwise(registry, scenario, tmp_path):
    from ness.checkpoint import CheckpointStore
    from ness.runtime.system import NessSystem
    from ness.runtime.transaction import PredictionTransaction

    upper = {"id": "upper", "plugin": "tiny_upper_transformer", "config": {"in_dim": 16, "model_dim": 16, "heads": 2}, "inputs": {"main": "substrate://base_ts/state/final"}}
    cap = {"id": "cap", "plugin": "residual_point_cap", "config": {"channels": ["y0", "y1"], "horizons": 4, "features": {"neural": 16}},
           "inputs": {"baseline": "substrate://base_ts/forecast/point", "neural": "module://upper/hidden"}}
    learning = {"rule": "bp_direct", "optimizer": {"kind": "adam", "lr": 0.01, "clip_norm": 1.0,
                                                   "schedule": {"kind": "cosine", "warmup_steps": 2, "total_steps": 6, "min_lr_ratio": 0.1},
                                                   "param_groups": [{"match": "upper/*", "lr_scale": 0.1, "weight_decay": 0.01}]}}
    exp = experiment({"a": arm([BASE, upper, cap], learning=learning)})
    sysm = NessSystem.build(exp, exp.arm("a"), registry, scenario=scenario)
    assert sysm.optimizer.configured and "optimizer" in sysm.manifest().extra["algorithm_specs"]
    reqs = list(scenario.iter_requests("train", tuple(sysm.tasks.values()), {"max_requests": 24}))

    def steps(system, chunk):
        tx = PredictionTransaction(system, "prequential")
        diags = []
        for i in range(0, len(chunk), 4):
            for req in chunk[i:i + 4]:
                tx.predict(req)
                tx.reveal(req.request_id, {q.task_id: scenario.outcome(req.request_id, q.query_id) for q in req.queries})
            diags.append(tx.learn_step())
        return diags

    d = steps(sysm, reqs[:12])                       # 3 updates: warm-up then decay
    ups = [g for g in d[0]["optimizer"]["groups"] if g.startswith("upper/")]
    caps = [g for g in d[0]["optimizer"]["groups"] if g.startswith("cap/")]
    assert ups and caps
    for k, dg in enumerate(d, start=1):
        f = ScheduleSpec("cosine", 2, 6, 0.1).factor(k)
        assert dg["optimizer"]["groups"][ups[0]]["lr"] == pytest.approx(0.001 * f, rel=1e-15)
        assert dg["optimizer"]["groups"][caps[0]]["lr"] == pytest.approx(0.01 * f, rel=1e-15)
    store = CheckpointStore(tmp_path / "ck")
    mid = store.stage_and_publish(sysm.snapshot("learner"), "a", store.head("a"))
    restored = NessSystem.from_snapshot(store.load(mid), registry, scenario)
    assert restored.optimizer.step == 3 and restored.optimizer.config_canonical() == sysm.optimizer.config_canonical()
    steps(sysm, reqs[12:24])
    steps(restored, reqs[12:24])
    for nid in ("upper", "cap"):
        for g, grp in sysm.node_states[nid].params.items():
            for n, a in grp.items():
                np.testing.assert_array_equal(a, restored.node_states[nid].params[g][n])
    # a checkpoint whose optimizer configuration differs from the arm's is refused
    other = copy.deepcopy(exp.raw)
    other["arms"]["a"]["learning"]["optimizer"]["param_groups"][0]["lr_scale"] = 0.2
    snap = store.load(mid)
    snap.spec["experiment"] = other
    from ness.contracts import ContractViolation
    with pytest.raises(ContractViolation, match="optimizer configuration"):
        NessSystem.from_snapshot(snap, registry, scenario)
