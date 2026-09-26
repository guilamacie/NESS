"""End-to-end vertical slice and the substitution proof (addendum §10.3). Every variation is
configuration-only; no core file is touched by any test."""

import copy

import numpy as np
import pytest

from ness.experiments import run_experiment

from ness_test_helpers import BASE, BASELINE_ARM, FULL_ARM, LEARNING, MEMORY, MEMORY_QUERY, PROGRAM, REGIME, SEMANTIC, UPPER, arm, build, cap, experiment, first_requests, needs_jax

pytestmark = needs_jax

CONV = {"id": "base_ts", "plugin": "toy_frozen_conv", "config": {"width": 16, "kernel": 3, "layers": 2, "channels": ["y0", "y1"], "horizons": 4, "seed": 11, "pretrain_steps": 120}, "inputs": {"history": "observation://target_history"}}
MLP = {"id": "upper", "plugin": "tiny_upper_mlp", "config": {"in_dim": 16, "model_dim": 16}, "inputs": {"main": "substrate://base_ts/state/final"}}
IS_REGIME = copy.deepcopy(REGIME); IS_REGIME["plugin"] = "importance_sampling_regime_reasoner"; IS_REGIME["config"]["samples"] = 3000
CAP_INPUTS = {"neural": "module://upper/hidden", "semantic": "semantic://semantic/features", "program": "program://slope_program/value", "posterior": "reasoner://regime/posterior"}
CAP_FEATS = {"neural": 16, "semantic": 5, "program": 2, "posterior": 2}


def test_full_experiment_runs_all_arms_and_reports_paired_contrasts(registry, tmp_path):
    arms = {
        "native_baseline": BASELINE_ARM,
        "neural_only_cap": arm([BASE, UPPER, cap({"neural": 16}, {"neural": "module://upper/hidden"})]),
        "neuro_symbolic_probabilistic_cap": FULL_ARM,
        "with_memory": arm([BASE, UPPER, SEMANTIC, PROGRAM, REGIME, MEMORY_QUERY, cap({**CAP_FEATS, "memory": 8}, {**CAP_INPUTS, "memory": "memory://analogues/evidence"})], memory=MEMORY),
        "provider_substitution": arm([CONV, MLP, SEMANTIC, PROGRAM, REGIME, cap(CAP_FEATS, CAP_INPUTS)]),
    }
    report = run_experiment(experiment(arms), registry, tmp_path)
    by_arm = {r.arm_id: r for r in report.arms.values()}  # report keys are "<arm>@s<seed>"
    assert set(by_arm) == set(arms) and all(k == r.key for k, r in report.arms.items())
    for r in by_arm.values():
        assert r.status == "ok" and np.isfinite(r.test["future_value"]["loss"]) and r.test["future_value"]["n_requests"] == 6
        assert r.checkpoint_manifest is not None and r.restore_check["max_abs_diff"] == 0.0 and r.causal_check == "pass"
    assert by_arm["native_baseline"].n_updates == 0 and by_arm["neural_only_cap"].n_updates == 2
    assert by_arm["with_memory"].memory_snapshot is not None
    assert by_arm["provider_substitution"].node_versions["base_ts"][0] == "toy_frozen_conv"
    assert {k.split("@")[0] for k in report.paired} == set(arms) - {"native_baseline"} and (tmp_path / "report.json").exists()
    ids = [tuple(x["request_id"] for x in r.per_request) for r in by_arm.values()]
    assert all(i == ids[0] for i in ids)  # T11: identical example streams across arms


class TestSubstitutionProof:
    """Addendum §10.3 / T33: every swap is a config change validated by compile + one prediction."""

    def _ok(self, registry, scenario, nodes, **kw):
        sysm = build(registry, arm(nodes, **kw), scenario)
        req = first_requests(scenario, sysm)[0]
        res, rec, _ = sysm.predict(req)
        assert np.all(np.isfinite(res.outputs[req.queries[0].query_id].values))
        return sysm, rec

    def test_swap_lower_substrate(self, registry, scenario):
        sysm, _ = self._ok(registry, scenario, [CONV, UPPER, cap({"neural": 16}, {"neural": "module://upper/hidden"})])
        assert sysm.compiled.node("base_ts").descriptor.plugin_id == "toy_frozen_conv"

    def test_swap_upper_transformer_for_mlp(self, registry, scenario):
        sysm, _ = self._ok(registry, scenario, [BASE, MLP, cap({"neural": 16}, {"neural": "module://upper/hidden"})])
        assert sysm.compiled.node("upper").descriptor.capabilities["architecture"] == "mlp"

    def test_toggle_early_state_residual(self, registry, scenario):
        with_res, _ = self._ok(registry, scenario, [BASE, UPPER, cap({"neural": 16}, {"neural": "module://upper/hidden"})])
        no_res_upper = copy.deepcopy(UPPER); no_res_upper["inputs"]["main"] = "substrate://base_ts/state/final"
        without, _ = self._ok(registry, scenario, [BASE, no_res_upper, cap({"neural": 16}, {"neural": "module://upper/hidden"})])
        assert "upper.main#merge:gated_add" in with_res.edge_params and not without.edge_params
        assert with_res.compiled.resolved_wiring()["upper"]["main"] != without.compiled.resolved_wiring()["upper"]["main"]

    def test_residual_merge_operators_are_deterministic_and_distinct(self, registry, scenario):  # T35
        outs = {}
        for op in ("add", "gated_add", "weighted_sum", "concat"):
            u = copy.deepcopy(UPPER)
            u["inputs"]["main"]["merge"] = op
            if op == "concat":
                u["config"]["in_dim"] = 32
            sysm, rec = self._ok(registry, scenario, [BASE, u, cap({"neural": 16}, {"neural": "module://upper/hidden"})])
            outs[op] = rec.port_values[("upper", "main#assembled")].payload
            _, rec2, _ = sysm.predict(first_requests(scenario, sysm)[0])
            np.testing.assert_array_equal(outs[op], rec2.port_values[("upper", "main#assembled")].payload)
        assert outs["concat"].shape[-1] == 32 and not np.allclose(outs["add"], outs["gated_add"])

    def test_enable_disable_symbolic_program(self, registry, scenario):
        self._ok(registry, scenario, [BASE, UPPER, SEMANTIC, PROGRAM, cap({"neural": 16, "semantic": 5, "program": 2}, {k: CAP_INPUTS[k] for k in ("neural", "semantic", "program")})])
        self._ok(registry, scenario, [BASE, UPPER, SEMANTIC, cap({"neural": 16, "semantic": 5}, {k: CAP_INPUTS[k] for k in ("neural", "semantic")})])

    def test_swap_probabilistic_reasoner(self, registry, scenario):  # T37 at system level
        exact, r1 = self._ok(registry, scenario, [BASE, UPPER, SEMANTIC, PROGRAM, REGIME, cap(CAP_FEATS, CAP_INPUTS)])
        approx, r2 = self._ok(registry, scenario, [BASE, UPPER, SEMANTIC, PROGRAM, IS_REGIME, cap(CAP_FEATS, CAP_INPUTS)])
        p1, p2 = r1.port_values[("regime", "posterior")].payload, r2.port_values[("regime", "posterior")].payload
        np.testing.assert_allclose(p1.weights, p2.weights, atol=0.05)
        assert p1.weight_semantics == "exact_posterior" and p2.weight_semantics == "approximate_posterior"
        assert "effective_sample_size" in r2.node_diagnostics["regime"]

    def test_enable_disable_memory(self, registry, scenario):
        with_mem, _ = self._ok(registry, scenario, [BASE, UPPER, MEMORY_QUERY, cap({"neural": 16, "memory": 8}, {"neural": "module://upper/hidden", "memory": "memory://analogues/evidence"})], memory=MEMORY)
        without, _ = self._ok(registry, scenario, [BASE, UPPER, cap({"neural": 16}, {"neural": "module://upper/hidden"})])
        assert with_mem.memory is not None and without.memory is None

    def test_change_cap_writer_and_consumer(self, registry, scenario):
        bounded = cap({"neural": 16}, {"neural": "module://upper/hidden"}, writer={"plugin": "point_residual_writer", "config": {"bound": 0.5}}, consumer={"plugin": "concat_linear_consumer", "config": {"hidden": 8}})
        sysm, _ = self._ok(registry, scenario, [BASE, UPPER, bounded])
        assert sysm.compiled.node("cap").module.writer.bound == 0.5 and "w1" in sysm.node_states["cap"].params["consumer"]

    def test_quantile_task_and_cap(self, registry, scenario):
        base_q = copy.deepcopy(BASE); base_q["config"]["quantile_levels"] = [0.1, 0.5, 0.9]
        qcap = {"id": "cap", "plugin": "residual_quantile_cap", "config": {"channels": ["y0", "y1"], "horizons": 4, "levels": [0.1, 0.5, 0.9], "features": {"neural": 16}},
                "inputs": {"baseline": "substrate://base_ts/forecast/quantiles", "neural": "module://upper/hidden"}}
        tasks = [{"id": "future_value", "plugin": "quantile_forecast_task", "config": {"horizons": 4, "channels": ["y0", "y1"], "levels": [0.1, 0.5, 0.9]}}]
        exp = experiment({"q": arm([base_q, UPPER, qcap])}, tasks=tasks)
        from ness.runtime.system import NessSystem
        sysm = NessSystem.build(exp, exp.arm("q"), registry, scenario)
        req = first_requests(scenario, sysm)[0]
        res, rec, _ = sysm.predict(req)
        fc = res.outputs[req.queries[0].query_id]
        assert fc.forecast_type == "quantile" and fc.is_monotone()
        np.testing.assert_array_equal(fc.values, rec.port_values[("base_ts", "forecast/quantiles")].payload.values)  # zero-init parity
        assert sysm.learn.__self__ is sysm  # trainable path wired

    def test_restore_exact_composition_from_manifest(self, registry, scenario, tmp_path):
        from ness.checkpoint import CheckpointStore
        from ness.runtime.system import NessSystem
        sysm = build(registry, arm([CONV, MLP, SEMANTIC, PROGRAM, IS_REGIME, cap(CAP_FEATS, CAP_INPUTS)]), scenario)
        store = CheckpointStore(tmp_path)
        mid = store.stage_and_publish(sysm.snapshot(), "x", None)
        restored = NessSystem.from_snapshot(store.load(mid), registry, scenario)
        assert restored.compiled.resolved_wiring() == sysm.compiled.resolved_wiring()
        assert [c.plugin_id for c in restored.manifest().components] == [c.plugin_id for c in sysm.manifest().components]
