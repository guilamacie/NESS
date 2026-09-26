"""Data-parallel correctness on >= 2 devices. CPU legs force two host devices in a subprocess
(``XLA_FLAGS=--xla_force_host_platform_device_count=2``); GPU legs run only when >= 2 GPUs are
visible and are marked ``gpu``/``multidevice``."""

import json
import os
import subprocess
import sys

import pytest

_PROBE = r"""
import json, numpy as np, sys
sys.path.insert(0, {tests_dir!r})
from ness.runtimes.bootstrap import bootstrap, RuntimeRequest
from ness.runtimes.devices import DeviceSpec
rep = bootstrap(RuntimeRequest(backend="fabricpc", platform={platform!r}, devices=DeviceSpec(platform={platform!r}, selection="all", count={count})))
import jax
assert jax.device_count() >= 2, jax.devices()
from ness_test_helpers import BASE, arm, build, first_requests, experiment
from ness.plugin_api import fresh_registry
from ness.learning import BatchItem
reg = fresh_registry(); sc = reg.create("toy_temporal_dataset", {{"n_steps": 400, "context": 32, "horizon": 4, "seed": 1234, "train_fraction": 0.7}})
SEM = {{"id": "semantic", "plugin": "simple_temporal_semantics", "config": {{"window": 8, "channels": 2, "use_neural": False}}, "inputs": {{"raw": "observation://target_history"}}}}
INPUTS = {{"baseline": "substrate://base_ts/forecast/point", "neural": {{"from": "substrate://base_ts/state/final", "boundary": ["mean_pool"]}}, "semantic": "semantic://semantic/features"}}
out = {{"devices": [(d.id, d.platform) for d in jax.devices()], "results": {{}}}}
for rule, profile in (("fabricpc_pc_local", "fabricpc_spc"), ("fabricpc_bp_through_inference", "fabricpc_spc"), ("bp_direct", None)):
    if rule == "bp_direct":
        cap = {{"id": "cap", "plugin": "jax_dense_workspace_cap", "config": {{"channels": ["y0", "y1"], "horizons": 4, "features": {{"neural": 16, "semantic": 4}}, "hidden": [8]}}, "inputs": INPUTS}}
        mk = lambda dp: arm([BASE, SEM, cap], learning={{"rule": "bp_direct", "optimizer": {{"kind": "sgd", "lr": 0.05}}, "rule_config": {{"data_parallel": dp, "batched": True}}}})
    else:
        cap = {{"id": "cap", "plugin": "fabricpc_residual_cap", "config": {{"channels": ["y0", "y1"], "horizons": 4, "features": {{"neural": 16, "semantic": 4}}, "hidden": [8], "inference": {{"profile": profile, "eta_infer": 0.05, "infer_steps": 8}}}}, "inputs": INPUTS}}
        def mk(dp, rule=rule, profile=profile):
            a = arm([BASE, SEM, cap], learning={{"rule": rule, "optimizer": {{"kind": "sgd", "lr": 0.05}}, "rule_config": {{"data_parallel": dp}}}})
            a["inference"] = {{"profile": profile}}
            return a
    single, multi = build(reg, mk(False), sc), build(reg, mk(True), sc)
    rng = np.random.default_rng(1)
    for k, v in single.node_states["cap"].params["workspace"].items():
        w = rng.normal(0, 0.3, v.shape); single.node_states["cap"].params["workspace"][k] = w; multi.node_states["cap"].params["workspace"][k] = w.copy()
    reqs = first_requests(sc, single, "train", 8)   # divisible by 2
    def batch(s):
        items = []
        for r in reqs:
            _, rec, _ = s.predict(r, mode="train"); items.append(BatchItem(rec, {{q.task_id: sc.outcome(r.request_id, q.query_id) for q in r.queries}}))
        return items
    b1, b2 = batch(single), batch(multi)
    r1 = single.rules[rule].gradients(single.learning_context(), b1, single.owned_groups())
    r2 = multi.rules[rule].gradients(multi.learning_context(), b2, multi.owned_groups())
    worst = max(float(np.max(np.abs(r1[1]["cap/workspace"][k] - r2[1]["cap/workspace"][k]))) for k in r1[1]["cap/workspace"])
    single.learn(b1); multi.learn(b2)
    pdiff = max(float(np.max(np.abs(single.predict(r)[0].outputs[r.queries[0].query_id].values - multi.predict(r)[0].outputs[r.queries[0].query_id].values))) for r in reqs[:3])
    # odd batch must be refused, never silently dropped
    try:
        multi.rules[rule].gradients(multi.learning_context(), b2[:7], multi.owned_groups()); odd = "accepted"
    except Exception as exc:
        odd = type(exc).__name__
    out["results"][rule] = {{"loss_single": r1[0], "loss_multi": r2[0], "grad_max_diff": worst, "pred_max_diff_after_update": pdiff,
                             "realized": r2[2].get("realized_devices"), "mode": r2[2].get("mode"), "odd_batch": odd}}
print(json.dumps(out))
"""


def _run(platform: str, count: int, env_extra: dict) -> dict:
    code = _PROBE.format(tests_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), platform=platform, count=count)
    env = {k: v for k, v in os.environ.items() if k not in ("XLA_FLAGS", "JAX_PLATFORMS")}
    env.update(env_extra)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=1500)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _check(out):
    assert len(out["devices"]) >= 2
    for rule, r in out["results"].items():
        assert abs(r["loss_single"] - r["loss_multi"]) < 1e-9, (rule, r)
        assert r["grad_max_diff"] < 1e-9, (rule, r)
        assert r["pred_max_diff_after_update"] < 1e-9, (rule, r)
        assert r["realized"] is not None and len(r["realized"]) == 2, (rule, r)
        assert r["odd_batch"] == "UnsupportedCapability", (rule, r)


@pytest.mark.multidevice
def test_data_parallel_matches_single_device_on_two_forced_cpu_devices():
    out = _run("cpu", 2, {"XLA_FLAGS": "--xla_force_host_platform_device_count=2"})
    _check(out)


def _gpu_count() -> int:
    """Count GPUs in a fresh interpreter: this test process may already be pinned to CPU."""
    env = {k: v for k, v in os.environ.items() if k not in ("XLA_FLAGS", "JAX_PLATFORMS")}
    code = "import jax; print(len([d for d in jax.devices() if d.platform == 'gpu']))"
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
        return int(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else 0
    except Exception:
        return 0


@pytest.mark.gpu
@pytest.mark.multidevice
def test_data_parallel_matches_single_device_on_two_gpus():
    n = _gpu_count()
    if n < 2:
        pytest.skip(f"needs >= 2 GPUs visible to JAX (have {n})")
    out = _run("gpu", 2, {})
    assert all(p == "gpu" for _, p in out["devices"])
    _check(out)
