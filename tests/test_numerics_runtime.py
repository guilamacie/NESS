"""N-05 / N-09 (ADR-0014, ADR-0015): numerics in the runtime contract, inference-only runtimes,
and the launcher request honoured by on-demand bootstraps (acceptance C1-C4). Subprocess-based
where a fresh JAX backend is needed."""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

from ness.contracts import ValidationError
from ness.runtimes.bootstrap import NumericsSpec, RuntimeRequest, node_backend, parse_xla_flags
from ness.runtime.system import required_backend

HAS_JAX = importlib.util.find_spec("jax") is not None
needs_jax = pytest.mark.skipif(not HAS_JAX, reason="jax not installed")
_CLEAN = {k: v for k, v in os.environ.items() if k not in ("JAX_PLATFORMS", "XLA_FLAGS", "FABRICPC_SKIP_XLA_FLAGS", "FABRICPC_DISABLE_TRITON_GEMM",
                                                           "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_DEFAULT_MATMUL_PRECISION", "NVIDIA_TF32_OVERRIDE", "JAX_ENABLE_X64")}

PRELUDE = r'''
import json, os, sys
import numpy as np
from ness.plugin_api import default_registry, PluginDescriptor
from ness.experiments import parse_experiment
from ness.runtime.system import NessSystem
from ness.reference_plugins.frozen_substrates import ToyFrozenTransformer
from ness.runtimes.bootstrap import ensure_bootstrapped, current_report

class JaxInferenceProvider(ToyFrozenTransformer):
    """A frozen provider that computes with JAX but is never differentiated (jax_inference)."""
    plugin_id = "jax_inference_provider"
    runtime = "jax_inference"
    def forward(self, inputs, state, ctx):
        ensure_bootstrapped("jax")
        import jax.numpy as jnp
        out = super().forward(inputs, state, ctx)
        out.ports["state/final"] = np.asarray(jnp.asarray(out.ports["state/final"]) * 1.0)
        return out

class LyingProvider(ToyFrozenTransformer):
    """Declares runtime numpy but needs JAX at prediction time: must fail closed."""
    plugin_id = "lying_provider"
    def forward(self, inputs, state, ctx):
        ensure_bootstrapped("jax")
        return super().forward(inputs, state, ctx)

reg = default_registry()
reg.register(PluginDescriptor("jax_inference_provider", "substrate", "0.1.0", JaxInferenceProvider, ("jax",)))
reg.register(PluginDescriptor("lying_provider", "substrate", "0.1.0", LyingProvider, ()))
BASE = {"id": "base_ts", "plugin": "toy_frozen_transformer", "config": {"width": 16, "heads": 2, "layers": 2, "channels": ["y0", "y1"], "horizons": 4, "seed": 7, "pretrain_steps": 40},
        "inputs": {"history": "observation://target_history"}}
UPPER = {"id": "upper", "plugin": "tiny_upper_transformer", "config": {"in_dim": 16, "model_dim": 16, "heads": 2}, "inputs": {"main": "substrate://base_ts/state/final"}}
CAP = {"id": "cap", "plugin": "residual_point_cap", "config": {"channels": ["y0", "y1"], "horizons": 4, "features": {"neural": 16}},
       "inputs": {"baseline": "substrate://base_ts/forecast/point", "neural": "module://upper/hidden"}}

def build(nodes, runtime, output="module://cap/forecast", learning={"rule": "bp_direct", "optimizer": {"kind": "adam", "lr": 0.01}}):
    d = {"schema_version": "ness.experiment/3", "protocol_id": "t", "runtime": runtime,
         "scenario": {"plugin": "toy_temporal_dataset", "config": {"n_steps": 200, "context": 32, "horizon": 4, "seed": 1, "train_fraction": 0.7}},
         "tasks": [{"id": "future_value", "plugin": "point_forecast_task", "config": {"horizons": 4, "channels": ["y0", "y1"]}}],
         "protocol": {"batch_size": 2, "updates": 1}, "arms": {"a": {"seed": 0, "composition": {"nodes": nodes, "output": {"task": "future_value", "from": output}}}}}
    if learning:
        d["arms"]["a"]["learning"] = learning
    exp = parse_experiment(d)
    return exp, NessSystem.build(exp, exp.arm("a"), reg)

def first_request(system):
    return next(iter(system.scenario.iter_requests("test", tuple(system.tasks.values()), {"max_requests": 1})))
'''


def run(code: str, env: dict | None = None) -> dict:
    proc = subprocess.run([sys.executable, "-c", PRELUDE + code], capture_output=True, text=True, env={**_CLEAN, **(env or {})}, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------ in-process rules
def test_backend_rules_and_numerics_parsing():
    assert required_backend({"jax_inference"}) == "jax" and required_backend({"torch", "numpy"}) == "numpy"
    assert required_backend({"numpy", "fabricpc", "jax"}) == "fabricpc"
    assert node_backend("numpy", ("jax",)) == "jax" and node_backend("jax_inference") == "jax" and node_backend("torch") == "numpy"
    assert RuntimeRequest.from_config({"backend": "jax"}).numerics is None
    assert RuntimeRequest.from_config({"backend": "jax", "numerics": {"matmul_precision": "highest"}}).numerics == NumericsSpec("highest", False)
    assert "numerics" not in RuntimeRequest.from_config({"backend": "jax"}).canonical()
    with pytest.raises(ValidationError, match="unknown keys"):
        RuntimeRequest.from_config({"numerics": {"precision": "highest"}})
    with pytest.raises(ValidationError, match="matmul_precision"):
        RuntimeRequest.from_config({"numerics": {"matmul_precision": "tf32"}})
    with pytest.raises(ValidationError, match="true/false"):
        RuntimeRequest.from_config({"numerics": {"deterministic": "yes"}})
    assert parse_xla_flags("--a=1 --b --c=x") == {"--a": "1", "--b": None, "--c": "x"}


# ------------------------------------------------------------------ C1 / C2
@needs_jax
def test_C1_matmul_precision_is_part_of_the_manifest_identity():
    code = r'''
import jax
exp, s = build([BASE, UPPER, CAP], {"backend": "jax", "platform": "cpu", "numerics": {"matmul_precision": PREC}})
print(json.dumps({"mid": s.manifest().manifest_id, "num": s.manifest().extra["runtime"].get("numerics"), "jaxprec": str(jax.config.jax_default_matmul_precision),
                  "report": s.runtime_report.canonical().get("numerics")}))
'''
    hi = run("PREC = 'highest'\n" + code)
    lo = run("PREC = 'high'\n" + code)
    assert hi["mid"] != lo["mid"]
    assert hi["jaxprec"] == "highest" and lo["jaxprec"] == "high"
    assert hi["num"]["declared"] == {"matmul_precision": "highest", "deterministic": False} and hi["num"]["matmul_precision"] == "highest"
    assert hi["report"]["applied"] is True


@needs_jax
def test_C2_without_numerics_the_runtime_identity_has_the_v02_shape():
    out = run(r'''
exp, s = build([BASE, UPPER, CAP], {"backend": "jax", "platform": "cpu"})
print(json.dumps({"keys": sorted(s.manifest().extra["runtime"]), "report": s.runtime_report.canonical().get("numerics", "absent")}))
''')
    assert out["keys"] == ["backend", "fabricpc", "fabricpc_adapter", "jax", "jaxlib", "platform", "x64"]
    assert out["report"] == "absent"


@needs_jax
def test_user_environment_numerics_enter_identity():
    out = run(r'''
exp, s = build([BASE, UPPER, CAP], {"backend": "jax", "platform": "cpu"})
print(json.dumps({"num": s.manifest().extra["runtime"].get("numerics")}))
''', env={"JAX_DEFAULT_MATMUL_PRECISION": "highest", "XLA_FLAGS": "--xla_cpu_enable_fast_math=false --xla_force_host_platform_device_count=1"})
    assert out["num"]["declared"] is None and out["num"]["matmul_precision"] == "highest"
    assert out["num"]["env"] == {"JAX_DEFAULT_MATMUL_PRECISION": "highest"} and out["num"]["user_xla_numeric_flags"] == {"--xla_cpu_enable_fast_math": "false"}


@needs_jax
def test_deterministic_xla_flags_applied_merged_and_conflicts_fail_closed():
    out = run(r'''
exp, s = build([BASE, UPPER, CAP], {"backend": "jax", "platform": "cpu", "numerics": {"deterministic": True}})
print(json.dumps({"xla": os.environ["XLA_FLAGS"], "applied": s.runtime_report.numerics["applied_xla_flags"]}))
''', env={"XLA_FLAGS": "--xla_force_host_platform_device_count=1"})
    assert out["xla"].split() == ["--xla_force_host_platform_device_count=1", "--xla_gpu_deterministic_ops=true", "--xla_gpu_autotune_level=0"]
    assert out["applied"] == {"--xla_gpu_deterministic_ops": "true", "--xla_gpu_autotune_level": "0"}
    same = run(r'''
exp, s = build([BASE, UPPER, CAP], {"backend": "jax", "platform": "cpu", "numerics": {"deterministic": True}})
print(json.dumps({"applied": s.runtime_report.numerics["applied_xla_flags"]}))
''', env={"XLA_FLAGS": "--xla_gpu_autotune_level=0"})
    assert same["applied"] == {"--xla_gpu_deterministic_ops": "true"}          # an agreeing user flag is kept, not duplicated
    for env, match in (({"XLA_FLAGS": "--xla_gpu_autotune_level=2"}, "conflict|sets"), ({"JAX_DEFAULT_MATMUL_PRECISION": "high"}, "conflicts")):
        err = run(r'''
from ness.contracts import ContractViolation
try:
    build([BASE, UPPER, CAP], {"backend": "jax", "platform": "cpu", "numerics": {"deterministic": True, "matmul_precision": "highest"}})
    print(json.dumps({"err": None}))
except ContractViolation as e:
    print(json.dumps({"err": str(e)}))
''', env=env)
        assert err["err"] is not None
        import re
        assert re.search(match, err["err"])


# ------------------------------------------------------------------ C3 / C4
@needs_jax
def test_C3_inference_only_nodes_bootstrap_jax_with_the_requested_settings():
    out = run(r'''
import jax
from ness.experiments.runner import experiment_backends
prov = dict(BASE, plugin="jax_inference_provider")
exp, s = build([prov], {"backend": "auto", "platform": "cpu", "x64": False}, output="substrate://base_ts/forecast/point", learning=None)
res, rec, _ = s.predict(first_request(s))
print(json.dumps({"backend": s.runtime_report.backend, "x64_req": s.runtime_report.request.x64, "x64_on": bool(jax.config.jax_enable_x64),
                  "manifest": s.manifest().extra["runtime"], "diff": s.compiled.node("base_ts").differentiable, "platform": jax.default_backend(),
                  "runner": sorted(experiment_backends(exp, list(exp.arms), reg, s.scenario)), "notes": current_report().notes}))
''')
    assert out["backend"] == "jax" and out["x64_req"] is False and out["x64_on"] is False and out["platform"] == "cpu"
    assert out["manifest"]["backend"] == "jax" and out["manifest"]["x64"] is False
    assert out["diff"] is False and out["runner"] == ["jax"]
    assert not any("defaults" in n for n in out["notes"])


@needs_jax
def test_C4_on_demand_bootstrap_honours_the_launcher_request():
    out = run(r'''
import jax
from ness.runtimes.bootstrap import bootstrap, RuntimeRequest
bootstrap(RuntimeRequest.from_config({"backend": "numpy", "x64": False, "platform": "cpu"}))
rep = ensure_bootstrapped("jax")
print(json.dumps({"backend": rep.backend, "x64": rep.request.x64, "x64_on": bool(jax.config.jax_enable_x64), "platform": jax.default_backend(), "notes": rep.notes}))
''')
    assert out["backend"] == "jax" and out["x64"] is False and out["x64_on"] is False and out["platform"] == "cpu"
    assert any("raised on demand" in n for n in out["notes"]) and not any("defaults" in n for n in out["notes"])


@needs_jax
def test_a_node_that_hides_its_backend_fails_closed_at_prediction():
    out = run(r'''
from ness.contracts import ContractViolation
liar = dict(BASE, plugin="lying_provider")
exp, s = build([liar], {"backend": "auto", "platform": "cpu"}, output="substrate://base_ts/forecast/point", learning=None)
errs = []
for _ in range(2):
    try:
        s.predict(first_request(s))
        errs.append(None)
    except ContractViolation as e:
        errs.append(str(e))
print(json.dumps({"errs": errs, "built": s.runtime_report.backend}))
''')
    assert out["built"] == "numpy"
    assert all(e is not None and "truthfully" in e for e in out["errs"])   # and it stays failed (no silent success after the upgrade)
