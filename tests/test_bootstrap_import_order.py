"""Import-order / initialisation regression tests (subprocess-based: a JAX backend cannot be
un-initialised in-process). Core `import ness` must not import JAX or FabricPC, must not
initialise CUDA/XLA, and must not mutate XLA flags."""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

HAS_JAX = importlib.util.find_spec("jax") is not None
HAS_FPC = importlib.util.find_spec("fabricpc") is not None

_CLEAN = {k: v for k, v in os.environ.items() if k not in ("JAX_PLATFORMS", "XLA_FLAGS", "FABRICPC_SKIP_XLA_FLAGS", "FABRICPC_DISABLE_TRITON_GEMM", "XLA_PYTHON_CLIENT_PREALLOCATE")}


def run(code: str, env: dict | None = None) -> dict:
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**_CLEAN, **(env or {})}, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_import_ness_imports_neither_jax_nor_fabricpc():
    out = run("""
import json, os, sys
before = dict(os.environ)
import ness, ness.contracts, ness.composition, ness.plugin_api, ness.symbolic, ness.memory, ness.probabilistic, ness.checkpoint, ness.runtimes.bootstrap
from ness.plugin_api import default_registry
default_registry()            # plugin discovery reads descriptors only
from ness.cli.main import main
main(["plugins"]) ; main(["audit"])
print(json.dumps({"jax": "jax" in sys.modules, "fabricpc": "fabricpc" in sys.modules,
                  "xla_flags_changed": before.get("XLA_FLAGS") != os.environ.get("XLA_FLAGS"),
                  "jax_platforms_set": "JAX_PLATFORMS" in os.environ and "JAX_PLATFORMS" not in before}))
""")
    assert out == {"jax": False, "fabricpc": False, "xla_flags_changed": False, "jax_platforms_set": False}


def test_schema_only_validation_and_checkpoint_inspection_need_no_jax():
    cfg = os.path.join(os.path.dirname(__file__), "..", "examples", "vertical_slice_timeseries", "configs", "vertical_slice.yaml")
    out = run(f"""
import json, sys
from ness.cli.main import main
rc = main(["validate", {cfg!r}, "--schema-only"])
print(json.dumps({{"rc": rc, "jax": "jax" in sys.modules, "fabricpc": "fabricpc" in sys.modules}}))
""")
    assert out == {"rc": 0, "jax": False, "fabricpc": False}


@pytest.mark.skipif(not HAS_JAX, reason="jax not installed")
def test_jax_backend_import_does_not_initialise_backend_until_bootstrap():
    out = run("""
import json, sys
import ness
from ness.runtimes.bootstrap import jax_backend_initialized, bootstrap, RuntimeRequest, current_report
import ness.backends.jax.region as region     # imports jax; must NOT initialise the backend
before = jax_backend_initialized()
rep = bootstrap(RuntimeRequest(backend="jax", platform="cpu"))
after = jax_backend_initialized()
import jax
print(json.dumps({"before": before, "after": after, "platform": jax.default_backend(), "x64": bool(jax.config.jax_enable_x64),
                  "report_backend": rep.backend, "fabricpc": "fabricpc" in sys.modules}))
""")
    assert out["before"] is False and out["after"] is True
    assert out["platform"] == "cpu" and out["x64"] is True and out["report_backend"] == "jax" and out["fabricpc"] is False


@pytest.mark.skipif(not HAS_FPC, reason="fabricpc not installed")
def test_fabricpc_bootstrap_runs_setup_jax_before_backend_init_and_records_env():
    out = run("""
import json, os, sys
import ness
from ness.runtimes.bootstrap import bootstrap, RuntimeRequest
assert "fabricpc" not in sys.modules and "jax" not in sys.modules
rep = bootstrap(RuntimeRequest(backend="fabricpc", platform="cpu"))
import jax
print(json.dumps({"setup_before": rep.setup_before_backend_init, "warnings": rep.warnings, "platform": jax.default_backend(),
                  "jax_platforms_env": os.environ.get("JAX_PLATFORMS"), "xla_flags": os.environ.get("XLA_FLAGS", ""),
                  "fabricpc_version": rep.versions.get("fabricpc"), "devices": [d.id for d in rep.devices.selected]}))
""")
    assert out["setup_before"] is True and out["warnings"] == [] and out["platform"] == "cpu"
    assert out["jax_platforms_env"] == "cpu"                                  # FabricPC's documented behaviour
    assert "--xla_gpu_deterministic_ops" in out["xla_flags"]                  # FabricPC owns these flags, not NESS
    assert out["fabricpc_version"].startswith("0.6.")


@pytest.mark.skipif(not HAS_FPC, reason="fabricpc not installed")
def test_late_fabricpc_bootstrap_is_detected_and_fails_closed_for_scientific_profile():
    out = run("""
import json
import jax, jax.numpy as jnp
jnp.ones(2).sum().block_until_ready()          # backend initialised BEFORE bootstrap (the failure mode)
from ness.runtimes.bootstrap import bootstrap, RuntimeRequest
from ness.contracts import ContractViolation
try:
    bootstrap(RuntimeRequest(backend="fabricpc", platform="cpu", profile="scientific"))
    res = "accepted"
except ContractViolation as exc:
    res = "refused"
print(json.dumps({"scientific": res}))
""")
    assert out["scientific"] == "refused"
    out2 = run("""
import json
import jax, jax.numpy as jnp
jnp.ones(2).sum().block_until_ready()
from ness.runtimes.bootstrap import bootstrap, RuntimeRequest
rep = bootstrap(RuntimeRequest(backend="fabricpc", platform="cpu"))
print(json.dumps({"setup_before": rep.setup_before_backend_init, "noted": any("did not apply" in n for n in rep.notes)}))
""")
    assert out2["setup_before"] is False and out2["noted"] is True


@pytest.mark.skipif(not HAS_FPC, reason="fabricpc not installed")
def test_user_environment_wins_over_request():
    out = run("""
import json, os
from ness.runtimes.bootstrap import bootstrap, RuntimeRequest
rep = bootstrap(RuntimeRequest(backend="fabricpc", platform="cpu"))
print(json.dumps({"xla": os.environ["XLA_FLAGS"], "warn": [w for w in rep.warnings]}))
""", env={"XLA_FLAGS": "--xla_gpu_autotune_level=3"})
    assert "--xla_gpu_autotune_level=3" in out["xla"] and "--xla_gpu_autotune_level=1" not in out["xla"]  # user's value kept


@pytest.mark.skipif(not HAS_JAX, reason="jax not installed")
def test_jax_then_fabricpc_in_one_process_is_an_explicit_error():
    if not HAS_FPC:
        pytest.skip("fabricpc not installed")
    out = run("""
import json
from ness.runtimes.bootstrap import bootstrap, RuntimeRequest
from ness.contracts import ContractViolation
bootstrap(RuntimeRequest(backend="jax", platform="cpu"))
try:
    bootstrap(RuntimeRequest(backend="fabricpc", platform="cpu")); r = "accepted"
except ContractViolation as exc:
    r = "refused"
print(json.dumps({"r": r}))
""")
    assert out["r"] == "refused"


def test_pinned_platform_request_is_satisfied_by_realised_platform():
    """In-process rule: a bootstrap that auto-selected a platform satisfies a later request that pins
    that same platform (judged on what was realised, not on what the first request asked for);
    a different pinned platform, x64 mismatch or lower backend rank are still incompatible."""
    from ness.runtimes.bootstrap import RuntimeReport, RuntimeRequest, _compatible

    def report(**kw):
        req = RuntimeRequest(backend=kw.pop("backend", "jax"), platform=kw.pop("platform", None), x64=kw.pop("x64", True))
        return RuntimeReport(req, req.backend, {}, {}, None, kw.pop("realised", "cpu"), None, [], [])

    assert _compatible(report(), RuntimeRequest(backend="jax", platform="cpu"))
    assert _compatible(report(), RuntimeRequest(backend="jax"))
    assert _compatible(report(backend="fabricpc"), RuntimeRequest(backend="jax", platform="cpu"))
    assert not _compatible(report(), RuntimeRequest(backend="jax", platform="gpu"))
    assert not _compatible(report(platform="gpu", realised="gpu"), RuntimeRequest(backend="jax", platform="cpu"))
    assert not _compatible(report(x64=False), RuntimeRequest(backend="jax", platform="cpu"))
    assert not _compatible(report(), RuntimeRequest(backend="fabricpc"))
    assert not _compatible(report(realised=None), RuntimeRequest(backend="jax", platform="cpu"))
