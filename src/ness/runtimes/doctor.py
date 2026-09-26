"""``ness doctor``: unambiguous runtime diagnostics as text and machine-readable JSON."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Any

from .. import __version__
from .bootstrap import RuntimeRequest, bootstrap, current_report, jax_backend_initialized

_SAFE_ENV = ("JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "JAX_ENABLE_X64",
             "FABRICPC_SKIP_XLA_FLAGS", "FABRICPC_DISABLE_TRITON_GEMM", "CUDA_VISIBLE_DEVICES", "TF_CPP_MIN_LOG_LEVEL")


def run_doctor(backend: str = "auto", platform: str | None = None, probe: bool = True, profile: str = "development") -> dict[str, Any]:
    """Bootstrap the requested backend (first thing in the process), then report."""
    have_jax = importlib.util.find_spec("jax") is not None
    have_fpc = importlib.util.find_spec("fabricpc") is not None
    if backend == "auto":
        backend = "fabricpc" if have_fpc else ("jax" if have_jax else "numpy")
    out: dict[str, Any] = {"ness": __version__, "python": sys.version.split()[0], "requested_backend": backend, "requested_platform": platform,
                           "jax_installed": have_jax, "fabricpc_installed": have_fpc, "jax_initialized_before_doctor": jax_backend_initialized(),
                           "environment": {k: os.environ[k] for k in _SAFE_ENV if k in os.environ}, "errors": []}
    try:
        rep = bootstrap(RuntimeRequest(backend=backend, platform=platform, profile=profile))
        out["bootstrap"] = rep.canonical()
    except Exception as exc:
        out["errors"].append(f"bootstrap: {type(exc).__name__}: {exc}")
        rep = current_report()
        out["bootstrap"] = rep.canonical() if rep else None
    if backend in ("jax", "fabricpc") and have_jax:
        try:
            import jax
            out["jax"] = {"version": jax.__version__, "default_backend": jax.default_backend(), "device_count": jax.device_count(),
                          "devices": [{"id": d.id, "platform": d.platform, "kind": getattr(d, "device_kind", d.platform)} for d in jax.devices()],
                          "process_index": jax.process_index(), "process_count": jax.process_count(), "x64": bool(jax.config.jax_enable_x64)}
            try:
                from jax._src.xla_bridge import backends_are_initialized  # diagnostic only
                out["jax"]["backend_initialized"] = bool(backends_are_initialized())
            except Exception:
                pass
        except Exception as exc:
            out["errors"].append(f"jax diagnostics: {type(exc).__name__}: {exc}")
    if backend == "fabricpc" and have_fpc:
        try:
            from ..backends.fabricpc.version import describe_installed
            out["fabricpc"] = describe_installed()
            if probe and not out["errors"]:
                from ..backends.fabricpc.capabilities import probe_capabilities
                out["fabricpc"]["capabilities"] = probe_capabilities().canonical()
        except Exception as exc:
            out["errors"].append(f"fabricpc diagnostics: {type(exc).__name__}: {exc}")
    from ..inference import INFERENCE_PROFILES
    from ..learning import LEARNING_PROFILES
    out["profiles"] = {"inference": INFERENCE_PROFILES.canonical(), "learning": LEARNING_PROFILES.canonical()}
    from ..runtime.system import dependency_lock
    out["dependency_lock"] = dependency_lock()
    return out


def format_doctor(d: dict[str, Any]) -> str:
    lines = [f"ness {d['ness']}  python {d['python']}  requested backend {d['requested_backend']} platform {d['requested_platform']}",
             f"jax installed: {d['jax_installed']}   fabricpc installed: {d['fabricpc_installed']}   jax initialised before doctor: {d['jax_initialized_before_doctor']}"]
    b = d.get("bootstrap")
    if b:
        lines.append(f"bootstrap: backend={b['backend']} jax_backend={b['jax_backend']} setup_before_backend_init={b['setup_before_backend_init']} process {b['process_index']}/{b['process_count']}")
        for k, v in b["versions"].items():
            lines.append(f"  {k:20s} {v}")
        if b["devices"]:
            dv = b["devices"]
            lines.append(f"  devices requested={dv['requested']}  visible={[(x['id'], x['platform'], x['kind']) for x in dv['visible']]}")
            lines.append(f"          selected={[x['id'] for x in dv['selected']]} realized={[x['id'] for x in dv['realized']]} fallback_applied={dv['fallback_applied']}")
        for n in b["notes"]:
            lines.append(f"  note: {n}")
    if d.get("environment"):
        lines.append("environment (JAX/XLA/FabricPC flags only): " + ", ".join(f"{k}={v}" for k, v in d["environment"].items()))
    if "fabricpc" in d:
        f = d["fabricpc"]
        lines.append(f"fabricpc: installed {f.get('installed_version')} family {f.get('family')} adapter {f.get('adapter')} status {f.get('status')} ({f.get('reason', '')})")
        caps = f.get("capabilities")
        if caps:
            lines.append("  capabilities:")
            for c in caps["capabilities"]:
                lines.append(f"    {c['name']:36s} {c['status']:12s} {c['evidence']}")
    lines.append("dependency lock: " + ", ".join(f"{k}={v}" for k, v in d["dependency_lock"].items()))
    for e in d["errors"]:
        lines.append(f"ERROR: {e}")
    return "\n".join(lines)
