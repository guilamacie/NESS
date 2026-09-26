"""The single owner of numerical-runtime bootstrap.

Nothing else in NESS may query JAX devices at import time, call FabricPC's ``setup_jax``,
or mutate XLA environment. Launchers (CLI, experiment runner, ``NessSystem.build``) call
``bootstrap(RuntimeRequest)`` once; plugins are constructed only afterwards. A second call
with a compatible request is a no-op; an incompatible one is an explicit error, because a
JAX backend cannot be re-initialised in-process.

Order for the FabricPC backend (as FabricPC 0.6 documents it):
  resolve config -> respect user JAX/XLA env -> ``fabricpc.setup_jax(platform)`` before the
  first JAX computation -> inspect devices -> only then construct graphs.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform as _platform
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any

from .. import __version__
from ..contracts import ContractViolation, PluginDependencyMissing, UnsupportedCapability, ValidationError
from .devices import DeviceResolution, DeviceSpec, resolve_devices

BACKENDS = ("numpy", "jax", "fabricpc")
_XLA_KEYS = ("JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "JAX_ENABLE_X64",
             "FABRICPC_SKIP_XLA_FLAGS", "FABRICPC_DISABLE_TRITON_GEMM", "CUDA_VISIBLE_DEVICES", "TF_CPP_MIN_LOG_LEVEL")


@dataclass(frozen=True, slots=True)
class RuntimeRequest:
    backend: str = "numpy"                       # numpy | jax | fabricpc
    platform: str | None = None                  # cpu | gpu | None (auto)
    devices: DeviceSpec = field(default_factory=DeviceSpec)
    x64: bool = True
    profile: str = "development"                 # development | scientific (scientific fails closed harder)

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValidationError(f"runtime.backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.platform not in (None, "cpu", "gpu", "tpu"):
            raise ValidationError(f"runtime.platform must be cpu|gpu|tpu|null, got {self.platform!r}")

    @classmethod
    def from_config(cls, d: dict[str, Any] | None) -> "RuntimeRequest":
        d = dict(d or {})
        dev = DeviceSpec.from_config(d.get("devices"))
        platform = d.get("platform")
        if platform == "auto":
            platform = None
        if platform is None and dev.platform != "any":
            platform = dev.platform
        return cls(d.get("backend", "numpy"), platform, dev, bool(d.get("x64", True)), d.get("profile", "development"))

    def canonical(self) -> dict:
        return {"backend": self.backend, "platform": self.platform, "devices": self.devices.canonical(), "x64": self.x64, "profile": self.profile}


@dataclass
class RuntimeReport:
    request: RuntimeRequest
    backend: str
    versions: dict[str, str]
    environment: dict[str, str]
    devices: DeviceResolution | None
    jax_backend: str | None
    setup_before_backend_init: bool | None     # FabricPC: did setup_jax run before the backend initialised?
    warnings: list[str]
    notes: list[str]
    process_index: int = 0
    process_count: int = 1

    def canonical(self) -> dict:
        return {"request": self.request.canonical(), "backend": self.backend, "versions": dict(sorted(self.versions.items())),
                "environment": dict(sorted(self.environment.items())), "devices": self.devices.canonical() if self.devices else None,
                "jax_backend": self.jax_backend, "setup_before_backend_init": self.setup_before_backend_init,
                "warnings": list(self.warnings), "notes": list(self.notes), "process_index": self.process_index, "process_count": self.process_count}


_STATE: dict[str, Any] = {"report": None}


def current_report() -> RuntimeReport | None:
    return _STATE["report"]


def reset_for_tests() -> None:
    """Forget the bootstrap record (the JAX backend itself cannot be reset in-process)."""
    _STATE["report"] = None


def _versions() -> dict[str, str]:
    v = {"python": _platform.python_version(), "ness": __version__, "platform": _platform.platform()}
    for name in ("numpy", "jax", "jaxlib", "fabricpc", "optax", "orbax-checkpoint", "jax-cuda12-plugin", "jax-cuda13-plugin"):
        try:
            import importlib.metadata as md
            v[name] = md.version(name)
        except Exception:
            v[name] = "absent"
    return v


def _env() -> dict[str, str]:
    return {k: os.environ[k] for k in _XLA_KEYS if k in os.environ}


def jax_backend_initialized() -> bool | None:
    """True/False when detectable, None when JAX is not imported. Uses the same private
    hook FabricPC uses, defensively; it is diagnostic only and never changes behaviour."""
    if "jax" not in sys.modules:
        return None
    try:
        from jax._src.xla_bridge import backends_are_initialized  # type: ignore
        return bool(backends_are_initialized())
    except Exception:
        return None


_RANK = {"numpy": 0, "jax": 1, "fabricpc": 2}


def _realized_platform(report: RuntimeReport) -> str | None:
    """Platform the existing bootstrap actually landed on (cpu|gpu|tpu), if JAX was initialised."""
    if report.jax_backend in ("cpu", "gpu", "tpu"):
        return report.jax_backend
    if report.devices is not None and report.devices.platform in ("cpu", "gpu", "tpu"):
        return report.devices.platform
    return None


def _compatible(existing: RuntimeReport, request: RuntimeRequest) -> bool:
    """An existing bootstrap satisfies a request when it is at least as capable (fabricpc
    implies jax), agrees on x64, and the request does not pin a different platform/device set.
    A pinned platform is judged against the platform the existing bootstrap *realised*: a
    process that auto-selected CPU satisfies a later ``platform: cpu`` request."""
    ex = existing.request
    if _RANK[ex.backend] < _RANK[request.backend]:
        return False
    if ex.x64 != request.x64:
        return False
    if request.platform is not None:
        if ex.platform not in (None, request.platform):
            return False
        realized = _realized_platform(existing)
        if realized != request.platform:
            return False
    return request.devices in (ex.devices, DeviceSpec())


def bootstrap(request: RuntimeRequest | None = None) -> RuntimeReport:
    """Establish the selected numerical environment exactly once."""
    request = request or RuntimeRequest()
    existing = _STATE["report"]
    if existing is not None:
        if request.backend == "numpy" or _compatible(existing, request):
            return existing
        if existing.request.backend == "numpy":
            pass  # numpy bootstrap never initialised JAX; upgrading is fine
        elif existing.request.backend == "jax" and request.backend == "fabricpc":
            raise ContractViolation(
                "runtime already bootstrapped as 'jax' (backend initialised); FabricPC's setup_jax must run before the first JAX computation. "
                "Bootstrap 'fabricpc' first in this process or run the FabricPC arm in a fresh process.")
        else:
            raise ContractViolation(f"runtime already bootstrapped with {existing.request.canonical()}; incompatible request {request.canonical()} cannot re-initialise JAX in-process")

    notes: list[str] = []
    caught: list[str] = []
    if request.backend == "numpy":
        report = RuntimeReport(request, "numpy", _versions(), _env(), None, None, None, [], ["numpy backend: no JAX initialisation"])
        _STATE["report"] = report
        return report

    if importlib.util.find_spec("jax") is None:
        raise PluginDependencyMissing(f"runtime.backend={request.backend!r} requires jax; install ness[{'fabricpc' if request.backend == 'fabricpc' else 'jax'}]")

    setup_before: bool | None = None
    if request.backend == "fabricpc":
        if importlib.util.find_spec("fabricpc") is None:
            raise PluginDependencyMissing("runtime.backend='fabricpc' requires fabricpc; install ness[fabricpc] (GPU: pip install -U 'ness[fabricpc]' 'fabricpc[cuda12]')")
        from ..backends.fabricpc.version import check_installed_version
        check_installed_version(profile=request.profile)  # fail closed on unknown API families
        already = jax_backend_initialized()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import fabricpc  # noqa: WPS433  (the one sanctioned import site outside the adapter)
            platform_arg = {"gpu": "cuda", "cpu": "cpu", "tpu": "tpu"}.get(request.platform or "", None)
            fabricpc.setup_jax(platform_arg)  # respects JAX_PLATFORMS / XLA_FLAGS already in the environment
        caught = [str(x.message) for x in w]
        late = any("after the JAX backend was initialized" in m for m in caught)
        setup_before = not late and (already in (False, None))
        if late:
            msg = "fabricpc.setup_jax ran after the JAX backend initialised; its settings did not apply"
            if request.profile == "scientific":
                raise ContractViolation(msg + " (scientific profile fails closed; bootstrap before any JAX computation)")
            notes.append(msg)
    import jax  # noqa: WPS433

    if request.x64:
        jax.config.update("jax_enable_x64", True)
    if request.backend == "jax" and request.platform is not None:
        env_platform = os.environ.get("JAX_PLATFORMS")
        want = {"gpu": "cuda", "cpu": "cpu", "tpu": "tpu"}[request.platform]
        if env_platform is None:
            if jax_backend_initialized():
                notes.append("JAX backend already initialised before bootstrap; platform request could not be applied")
            else:
                jax.config.update("jax_platforms", want)
        elif env_platform != want:
            notes.append(f"JAX_PLATFORMS={env_platform!r} in the environment wins over runtime.platform={request.platform!r}")
    # First device query: the backend initialises here (the deadline for all settings above).
    try:
        resolution = resolve_devices(request.devices, jax)
    except UnsupportedCapability:
        _STATE["report"] = RuntimeReport(request, request.backend, _versions(), _env(), None, jax.default_backend(), setup_before, caught, notes)
        raise
    report = RuntimeReport(request, request.backend, _versions(), _env(), resolution, jax.default_backend(), setup_before, caught, notes,
                           int(jax.process_index()), int(jax.process_count()))
    _STATE["report"] = report
    return report


def ensure_bootstrapped(backend: str) -> RuntimeReport:
    """Used by backends when a caller skipped the launcher: bootstraps with defaults and
    records that it happened implicitly (visible in diagnostics)."""
    rep = current_report()
    if rep is not None and (rep.backend == backend or backend == "numpy"):
        return rep
    rep = bootstrap(RuntimeRequest(backend=backend))
    rep.notes.append("implicit bootstrap with defaults (no launcher request)")
    return rep
