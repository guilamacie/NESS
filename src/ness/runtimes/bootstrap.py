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
             "FABRICPC_SKIP_XLA_FLAGS", "FABRICPC_DISABLE_TRITON_GEMM", "CUDA_VISIBLE_DEVICES", "TF_CPP_MIN_LOG_LEVEL",
             "JAX_DEFAULT_MATMUL_PRECISION", "NVIDIA_TF32_OVERRIDE")

# ---- module runtimes -> the backend a process must bootstrap for them (ADR-0015) --------------
# ``jax_inference`` (and ``fabricpc_inference``) are *inference-only* runtimes: the module computes
# with that backend but is never part of a differentiable region (its outputs are constants to
# every learning rule, exactly like ``numpy`` modules). Unknown runtimes (e.g. ``torch``) need no
# NESS-managed backend and rank as ``numpy``; their plugins still declare ``requires``.
RUNTIME_BACKEND = {"numpy": "numpy", "host": "numpy", "jax": "jax", "jax_inference": "jax", "fabricpc": "fabricpc",
                   "fabricpc_inference": "fabricpc"}
INFERENCE_ONLY_RUNTIMES = frozenset({"jax_inference", "fabricpc_inference"})
_BACKEND_RANK = {"numpy": 0, "jax": 1, "fabricpc": 2}


def backend_for_runtime(runtime: str | None) -> str:
    return RUNTIME_BACKEND.get(runtime or "numpy", "numpy")


def backend_for_requires(requires: tuple[str, ...] | list[str] | None) -> str:
    req = set(requires or ())
    return "fabricpc" if "fabricpc" in req else ("jax" if "jax" in req else "numpy")


def max_backend(*backends: str) -> str:
    return max(backends or ("numpy",), key=lambda b: _BACKEND_RANK[b])


def node_backend(runtime: str | None, requires: tuple[str, ...] | list[str] | None = ()) -> str:
    """The one rule both launch paths use: the more capable of what the module's runtime needs
    and what its declared ``requires`` imply."""
    return max_backend(backend_for_runtime(runtime), backend_for_requires(requires))


# ---- numerics (ADR-0014) -----------------------------------------------------------------------
MATMUL_PRECISIONS = ("default", "high", "highest")
# XLA flags applied for ``deterministic: true`` (bitwise-reproducible GPU results across processes;
# harmless on CPU). A user flag with a different value is a conflict, never silently overridden.
DETERMINISTIC_XLA_FLAGS = (("--xla_gpu_deterministic_ops", "true"), ("--xla_gpu_autotune_level", "0"))
# XLA flags known to change numerical results; recorded (and, when the user set them, part of
# predictor identity). Curated, not exhaustive: extend when a new flag is shown to matter.
NUMERIC_XLA_FLAGS = ("--xla_gpu_deterministic_ops", "--xla_gpu_autotune_level", "--xla_gpu_enable_triton_gemm", "--xla_gpu_triton_gemm_any",
                     "--xla_gpu_enable_cublaslt", "--xla_gpu_enable_fast_min_max", "--xla_gpu_exhaustive_tiling_search",
                     "--xla_cpu_enable_fast_math", "--xla_cpu_enable_fast_min_max", "--xla_cpu_fast_math_honor_nans",
                     "--xla_cpu_fast_math_honor_infs", "--xla_cpu_fast_math_honor_division", "--xla_cpu_fast_math_honor_functions")
NUMERIC_ENV_KEYS = ("JAX_DEFAULT_MATMUL_PRECISION", "NVIDIA_TF32_OVERRIDE")


def parse_xla_flags(text: str | None) -> dict[str, str | None]:
    """``--name=value`` tokens -> {name: value}; a bare ``--name`` maps to None. Order-preserving."""
    out: dict[str, str | None] = {}
    for tok in (text or "").split():
        name, eq, val = tok.partition("=")
        out[name] = val if eq else None
    return out


def numeric_xla_flags(text: str | None) -> dict[str, str | None]:
    return {k: v for k, v in parse_xla_flags(text).items() if k in NUMERIC_XLA_FLAGS}


@dataclass(frozen=True, slots=True)
class NumericsSpec:
    """``runtime.numerics``: matrix-product precision and deterministic XLA, applied by the
    bootstrap before the first JAX computation."""

    matmul_precision: str | None = None      # default | high | highest (None: leave JAX's setting alone)
    deterministic: bool = False

    def __post_init__(self) -> None:
        if self.matmul_precision is not None and self.matmul_precision not in MATMUL_PRECISIONS:
            raise ValidationError(f"runtime.numerics.matmul_precision must be one of {MATMUL_PRECISIONS}, got {self.matmul_precision!r}")

    @classmethod
    def from_config(cls, d: Any) -> "NumericsSpec | None":
        if d is None:
            return None
        if not isinstance(d, dict):
            raise ValidationError(f"runtime.numerics must be a mapping, got {type(d).__name__}")
        unknown = set(d) - {"matmul_precision", "deterministic"}
        if unknown:
            raise ValidationError(f"runtime.numerics has unknown keys {sorted(unknown)}; allowed: matmul_precision, deterministic")
        det = d.get("deterministic", False)
        if not isinstance(det, bool):
            raise ValidationError(f"runtime.numerics.deterministic must be true/false, got {det!r}")
        return cls(d.get("matmul_precision"), det)

    def canonical(self) -> dict:
        return {"matmul_precision": self.matmul_precision, "deterministic": self.deterministic}


@dataclass(frozen=True, slots=True)
class RuntimeRequest:
    backend: str = "numpy"                       # numpy | jax | fabricpc
    platform: str | None = None                  # cpu | gpu | None (auto)
    devices: DeviceSpec = field(default_factory=DeviceSpec)
    x64: bool = True
    profile: str = "development"                 # development | scientific (scientific fails closed harder)
    numerics: NumericsSpec | None = None         # runtime.numerics; None = not declared (v0.2 behaviour)

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
        return cls(d.get("backend", "numpy"), platform, dev, bool(d.get("x64", True)), d.get("profile", "development"),
                   NumericsSpec.from_config(d.get("numerics")))

    def canonical(self) -> dict:
        out = {"backend": self.backend, "platform": self.platform, "devices": self.devices.canonical(), "x64": self.x64, "profile": self.profile}
        if self.numerics is not None:
            out["numerics"] = self.numerics.canonical()
        return out


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
    numerics: dict[str, Any] | None = None       # effective numerics record (see numerics_identity)

    def canonical(self) -> dict:
        out = {"request": self.request.canonical(), "backend": self.backend, "versions": dict(sorted(self.versions.items())),
               "environment": dict(sorted(self.environment.items())), "devices": self.devices.canonical() if self.devices else None,
               "jax_backend": self.jax_backend, "setup_before_backend_init": self.setup_before_backend_init,
               "warnings": list(self.warnings), "notes": list(self.notes), "process_index": self.process_index, "process_count": self.process_count}
        if self.numerics is not None:
            out["numerics"] = self.numerics
        return out


def numerics_identity(report: "RuntimeReport | None") -> dict[str, Any] | None:
    """The numerics facts that belong to predictor identity: the declared ``runtime.numerics``,
    the effective JAX matmul precision, and the numeric XLA flags / environment variables *the
    user* set. ``None`` when nothing was declared and the user environment sets nothing numeric,
    so manifests of such systems are unchanged from v0.2. Flags FabricPC's ``setup_jax`` adds are
    recorded in the report but are a function of the FabricPC version, already in identity."""
    if report is None or report.numerics is None:
        return None
    n = report.numerics
    if n.get("declared") is None and not n.get("user_xla_numeric_flags") and not n.get("env"):
        return None
    return {"declared": n.get("declared"), "matmul_precision": n.get("matmul_precision"),
            "user_xla_numeric_flags": n.get("user_xla_numeric_flags"), "applied_xla_flags": n.get("applied_xla_flags"), "env": n.get("env")}


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
    if request.numerics is not None and request.numerics != ex.numerics:
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
    user_xla = os.environ.get("XLA_FLAGS", "")
    user_env = {k: os.environ[k] for k in NUMERIC_ENV_KEYS if k in os.environ}
    if request.backend == "numpy":
        numerics = None
        if request.numerics is not None or numeric_xla_flags(user_xla) or user_env:
            numerics = {"declared": request.numerics.canonical() if request.numerics else None, "matmul_precision": None,
                        "user_xla_numeric_flags": numeric_xla_flags(user_xla), "applied_xla_flags": {}, "effective_xla_numeric_flags": numeric_xla_flags(user_xla),
                        "env": user_env, "applied": False}
            if request.numerics is not None:
                notes.append("runtime.numerics recorded but not applied: numpy backend initialises no JAX")
        report = RuntimeReport(request, "numpy", _versions(), _env(), None, None, None, [], ["numpy backend: no JAX initialisation", *notes], numerics=numerics)
        _STATE["report"] = report
        return report
    applied_flags = _apply_numerics_before_backend(request, user_xla, user_env)

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
    if request.numerics is not None and request.numerics.matmul_precision is not None:
        jax.config.update("jax_default_matmul_precision", request.numerics.matmul_precision)
    numerics = None
    if request.numerics is not None or numeric_xla_flags(user_xla) or user_env or numeric_xla_flags(os.environ.get("XLA_FLAGS")):
        mp = jax.config.jax_default_matmul_precision
        numerics = {"declared": request.numerics.canonical() if request.numerics else None, "matmul_precision": None if mp is None else str(mp),
                    "user_xla_numeric_flags": numeric_xla_flags(user_xla), "applied_xla_flags": applied_flags,
                    "effective_xla_numeric_flags": numeric_xla_flags(os.environ.get("XLA_FLAGS")), "env": user_env, "applied": True}
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
        _STATE["report"] = RuntimeReport(request, request.backend, _versions(), _env(), None, jax.default_backend(), setup_before, caught, notes, numerics=numerics)
        raise
    report = RuntimeReport(request, request.backend, _versions(), _env(), resolution, jax.default_backend(), setup_before, caught, notes,
                           int(jax.process_index()), int(jax.process_count()), numerics)
    _STATE["report"] = report
    return report


def _apply_numerics_before_backend(request: RuntimeRequest, user_xla: str, user_env: dict[str, str]) -> dict[str, str]:
    """Validate ``runtime.numerics`` against the user's environment and merge the deterministic
    XLA flags into ``XLA_FLAGS`` (read once, at backend initialisation). Conflicts fail closed."""
    num = request.numerics
    if num is None:
        return {}
    if num.matmul_precision is not None and "JAX_DEFAULT_MATMUL_PRECISION" in user_env and user_env["JAX_DEFAULT_MATMUL_PRECISION"] != num.matmul_precision:
        raise ContractViolation(f"runtime.numerics.matmul_precision={num.matmul_precision!r} conflicts with JAX_DEFAULT_MATMUL_PRECISION="
                                f"{user_env['JAX_DEFAULT_MATMUL_PRECISION']!r} in the environment; remove one of them")
    applied: dict[str, str] = {}
    if not num.deterministic:
        return applied
    flags = parse_xla_flags(user_xla)
    missing = []
    for name, value in DETERMINISTIC_XLA_FLAGS:
        if name in flags:
            if flags[name] != value:
                raise ContractViolation(f"runtime.numerics.deterministic needs {name}={value} but XLA_FLAGS sets {name}={flags[name]}; "
                                        "remove the conflicting flag or set deterministic: false")
        else:
            missing.append((name, value))
    if missing:
        if jax_backend_initialized():
            raise ContractViolation("runtime.numerics.deterministic: the JAX backend was initialised before the bootstrap, so XLA flags "
                                    f"{[n for n, _ in missing]} can no longer apply; bootstrap before any JAX computation")
        os.environ["XLA_FLAGS"] = " ".join([user_xla.strip(), *(f"{n}={v}" for n, v in missing)]).strip()
        applied = {n: v for n, v in missing}
    return applied


def ensure_bootstrapped(backend: str) -> RuntimeReport:
    """Used by backends and inference-only modules that need a numerical backend. A process that
    already bootstrapped at least ``backend`` is returned as is (fabricpc satisfies jax). When a
    launcher request exists but names a less capable backend, the *recorded request* is honoured
    (platform, devices, x64, profile, numerics) with only the backend raised - never a silent
    re-bootstrap with defaults (ADR-0015). Without any launcher request, defaults are used and the
    implicit bootstrap is recorded in the report's notes."""
    rep = current_report()
    if rep is not None and (backend == "numpy" or _RANK[rep.backend] >= _RANK[backend]):
        return rep
    if rep is not None:
        from dataclasses import replace as _replace
        upgraded = bootstrap(_replace(rep.request, backend=backend))
        upgraded.notes.append(f"backend raised on demand from {rep.backend!r} to {backend!r}; the launcher request's settings were honoured")
        return upgraded
    rep = bootstrap(RuntimeRequest(backend=backend))
    rep.notes.append("implicit bootstrap with defaults (no launcher request)")
    return rep
