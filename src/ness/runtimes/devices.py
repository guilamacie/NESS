"""Device specification and resolution. Experiment code never hard-codes device counts.

Four device sets are distinguished and recorded: *requested* (the spec), *visible* (what
the initialised JAX backend reports), *selected* (the subset the spec resolves to) and
*realized* (what the computation actually ran on, filled in by the executing backend).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contracts import UnsupportedCapability, ValidationError

PLATFORMS = ("cpu", "gpu", "tpu", "any")


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    platform: str = "any"            # cpu | gpu | tpu | any
    selection: str = "all"           # all | single | ids
    ids: tuple[int, ...] = ()        # with selection == "ids"
    count: int | None = None         # exact device count required (None = whatever selection yields)
    fallback: str = "error"          # error | cpu   (never silent)

    def __post_init__(self) -> None:
        if self.platform not in PLATFORMS:
            raise ValidationError(f"devices.platform must be one of {PLATFORMS}, got {self.platform!r}")
        if self.selection not in ("all", "single", "ids"):
            raise ValidationError(f"devices.selection must be all|single|ids, got {self.selection!r}")
        if self.selection == "ids" and not self.ids:
            raise ValidationError("devices.selection=ids requires a non-empty ids list")
        if self.fallback not in ("error", "cpu"):
            raise ValidationError("devices.fallback must be error|cpu")

    @classmethod
    def from_config(cls, d: dict[str, Any] | None) -> "DeviceSpec":
        d = dict(d or {})
        ids = d.get("ids")
        return cls(d.get("platform", "any"), "ids" if ids else d.get("selection", "all"), tuple(int(i) for i in (ids or ())),
                   d.get("count"), d.get("fallback", "error"))

    def canonical(self) -> dict:
        return {"platform": self.platform, "selection": self.selection, "ids": list(self.ids), "count": self.count, "fallback": self.fallback}


@dataclass(frozen=True, slots=True)
class DeviceRecord:
    id: int
    platform: str
    kind: str
    process_index: int

    def canonical(self) -> dict:
        return {"id": self.id, "platform": self.platform, "kind": self.kind, "process_index": self.process_index}


@dataclass
class DeviceResolution:
    requested: DeviceSpec
    visible: tuple[DeviceRecord, ...]
    selected: tuple[DeviceRecord, ...]
    realized: tuple[DeviceRecord, ...] = ()
    fallback_applied: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def platform(self) -> str:
        return self.selected[0].platform if self.selected else "none"

    def canonical(self) -> dict:
        return {"requested": self.requested.canonical(), "visible": [d.canonical() for d in self.visible],
                "selected": [d.canonical() for d in self.selected], "realized": [d.canonical() for d in self.realized],
                "fallback_applied": self.fallback_applied, "notes": list(self.notes)}


def _records(devices: Any) -> tuple[DeviceRecord, ...]:
    out = []
    for d in devices:
        out.append(DeviceRecord(int(d.id), str(d.platform), str(getattr(d, "device_kind", d.platform)), int(getattr(d, "process_index", 0))))
    return tuple(out)


def resolve_devices(spec: DeviceSpec, jax_module: Any) -> DeviceResolution:
    """Resolve a spec against the initialised JAX backend. Fails closed unless the spec
    declares ``fallback: cpu``; a fallback is recorded, never silent."""
    visible = _records(jax_module.devices())
    if spec.platform == "any":
        pool = [d for d in visible if d.platform == jax_module.default_backend()] or list(visible)
    else:
        pool = [d for d in visible if d.platform == spec.platform]
    notes: list[str] = []
    fallback = False
    if not pool:
        if spec.fallback == "cpu":
            cpu_pool = [d for d in _records(jax_module.devices("cpu")) if d.platform == "cpu"]
            if not cpu_pool:
                raise UnsupportedCapability("no CPU devices visible for the declared fallback")
            pool = cpu_pool
            fallback = True
            notes.append(f"requested platform {spec.platform!r} unavailable; declared fallback to cpu applied")
        else:
            raise UnsupportedCapability(
                f"requested devices.platform={spec.platform!r} but JAX sees only {sorted({d.platform for d in visible})}; "
                f"install the matching JAX hardware backend (e.g. pip install -U 'fabricpc[cuda12]') or declare devices.fallback: cpu. "
                f"Diagnose with `ness doctor`.")
    if spec.selection == "single":
        selected = pool[:1]
    elif spec.selection == "ids":
        by_id = {d.id: d for d in pool}
        missing = [i for i in spec.ids if i not in by_id]
        if missing:
            raise UnsupportedCapability(f"requested device ids {list(spec.ids)} but visible {spec.platform} ids are {sorted(by_id)}; missing {missing}")
        selected = [by_id[i] for i in spec.ids]
    else:
        selected = pool
    if spec.count is not None and len(selected) != spec.count:
        raise UnsupportedCapability(
            f"devices.count={spec.count} requested but {len(selected)} {selected[0].platform if selected else ''} device(s) resolved; "
            "a scientific run never silently shrinks its device set")
    return DeviceResolution(spec, visible, tuple(selected), (), fallback, notes)
