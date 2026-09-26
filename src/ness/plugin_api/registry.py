"""Plugin registry with Python entry-point discovery.

Descriptors are lightweight (``plugin_id``, kind, version, factory path, declared
optional dependencies). Discovering or validating a configuration never imports a
plugin's implementation; the factory module is imported only on ``create`` and only
after every declared dependency is importable. Missing dependency -> explicit error.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable

from ..contracts import MODULE_KINDS, ContractViolation, PluginDependencyMissing, PluginNotFound

ENTRY_POINT_GROUP = "ness.plugins"


@dataclass(frozen=True, slots=True)
class PluginDescriptor:
    plugin_id: str
    kind: str
    version: str
    factory: str | Callable[..., Any]  # "package.module:Attr" or a callable
    requires: tuple[str, ...] = ()     # importable module names this plugin needs
    summary: str = ""
    provided_by: str = "in-process"
    contract_version: str = "ness.contracts/1"

    def __post_init__(self) -> None:
        if self.kind not in MODULE_KINDS:
            raise ContractViolation(f"plugin {self.plugin_id}: unknown kind {self.kind!r}")
        if isinstance(self.factory, str) and ":" not in self.factory:
            raise ContractViolation(f"plugin {self.plugin_id}: factory must be 'module:attr'")

    def missing_dependencies(self) -> tuple[str, ...]:
        return tuple(r for r in self.requires if importlib.util.find_spec(r.split(".")[0]) is None)

    def canonical(self) -> dict:
        return {"plugin_id": self.plugin_id, "kind": self.kind, "version": self.version, "requires": list(self.requires), "provided_by": self.provided_by}


@dataclass
class PluginRegistry:
    _plugins: dict[str, PluginDescriptor] = field(default_factory=dict)
    _discovered: bool = False
    _discovery_errors: list[str] = field(default_factory=list)

    def register(self, descriptor: PluginDescriptor, replace: bool = False) -> None:
        if descriptor.plugin_id in self._plugins and not replace:
            existing = self._plugins[descriptor.plugin_id]
            if existing.canonical() == descriptor.canonical():
                return
            raise ContractViolation(f"plugin id {descriptor.plugin_id!r} already registered by {existing.provided_by}")
        self._plugins[descriptor.plugin_id] = descriptor

    def discover(self, group: str = ENTRY_POINT_GROUP) -> "PluginRegistry":
        """Load descriptors from installed distributions' entry points (idempotent)."""
        if self._discovered:
            return self
        for ep in md.entry_points(group=group):
            try:
                obj = ep.load()
            except Exception as exc:  # a broken descriptor module must not take the platform down
                self._discovery_errors.append(f"{ep.name}: {exc!r}")
                continue
            dist = getattr(ep, "dist", None)
            provided_by = f"{dist.name} {dist.version}" if dist is not None else "entry-point"
            descs = obj if isinstance(obj, (list, tuple)) else [obj]
            for d in descs:
                if not isinstance(d, PluginDescriptor):
                    self._discovery_errors.append(f"{ep.name}: entry point did not resolve to a PluginDescriptor")
                    continue
                if d.provided_by == "in-process":
                    d = PluginDescriptor(d.plugin_id, d.kind, d.version, d.factory, d.requires, d.summary, provided_by, d.contract_version)
                self.register(d)
        self._discovered = True
        return self

    @property
    def discovery_errors(self) -> tuple[str, ...]:
        return tuple(self._discovery_errors)

    def describe(self, plugin_id: str) -> PluginDescriptor:
        if plugin_id not in self._plugins:
            raise PluginNotFound(f"no plugin registered with id {plugin_id!r}; known: {sorted(self._plugins)}")
        return self._plugins[plugin_id]

    def has(self, plugin_id: str) -> bool:
        return plugin_id in self._plugins

    def ids(self, kind: str | None = None) -> tuple[str, ...]:
        return tuple(sorted(p for p, d in self._plugins.items() if kind is None or d.kind == kind))

    def descriptors(self) -> tuple[PluginDescriptor, ...]:
        return tuple(self._plugins[k] for k in sorted(self._plugins))

    def load_factory(self, plugin_id: str) -> Callable[..., Any]:
        d = self.describe(plugin_id)
        missing = d.missing_dependencies()
        if missing:
            raise PluginDependencyMissing(
                f"plugin {plugin_id!r} requires {list(missing)} which are not installed; "
                f"NESS does not substitute a fallback implementation")
        if callable(d.factory):
            return d.factory
        module_name, attr = d.factory.split(":", 1)
        module = importlib.import_module(module_name)
        try:
            return getattr(module, attr)
        except AttributeError as exc:
            raise PluginNotFound(f"factory {d.factory!r} for plugin {plugin_id!r} not found") from exc

    def create(self, plugin_id: str, config: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        """Instantiate a plugin. Factories that accept a ``registry`` keyword receive this
        registry so composite plugins (e.g. caps) can resolve their own sub-plugins."""
        import inspect
        factory = self.load_factory(plugin_id)
        try:
            accepts_registry = "registry" in inspect.signature(factory).parameters
        except (TypeError, ValueError):
            accepts_registry = False
        if accepts_registry and "registry" not in kwargs:
            kwargs["registry"] = self
        return factory(config or {}, **kwargs)


_DEFAULT: PluginRegistry | None = None


def default_registry(discover: bool = True) -> PluginRegistry:
    """Process-wide registry populated from entry points on first use."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PluginRegistry()
    if discover:
        _DEFAULT.discover()
    return _DEFAULT


def fresh_registry(discover: bool = True) -> PluginRegistry:
    reg = PluginRegistry()
    return reg.discover() if discover else reg
