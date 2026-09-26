"""Stable contributor-facing surface: protocols, base classes, registry and test kit.

A plugin package depends on ``ness.contracts`` and this package only.
"""

from .base import BaseModule
from .module import DifferentiableModule, MacroModule, ModuleOutputs, PortValue, RuntimeContext
from .registry import ENTRY_POINT_GROUP, PluginDescriptor, PluginRegistry, default_registry, fresh_registry

__all__ = ["BaseModule", "DifferentiableModule", "MacroModule", "ModuleOutputs", "PortValue", "RuntimeContext",
           "ENTRY_POINT_GROUP", "PluginDescriptor", "PluginRegistry", "default_registry", "fresh_registry"]
