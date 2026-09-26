"""Root test configuration: fixtures live in ``ness_test_helpers`` (a plain module, so that
sub-directory conftests never shadow it)."""

import pytest

from ness_test_helpers import *  # noqa: F401,F403  (constants and helper functions)
from ness_test_helpers import SCENARIO
from ness.plugin_api import PluginRegistry, fresh_registry


@pytest.fixture(scope="session")
def registry() -> PluginRegistry:
    reg = fresh_registry()
    if not reg.has("toy_frozen_transformer"):
        from ness.reference_plugins import register_all
        register_all(reg)
    return reg


@pytest.fixture(scope="session")
def scenario(registry):
    return registry.create(SCENARIO["plugin"], SCENARIO["config"])
