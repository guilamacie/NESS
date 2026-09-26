"""Pretend heavyweight provider. Importing this module requires a dependency that is not
installed; the registry must refuse to instantiate it (explicitly) while still validating
configurations that do not use it."""

import torch_fake_dependency  # noqa: F401  (intentionally absent)


class HeavyDependencySubstrate:  # pragma: no cover - never importable in tests
    def __init__(self, config):
        raise RuntimeError("unreachable")
