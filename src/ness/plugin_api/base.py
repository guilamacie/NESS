"""Convenience base class implementing the boring parts of ``MacroModule``."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import (
    IncompatibleVersion,
    ModuleDescriptor,
    ModuleState,
    SemVer,
    StateSnapshot,
    content_hash,
    flatten_params,
    unflatten_params,
)


class BaseModule:
    """Implements config canonicalisation and restricted state snapshot/restore.

    Subclasses set ``plugin_id``, ``plugin_version``, ``module_kind``, ``runtime``,
    ``state_schema_id`` and implement ``describe``/``initialize``/``forward``.
    """

    plugin_id: str = ""
    plugin_version: str = "0.0.0"
    module_kind: str = ""
    runtime: str = "numpy"
    state_schema_id: str = ""
    requires: tuple[str, ...] = ()

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.validate_config()

    # ----- config -------------------------------------------------------------
    def validate_config(self) -> None:  # pragma: no cover - overridden where needed
        return None

    def canonical_config(self) -> dict[str, Any]:
        return dict(sorted(self.config.items()))

    @property
    def config_hash(self) -> str:
        return content_hash({"plugin_id": self.plugin_id, "plugin_version": self.plugin_version, "config": self.canonical_config()})

    # ----- state -------------------------------------------------------------
    def initialize(self, rng: np.random.Generator) -> ModuleState:
        return ModuleState()

    def snapshot_state(self, state: ModuleState) -> StateSnapshot:
        arrays = {f"params/{k}": v for k, v in flatten_params(state.params).items()}
        arrays.update({f"buffers/{k}": np.asarray(v) for k, v in state.buffers.items()})
        return StateSnapshot(
            plugin_id=self.plugin_id,
            plugin_version=self.plugin_version,
            state_schema_id=self.state_schema_id,
            arrays=arrays,
            data={"meta": state.meta, "config_hash": self.config_hash},
        )

    def restore_state(self, snapshot: StateSnapshot) -> ModuleState:
        if snapshot.plugin_id != self.plugin_id:
            raise IncompatibleVersion(f"snapshot belongs to {snapshot.plugin_id}, not {self.plugin_id}")
        if not SemVer.parse(snapshot.plugin_version).compatible_with(SemVer.parse(self.plugin_version)):
            migrated = self.migrate_state(snapshot.plugin_version, snapshot)
            if migrated is None:
                raise IncompatibleVersion(
                    f"{self.plugin_id}: state from version {snapshot.plugin_version} is incompatible with {self.plugin_version} and no migration exists"
                )
            snapshot = migrated
        snapshot.require_schema(self.state_schema_id)
        if snapshot.data.get("config_hash") not in (None, self.config_hash):
            raise IncompatibleVersion(f"{self.plugin_id}: snapshot config hash differs from this instance's config")
        params_flat = {k[len("params/"):]: v for k, v in snapshot.arrays.items() if k.startswith("params/")}
        buffers = {k[len("buffers/"):]: v for k, v in snapshot.arrays.items() if k.startswith("buffers/")}
        return ModuleState(params=unflatten_params(params_flat), buffers=buffers, meta=dict(snapshot.data.get("meta", {})))

    def migrate_state(self, old_version: str, snapshot: StateSnapshot) -> StateSnapshot | None:
        """Override to migrate incompatible state. Default: no migration (fail closed)."""
        return None

    # ----- helpers -----------------------------------------------------------
    def _descriptor(self, **kwargs: Any) -> ModuleDescriptor:
        return ModuleDescriptor(
            plugin_id=self.plugin_id,
            plugin_version=self.plugin_version,
            module_kind=self.module_kind,
            runtime=self.runtime,
            state_schema_id=self.state_schema_id,
            config_hash=self.config_hash,
            requires=self.requires,
            **kwargs,
        )
