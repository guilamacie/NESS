"""Whole-system checkpoints: stage -> validate -> publish (expected-parent CAS) -> load.

A checkpoint is the ``PredictorManifest`` plus restricted-data fragments per component
(state snapshots), edge parameters, optimizer state, RNG state and the memory snapshot
export. Loading never executes code from the checkpoint: plugins are selected by the
validated manifest and re-instantiated by the registry; their state is data.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import CheckpointError, PredictorManifest, StateSnapshot, content_hash
from .blobs import LocalBlobStore


@dataclass
class SystemSnapshot:
    manifest: PredictorManifest
    components: dict[str, StateSnapshot]                 # node_id -> snapshot (also tasks/scenario under their ids)
    edge_params: dict[str, dict[str, np.ndarray]]
    optimizer: tuple[dict[str, np.ndarray], dict[str, Any]] | None
    rng_state: dict[str, Any]
    memory: tuple[dict[str, np.ndarray], dict[str, Any]] | None
    spec: dict[str, Any]                                 # the arm configuration (composition, tasks, learning, ...)
    kind: str = "serving"                                # serving | learner

    @property
    def manifest_id(self) -> str:
        return self.manifest.manifest_id


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


class CheckpointStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.blobs = LocalBlobStore(self.root / "blobs")
        (self.root / "manifests").mkdir(parents=True, exist_ok=True)
        (self.root / "staging").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ stage
    def stage(self, snap: SystemSnapshot) -> Path:
        index: dict[str, Any] = {
            "schema": "ness.checkpoint/1", "kind": snap.kind, "manifest": snap.manifest.canonical(), "manifest_id": snap.manifest_id,
            "spec": snap.spec, "components": {}, "edge_params": {}, "optimizer": None, "rng_state": snap.rng_state, "memory": None,
        }
        for cid, s in sorted(snap.components.items()):
            index["components"][cid] = {
                "plugin_id": s.plugin_id, "plugin_version": s.plugin_version, "state_schema_id": s.state_schema_id,
                "arrays": {k: self.blobs.put_array(v) for k, v in sorted(s.arrays.items())}, "data": s.data, "content_hash": s.content_hash(),
            }
        for gid, grp in sorted(snap.edge_params.items()):
            index["edge_params"][gid] = {n: self.blobs.put_array(a) for n, a in sorted(grp.items())}
        if snap.optimizer is not None:
            arrays, data = snap.optimizer
            index["optimizer"] = {"arrays": {k: self.blobs.put_array(v) for k, v in sorted(arrays.items())}, "data": data}
        if snap.memory is not None:
            arrays, data = snap.memory
            index["memory"] = {"arrays": {k: self.blobs.put_array(v) for k, v in sorted(arrays.items())}, "data": data}
        staged = self.root / "staging" / f"{snap.manifest_id}.json"
        with open(staged, "w") as f:
            # NOT sort_keys: some configuration mappings (e.g. a reasoner's feature_ports) are
            # order-significant; insertion order is part of the recorded specification.
            json.dump(index, f, indent=1, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        self._validate_index(index)
        return staged

    def _validate_index(self, index: dict[str, Any]) -> None:
        for cid, c in index["components"].items():
            for k, d in c["arrays"].items():
                if not self.blobs.exists(d):
                    raise CheckpointError(f"component {cid} array {k} blob missing")
        for gid, grp in index["edge_params"].items():
            for n, d in grp.items():
                if not self.blobs.exists(d):
                    raise CheckpointError(f"edge params {gid}/{n} blob missing")

    # ------------------------------------------------------------------ publish
    def head(self, arm_id: str) -> str | None:
        p = self.root / f"HEAD.{arm_id}"
        return p.read_text().strip() if p.exists() else None

    def publish(self, staged: Path, arm_id: str, expected_parent: str | None) -> str:
        """Atomically publish a staged checkpoint. Compare-and-swap on the arm's HEAD."""
        index = json.loads(staged.read_text())
        self._validate_index(index)
        manifest_id = index["manifest_id"]
        current = self.head(arm_id)
        if current != expected_parent:
            raise CheckpointError(f"publish conflict for {arm_id}: HEAD is {current}, expected {expected_parent}")
        final = self.root / "manifests" / f"{manifest_id}.json"
        os.replace(staged, final)
        head_tmp = self.root / f"HEAD.{arm_id}.tmp"
        head_tmp.write_text(manifest_id)
        os.replace(head_tmp, self.root / f"HEAD.{arm_id}")
        return manifest_id

    def stage_and_publish(self, snap: SystemSnapshot, arm_id: str, expected_parent: str | None) -> str:
        return self.publish(self.stage(snap), arm_id, expected_parent)

    # ------------------------------------------------------------------ load
    def list_manifests(self) -> tuple[str, ...]:
        return tuple(sorted(p.stem for p in (self.root / "manifests").glob("*.json")))

    def load_index(self, manifest_id: str) -> dict[str, Any]:
        p = self.root / "manifests" / f"{manifest_id}.json"
        if not p.exists():
            raise CheckpointError(f"unknown manifest {manifest_id}")
        index = json.loads(p.read_text())
        if content_hash(index["manifest"]) != manifest_id:
            raise CheckpointError("manifest content does not match its id")
        return index

    def load(self, manifest_id: str) -> SystemSnapshot:
        index = self.load_index(manifest_id)
        comps: dict[str, StateSnapshot] = {}
        for cid, c in index["components"].items():
            arrays = {k: self.blobs.get_array(d) for k, d in c["arrays"].items()}
            s = StateSnapshot(c["plugin_id"], c["plugin_version"], c["state_schema_id"], arrays, c["data"])
            if s.content_hash() != c["content_hash"]:
                raise CheckpointError(f"component {cid} state hash mismatch")
            comps[cid] = s
        edge = {gid: {n: self.blobs.get_array(d) for n, d in grp.items()} for gid, grp in index["edge_params"].items()}
        opt = None
        if index["optimizer"] is not None:
            opt = ({k: self.blobs.get_array(d) for k, d in index["optimizer"]["arrays"].items()}, index["optimizer"]["data"])
        mem = None
        if index["memory"] is not None:
            mem = ({k: self.blobs.get_array(d) for k, d in index["memory"]["arrays"].items()}, index["memory"]["data"])
        m = index["manifest"]
        from ..contracts import ComponentRef  # local import keeps module import light
        manifest = PredictorManifest(
            m["schema_version"], m["protocol_id"], m["arm_id"], m["composition_hash"],
            tuple(ComponentRef(**c) for c in m["components"]), tuple(ComponentRef(**c) for c in m["tasks"]),
            ComponentRef(**m["scenario"]) if m["scenario"] else None, m["memory_snapshot_id"], m["inference_profile"],
            m["learning_profile"], m["credit_map_hash"], m["rng_state_hash"], m["dependency_lock"], m["parent_manifest_id"], m["extra"])
        if manifest.manifest_id != manifest_id:
            raise CheckpointError("reconstructed manifest id mismatch")
        return SystemSnapshot(manifest, comps, edge, opt, index["rng_state"], mem, index["spec"], index["kind"])
