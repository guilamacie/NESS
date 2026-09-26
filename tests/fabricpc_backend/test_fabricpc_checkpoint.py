"""Whole-system checkpoints with a FabricPC workspace: restricted arrays only, restore by
manifest, fail closed on a different graph/inference configuration."""

import copy
import json
import sys

import numpy as np
import pytest

from ness_test_helpers import BASE, arm, build, first_requests  # noqa: E402

from ness.checkpoint import CheckpointStore
from ness.contracts import ContractViolation, IncompatibleVersion
from ness.runtime.system import NessSystem
from ness.runtime.transaction import PredictionTransaction

SEM = {"id": "semantic", "plugin": "simple_temporal_semantics", "config": {"window": 8, "channels": 2, "use_neural": False}, "inputs": {"raw": "observation://target_history"}}
INPUTS = {"baseline": "substrate://base_ts/forecast/point", "neural": {"from": "substrate://base_ts/state/final", "boundary": ["mean_pool"]}, "semantic": "semantic://semantic/features"}


def fpc_arm(steps=10):
    a = arm([BASE, SEM, {"id": "cap", "plugin": "fabricpc_residual_cap", "config": {"channels": ["y0", "y1"], "horizons": 4, "features": {"neural": 16, "semantic": 4}, "hidden": [8],
                                                                                 "inference": {"profile": "fabricpc_spc", "eta_infer": 0.05, "infer_steps": steps}}, "inputs": INPUTS}],
            learning={"rule": "workspace_pc_local", "optimizer": {"kind": "adam", "lr": 0.01}})
    a["inference"] = {"profile": "fabricpc_spc"}
    return a


def test_learner_checkpoint_round_trip_restores_predictions_optimizer_and_algorithm_spec(registry, scenario, tmp_path):
    sysm = build(registry, fpc_arm(), scenario)
    tx = PredictionTransaction(sysm, "prequential")
    for req in first_requests(scenario, sysm, "train", 4):
        tx.predict(req)
        tx.reveal(req.request_id, {q.task_id: scenario.outcome(req.request_id, q.query_id) for q in req.queries})
    tx.learn_step()
    store = CheckpointStore(tmp_path)
    mid = store.stage_and_publish(sysm.snapshot("learner"), "f", None)
    idx = json.loads((tmp_path / "manifests" / f"{mid}.json").read_text())
    comp = idx["components"]["cap"]
    assert comp["state_schema_id"] == "ness.cap.fabricpc_residual/1" and all("|" in k for k in comp["arrays"])
    assert comp["data"]["meta"]["fabricpc_version"].startswith("0.6") and "adapter_version" in comp["data"]["meta"]
    for blob in (tmp_path / "blobs").rglob("*"):
        if blob.is_file():
            assert blob.read_bytes()[:6] == b"\x93NUMPY"  # no pickled FabricPC objects anywhere
    restored = NessSystem.from_snapshot(store.load(mid), registry, scenario)
    assert restored.manifest().manifest_id == mid and restored.n_updates == 1
    assert restored.manifest().extra["algorithm_specs"]["cap"] == sysm.manifest().extra["algorithm_specs"]["cap"]
    for r in first_requests(scenario, sysm, "test", 3):
        np.testing.assert_array_equal(sysm.predict(r)[0].outputs[r.queries[0].query_id].values, restored.predict(r)[0].outputs[r.queries[0].query_id].values)
    assert restored.optimizer.step == sysm.optimizer.step


def test_restore_into_different_inference_configuration_fails_closed(registry, scenario, tmp_path):
    sysm = build(registry, fpc_arm(10), scenario)
    store = CheckpointStore(tmp_path)
    snap = sysm.snapshot()
    snap.spec["experiment"]["arms"]["a"]["composition"]["nodes"][2]["config"]["inference"]["infer_steps"] = 20
    mid = store.stage_and_publish(snap, "f", None)
    with pytest.raises(ContractViolation):
        NessSystem.from_snapshot(store.load(mid), registry, scenario)


def test_module_level_restore_rejects_foreign_structure(registry, scenario):
    sysm = build(registry, fpc_arm(10), scenario)
    cap = sysm.compiled.node("cap").module
    snap = cap.snapshot_state(sysm.node_states["cap"])
    other = registry.create("fabricpc_residual_cap", {**cap.config, "hidden": [16]})
    with pytest.raises(IncompatibleVersion):
        other.restore_state(snap)
