import json

import numpy as np
import pytest

from ness.checkpoint import CheckpointStore, LocalBlobStore
from ness.contracts import CheckpointError, ContractViolation
from ness.runtime.system import NessSystem
from ness.runtime.transaction import PredictionTransaction

from ness_test_helpers import BASELINE_ARM, FULL_ARM, MEMORY, MEMORY_QUERY, BASE, UPPER, SEMANTIC, PROGRAM, REGIME, arm, build, cap, first_requests, needs_jax


def _train_a_bit(sysm, scenario, n=4):
    tx = PredictionTransaction(sysm, "prequential")
    for req in first_requests(scenario, sysm, "train", n):
        tx.predict(req)
        tx.reveal(req.request_id, {q.task_id: scenario.outcome(req.request_id, q.query_id) for q in req.queries})
    return tx.learn_step()


def _preds(sysm, scenario, n=3):
    return [sysm.predict(r)[0].outputs[r.queries[0].query_id].values for r in first_requests(scenario, sysm, "test", n)]


class TestBlobStore:
    def test_content_addressed_and_checksummed(self, tmp_path):
        bs = LocalBlobStore(tmp_path)
        d = bs.put_array(np.arange(5.0))
        assert bs.exists(d) and np.array_equal(bs.get_array(d), np.arange(5.0))
        (tmp_path / d[:2] / d).write_bytes(b"corrupt")
        with pytest.raises(CheckpointError, match="checksum"):
            bs.get(d)


class TestWholeSystemCheckpoint:
    def test_baseline_roundtrip_identity(self, registry, scenario, tmp_path):  # T27/T40 without jax
        sysm = build(registry, BASELINE_ARM, scenario)
        store = CheckpointStore(tmp_path)
        mid = store.stage_and_publish(sysm.snapshot(), "a", None)
        assert store.head("a") == mid == sysm.manifest().manifest_id
        restored = NessSystem.from_snapshot(store.load(mid), registry, scenario)
        for a, b in zip(_preds(sysm, scenario), _preds(restored, scenario)):
            np.testing.assert_array_equal(a, b)

    def test_T08_publish_is_compare_and_swap(self, registry, scenario, tmp_path):
        sysm = build(registry, BASELINE_ARM, scenario)
        store = CheckpointStore(tmp_path)
        mid = store.stage_and_publish(sysm.snapshot(), "a", None)
        with pytest.raises(CheckpointError, match="conflict"):
            store.stage_and_publish(sysm.snapshot(), "a", None)  # stale expected parent
        staged = store.stage(sysm.snapshot())
        assert staged.exists() and store.head("a") == mid  # staging never moves HEAD

    def test_checkpoint_contains_only_data(self, registry, scenario, tmp_path):
        sysm = build(registry, BASELINE_ARM, scenario)
        store = CheckpointStore(tmp_path)
        mid = store.stage_and_publish(sysm.snapshot(), "a", None)
        idx = json.loads((tmp_path / "manifests" / f"{mid}.json").read_text())
        assert set(idx) >= {"manifest", "components", "edge_params", "spec"}
        for blob in (tmp_path / "blobs").rglob("*"):
            if blob.is_file():
                assert blob.read_bytes()[:6] == b"\x93NUMPY"  # npy arrays only, no pickles

    @needs_jax
    def test_trained_learner_roundtrip_reproduces_predictions_and_optimizer(self, registry, scenario, tmp_path):
        sysm = build(registry, FULL_ARM, scenario)
        _train_a_bit(sysm, scenario)
        store = CheckpointStore(tmp_path)
        mid = store.stage_and_publish(sysm.snapshot("learner"), "full", None)
        restored = NessSystem.from_snapshot(store.load(mid), registry, scenario)
        assert restored.manifest().manifest_id == mid and restored.n_updates == 1
        for a, b in zip(_preds(sysm, scenario), _preds(restored, scenario)):
            np.testing.assert_array_equal(a, b)
        assert restored.optimizer.step == sysm.optimizer.step
        np.testing.assert_array_equal(restored.optimizer.m["cap/consumer"]["weight"], sysm.optimizer.m["cap/consumer"]["weight"])
        # continuing training from the restored learner matches continuing the original
        d1 = _train_a_bit(sysm, scenario)
        d2 = _train_a_bit(restored, scenario)
        assert np.isclose(d1["loss"], d2["loss"])

    @needs_jax
    def test_T27_memory_snapshot_survives_restore(self, registry, scenario, tmp_path):
        a = arm([BASE, UPPER, SEMANTIC, PROGRAM, REGIME, MEMORY_QUERY, cap({"neural": 16, "memory": 8}, {"neural": "module://upper/hidden", "memory": "memory://analogues/evidence"})], memory=MEMORY)
        sysm = build(registry, a, scenario)
        store = CheckpointStore(tmp_path)
        mid = store.stage_and_publish(sysm.snapshot(), "m", None)
        restored = NessSystem.from_snapshot(store.load(mid), registry, scenario)
        assert restored.memory.snapshot_id == sysm.memory.snapshot_id
        for x, y in zip(_preds(sysm, scenario), _preds(restored, scenario)):
            np.testing.assert_array_equal(x, y)

    def test_restore_refuses_incompatible_composition(self, registry, scenario, tmp_path):
        sysm = build(registry, BASELINE_ARM, scenario)
        store = CheckpointStore(tmp_path)
        snap = sysm.snapshot()
        snap.spec["experiment"]["arms"]["a"]["composition"]["nodes"][0]["config"]["width"] = 8  # tamper with the spec
        mid = store.stage_and_publish(snap, "a", None)
        with pytest.raises(ContractViolation):
            NessSystem.from_snapshot(store.load(mid), registry, scenario)
