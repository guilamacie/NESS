"""NESS JAX backend: batched (vmap) evaluation equals per-item evaluation exactly."""

import numpy as np
import pytest

from ness_test_helpers import FULL_ARM, build, first_requests, needs_jax

pytestmark = needs_jax


def test_batched_objective_equals_per_item(registry, scenario):
    from ness.learning import BatchItem
    sysm = build(registry, FULL_ARM, scenario)
    items = []
    for req in first_requests(scenario, sysm, "train", 4):
        _, rec, _ = sysm.predict(req, mode="train")
        items.append(BatchItem(rec, {q.task_id: scenario.outcome(req.request_id, q.query_id) for q in req.queries}))
    region, owned = sysm.region(), sysm.owned_groups()
    l1, g1, d1 = region.loss_and_grads(items, sysm.node_states, sysm.edge_params, owned, sysm.tasks)
    l2, g2, d2 = region.loss_and_grads_batched(items, sysm.node_states, sysm.edge_params, owned, sysm.tasks)
    assert abs(l1 - l2) < 1e-10 and d2["mode"] == "batched"
    for gid in g1:
        for n in g1[gid]:
            np.testing.assert_allclose(g1[gid][n], g2[gid][n], rtol=1e-9, atol=1e-12)
