"""Native FabricPC 0.6 smoke tests (no NESS semantics): distinguish 'FabricPC broken' from
'NESS adapter broken'. Mirrors scratch/native_smoke.py from the audit."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from fabricpc.core.energy import GaussianEnergy, graph_energy
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.inference_epc import EPCInference
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import GraphCycleError, TaskMap, graph
from fabricpc.graph_initialization import initialize_graph_state, initialize_params
from fabricpc.nodes import IdentityNode, Linear, SkipConnection
from fabricpc.training import build_clamps, make_train_step, pc_weight_gradients

from ness.backends.fabricpc.version import classify, describe_installed

KEY = jax.random.PRNGKey(0)
X = jax.random.normal(KEY, (8, 6))
Y = jax.random.normal(jax.random.PRNGKey(1), (8, 2))


def build(inference, skip=False, cyclic=False, unroll=None):
    inp, h1, h2 = IdentityNode(shape=(6,), name="input"), Linear(shape=(8,), name="h1"), Linear(shape=(8,), name="h2")
    out = Linear(shape=(2,), energy=GaussianEnergy(), name="out")
    nodes = [inp, h1, h2, out]
    edges = [Edge(source=inp, target=h1.slot("in")), Edge(source=h1, target=h2.slot("in")), Edge(source=h2, target=out.slot("in"))]
    if skip:
        sk = SkipConnection(shape=(8,), name="res")
        nodes = [inp, h1, h2, sk, out]
        edges = [Edge(source=inp, target=h1.slot("in")), Edge(source=h1, target=h2.slot("in")), Edge(source=h1, target=sk.slot("skip")),
                 Edge(source=h2, target=sk.slot("in")), Edge(source=sk, target=out.slot("in"))]
    if cyclic:
        edges.append(Edge(source=h2, target=h1.slot("in")))
    return graph(nodes=nodes, edges=edges, task_map=TaskMap(x=inp, y=out), inference=inference, unroll=unroll)


def settle(s, params, clamps):
    st0 = initialize_graph_state(s, 8, KEY, clamps=clamps, params=params)
    return float(graph_energy(st0, s)), float(graph_energy(run_inference(params, st0, clamps, s), s))


def test_installed_version_is_in_the_verified_family():
    info = describe_installed()
    assert info["status"] in ("verified", "qualified") and info["family"] == "0.6"
    assert classify("0.7.0").status == "unsupported" and classify("0.6.9").status == "qualified"


def test_feedforward_spc_lowers_energy_and_local_grads_exist():
    s = build(InferenceSGD(0.05, 20))
    p = initialize_params(s, KEY)
    c = build_clamps({"x": X, "y": Y}, s, clamp_target=True)
    e0, e1 = settle(s, p, c)
    assert e1 < e0
    g = pc_weight_gradients(p, run_inference(p, initialize_graph_state(s, 8, KEY, clamps=c, params=p), c, s), s, c)
    assert all(float(jnp.abs(w).sum()) > 0 for n in ("h1", "h2", "out") for w in g.nodes[n].weights.values())


def test_epc_lowers_energy():
    s = build(EPCInference(eta_infer=0.05, infer_steps=5))
    p = initialize_params(s, KEY)
    c = build_clamps({"x": X, "y": Y}, s, clamp_target=True)
    e0, e1 = settle(s, p, c)
    assert e1 < e0


def test_skip_connection_graph_settles():
    s = build(InferenceSGD(0.05, 10), skip=True)
    p = initialize_params(s, KEY)
    c = build_clamps({"x": X, "y": Y}, s, clamp_target=True)
    e0, e1 = settle(s, p, c)
    assert np.isfinite(e1) and e1 <= e0


def test_cyclic_requires_unroll_then_settles():
    with pytest.raises(GraphCycleError):
        build(InferenceSGD(0.05, 5), cyclic=True)
    s = build(InferenceSGD(0.05, 30), cyclic=True, unroll=2)
    assert len(s.schedule) > len(s.node_order)
    p = initialize_params(s, KEY)
    c = build_clamps({"x": X, "y": Y}, s, clamp_target=True)
    e0, e1 = settle(s, p, c)
    assert e1 < e0


@pytest.mark.parametrize("algorithm", ["pc", "backprop"])
def test_public_train_step_runs(algorithm):
    s = build(InferenceSGD(0.05, 10))
    p = initialize_params(s, KEY)
    opt = optax.sgd(0.01)
    step = make_train_step(s, opt, algorithm=algorithm)
    p2, o2, m, st = step(p, opt.init(p), {"x": X, "y": Y}, KEY)
    assert np.isfinite(float(m["energy"])) and st.batch_size == 8


def test_free_prediction_leaves_target_unclamped():
    s = build(InferenceSGD(0.05, 10))
    p = initialize_params(s, KEY)
    c = build_clamps({"x": X, "y": Y}, s, clamp_target=False)
    assert set(c) == {"input"}
    st = run_inference(p, initialize_graph_state(s, 8, KEY, clamps=c, params=p), c, s)
    assert st.nodes["out"].z_mu.shape == (8, 2)
