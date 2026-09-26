"""Probe-based ``BackendCapabilities`` for the FabricPC backend.

Every status is decided by an executable probe run *in this process through the adapter*
(the same code path real experiments use). "Function exists" never yields ``verified``.
Capabilities that need hardware absent here report ``unsupported`` with the reason; they
are recorded as verified only in the compatibility matrix entry of a machine that ran them.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ...contracts import BackendCapabilities, Capability, CapabilitySet, CapabilityStatus
from ...runtimes.bootstrap import current_report, ensure_bootstrapped
from . import ADAPTER_VERSION
from .translate import plan_inference
from .version import load_adapter

V, U, X = CapabilityStatus.VERIFIED, CapabilityStatus.UNSUPPORTED, CapabilityStatus.EXPERIMENTAL


def probe_capabilities(quick: bool = True) -> BackendCapabilities:
    ensure_bootstrapped("fabricpc")
    ad, info = load_adapter()
    import jax
    import jax.numpy as jnp

    caps: list[Capability] = []

    def add(name: str, status: CapabilityStatus, evidence: str) -> None:
        caps.append(Capability(name, status, evidence))

    def attempt(name: str, fn, evidence_ok: str):
        try:
            t = time.time()
            extra = fn()
            add(name, V, f"{evidence_ok}{'; ' + extra if extra else ''} ({time.time() - t:.2f}s)")
        except Exception as exc:  # explicit failure is information, not a fallback
            add(name, U, f"probe failed: {type(exc).__name__}: {str(exc)[:160]}")

    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 6))
    y = rng.normal(size=(4, 2))

    def spec(profile, readout="kaiming", **kw):
        return ad.WorkspaceGraphSpec(6, (8, 8), 2, "tanh", plan_inference({"profile": profile, **kw}), 0, readout)

    def basic():
        s = ad.build_structure(spec("fabricpc_feedforward", readout="zeros"))
        p = ad.init_params(s, 0)
        out, d = ad.free_prediction(s, p, x, False)
        assert out.shape == (4, 2)
        return f"readout zero-init => correction {float(jnp.abs(out).max()):.1e}"

    def spc():
        s = ad.build_structure(spec("fabricpc_spc", eta_infer=0.05, infer_steps=20))
        p = ad.init_params(s, 0)
        g, d = ad.clamped_local_gradients(s, p, x, y, True)
        assert float(d["objective"]) < float(d["energy_initial"])
        return f"clamped energy {float(d['energy_initial']):.3f}->{float(d['objective']):.3f}"

    def epc():
        s = ad.build_structure(spec("fabricpc_epc", eta_infer=0.05, infer_steps=5))
        p = ad.init_params(s, 0)
        g, d = ad.clamped_local_gradients(s, p, x, y, True)
        assert float(d["objective"]) < float(d["energy_initial"])
        return f"clamped energy {float(d['energy_initial']):.3f}->{float(d['objective']):.3f}"

    def cyclic():
        s = ad.build_structure(spec("fabricpc_spc_recurrent", eta_infer=0.05, infer_steps=20, unroll=2))
        p = ad.init_params(s, 0)
        g, d = ad.clamped_local_gradients(s, p, x, y, True)
        assert len(s.schedule) > len(s.node_order)
        return f"schedule {list(s.schedule)}"

    def skip():
        from fabricpc.core.topology import Edge
        from fabricpc.graph_assembly import TaskMap, graph
        from fabricpc.nodes import IdentityNode, Linear, SkipConnection
        from fabricpc.core.inference import InferenceSGD
        inp, h1, h2 = IdentityNode(shape=(6,), name="input"), Linear(shape=(8,), name="a"), Linear(shape=(8,), name="b")
        sk, out = SkipConnection(shape=(8,), name="res"), Linear(shape=(2,), name="readout")
        s = graph(nodes=[inp, h1, h2, sk, out], edges=[Edge(source=inp, target=h1.slot("in")), Edge(source=h1, target=h2.slot("in")),
                  Edge(source=h1, target=sk.slot("skip")), Edge(source=h2, target=sk.slot("in")), Edge(source=sk, target=out.slot("in"))],
                  task_map=TaskMap(x=inp, y=out), inference=InferenceSGD(0.05, 5))
        p = ad.init_params(s, 0)
        ad.free_prediction(s, p, x, True)
        return ""

    def custom_node():
        from fabricpc.nodes.base import NodeBase, SlotSpec
        from fabricpc.core.types import NodeParams
        from fabricpc.core.topology import Edge
        from fabricpc.graph_assembly import TaskMap, graph
        from fabricpc.nodes import IdentityNode, Linear
        from fabricpc.core.inference import InferenceSGD

        class Scale(NodeBase):
            @staticmethod
            def get_slots():
                return {"in": SlotSpec(name="in", is_multi_input=True)}

            @staticmethod
            def initialize_params(key, node_shape, input_shapes, weight_init, config):
                return NodeParams(weights={k: jnp.ones(()) for k in input_shapes}, biases={})

            @staticmethod
            def predict(params, inputs, state, node_info):
                z = None
                for k, v in inputs.items():
                    z = v * params.weights[k] if z is None else z + v * params.weights[k]
                return z, None

        inp, sc, out = IdentityNode(shape=(6,), name="input"), Scale(shape=(6,), name="s"), Linear(shape=(2,), name="readout")
        s = graph(nodes=[inp, sc, out], edges=[Edge(source=inp, target=sc.slot("in")), Edge(source=sc, target=out.slot("in"))], task_map=TaskMap(x=inp, y=out), inference=InferenceSGD(0.05, 5))
        p = ad.init_params(s, 0)
        ad.clamped_local_gradients(s, p, x, y, True)
        return ""

    def bp_ff():
        s = ad.build_structure(spec("fabricpc_feedforward"))
        p = ad.init_params(s, 0)
        loss, g, d = ad.bp_through_inference(s, p, x, lambda pred: jnp.mean((pred - jnp.asarray(y)) ** 2), False)
        arr = ad.grads_to_arrays(g)
        assert any(np.abs(a).sum() > 0 for a in arr.values())
        return f"loss {float(loss):.3f}"

    def bp_settled():
        s = ad.build_structure(spec("fabricpc_spc", eta_infer=0.05, infer_steps=10))
        p = ad.init_params(s, 0)
        loss, g, d = ad.bp_through_inference(s, p, x, lambda pred: jnp.mean((pred - jnp.asarray(y)) ** 2), True)
        return f"loss {float(loss):.3f} through 10 sPC steps"

    def pc_local():
        s = ad.build_structure(spec("fabricpc_spc", eta_infer=0.05, infer_steps=10))
        p = ad.init_params(s, 0)
        g, d = ad.clamped_local_gradients(s, p, x, y, True)
        arr = ad.grads_to_arrays(g)
        assert any(np.abs(a).sum() > 0 for a in arr.values())
        return f"denominator {int(d['denominator'])}"

    def dtypes():
        s = ad.build_structure(spec("fabricpc_feedforward"))
        p = ad.init_params(s, 0)
        dt = str(next(iter(p.nodes["h0"].weights.values())).dtype)
        return f"params dtype {dt} (x64={bool(jax.config.jax_enable_x64)})"

    def jit_ok():
        s = ad.build_structure(spec("fabricpc_spc", eta_infer=0.05, infer_steps=5))
        p = ad.init_params(s, 0)
        fn = ad.jit(lambda pp, xx: ad.free_prediction(s, pp, xx, True)[0])
        a = np.asarray(fn(p, x))
        b = np.asarray(ad.free_prediction(s, p, x, True)[0])
        assert np.allclose(a, b, atol=1e-9)
        return "jit == eager"

    def ckpt():
        s = ad.build_structure(spec("fabricpc_spc", eta_infer=0.05, infer_steps=5))
        p = ad.init_params(s, 0)
        arrays = ad.params_to_arrays(p)
        p2 = ad.arrays_to_params(s, arrays)
        a = np.asarray(ad.free_prediction(s, p, x, True)[0])
        b = np.asarray(ad.free_prediction(s, p2, x, True)[0])
        assert np.array_equal(a, b)
        return f"{len(arrays)} restricted arrays"

    def replay():
        s = ad.build_structure(spec("fabricpc_spc", eta_infer=0.05, infer_steps=10))
        p = ad.init_params(s, 0)
        a = np.asarray(ad.free_prediction(s, p, x, True)[0])
        b = np.asarray(ad.free_prediction(s, p, x, True)[0])
        assert np.array_equal(a, b)
        return "bitwise equal on this device"

    def diagnostics():
        s = ad.build_structure(spec("fabricpc_spc", eta_infer=0.05, infer_steps=10))
        p = ad.init_params(s, 0)
        _, d = ad.free_prediction(s, p, x, True)
        assert "energy_initial" in d and "energy_final" in d
        return "energy before/after available"

    attempt("basic_graph_execution", basic, "feedforward graph executes through the adapter")
    attempt("custom_node_support", custom_node, "external NodeBase subclass runs in inference + local gradients")
    attempt("skip_connections", skip, "SkipConnection graph settles")
    attempt("recurrent_cyclic_graph", cyclic, "cyclic hidden graph with unroll settles")
    attempt("state_based_pc_inference", spc, "InferenceSGD lowers clamped energy")
    attempt("error_parameterised_pc_inference", epc, "EPCInference lowers clamped energy")
    add("inference_schedule", X, "InferenceSchedule exists in 0.6 (composed ePC->sPC); not exercised by NESS plugins")
    attempt("ordinary_backprop_feedforward", bp_ff, "jax.grad of a task loss through the feedforward FabricPC graph")
    attempt("bp_through_deployed_finite_inference", bp_settled, "jax.grad through initialize_graph_state + run_inference (fori_loop -> scan)")
    attempt("local_pc_parameter_updates", pc_local, "pc_weight_gradients (batch-summed / prediction count)")
    add("nudge_or_equilibrium_propagation_rules", U, "FabricPC 0.6 provides no nudged phase; NESS has not implemented one")
    attempt("dtypes", dtypes, "float64 under NESS x64 bootstrap; float32 is FabricPC's default without x64")
    attempt("jit", jit_ok, "jitted adapter prediction matches eager")
    attempt("checkpoint_restore", ckpt, "params -> restricted arrays -> params reproduces predictions bitwise")
    attempt("deterministic_replay", replay, "same inputs/params -> identical outputs on one device (tolerance-level across devices)")
    attempt("solver_diagnostics", diagnostics, "energy trajectory endpoints reported")
    add("masked_objective", X, "fabricpc_bp_through_inference honours NESS masks; fabricpc_pc_local requires all-true masks")

    rep = current_report()
    devices = list(jax.devices())
    gpus = [d for d in devices if d.platform == "gpu"]
    if gpus:
        def single_gpu():
            s = ad.build_structure(spec("fabricpc_spc", eta_infer=0.05, infer_steps=5))
            p = ad.init_params(s, 0)
            out = ad.free_prediction(s, jax.device_put(p, gpus[0]), jax.device_put(jnp.asarray(x), gpus[0]), True)[0]
            assert gpus[0] in out.sharding.device_set
            return f"{getattr(gpus[0], 'device_kind', 'gpu')}"
        attempt("single_gpu", single_gpu, "prediction executed on a GPU device")
    else:
        add("single_gpu", U, "no GPU device visible to JAX in this process")
    if len(devices) >= 2:
        def data_parallel():
            s = ad.build_structure(spec("fabricpc_spc", eta_infer=0.05, infer_steps=5))
            p = ad.init_params(s, 0)
            mesh = ad.mesh_for(devices[:2])
            xb = rng.normal(size=(8, 6)); yb = rng.normal(size=(8, 2))
            g1, d1 = ad.clamped_local_gradients(s, p, xb, yb, True)
            sh = ad.shard_batch(mesh, {"x": xb, "y": yb})
            g2, d2 = ad.jit(lambda pp, xx, yy: ad.clamped_local_gradients(s, pp, xx, yy, True))(ad.replicate(mesh, p), sh["x"], sh["y"])
            a1, a2 = ad.grads_to_arrays(g1), ad.grads_to_arrays(g2)
            worst = max(float(np.max(np.abs(a1[k] - a2[k]))) for k in a1)
            assert worst < 1e-6, worst
            return f"{len(devices[:2])} x {devices[0].platform}: sharded vs single grads max diff {worst:.1e}"
        attempt("data_parallelism", data_parallel, "batch sharded over a 2-device 'data' mesh; parameters replicated; gradients match single-device")
    else:
        add("data_parallelism", U, "only one JAX device in this process (use XLA_FLAGS=--xla_force_host_platform_device_count=2 on CPU or >=2 GPUs)")
    add("model_parallelism", U, "FabricPC 0.6 reserves a 'model' mesh axis but ships no parameter/model sharding; NESS claims none")
    add("multi_host_execution", U, f"single process (jax.process_count()={jax.process_count()}); not exercised")
    return BackendCapabilities("fabricpc", f"fabricpc {info.installed_version} ({info.status}); adapter {ADAPTER_VERSION}", CapabilitySet(tuple(caps)))
