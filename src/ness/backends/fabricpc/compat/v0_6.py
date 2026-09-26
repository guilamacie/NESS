"""FabricPC 0.6.x compatibility adapter - the ONLY module that touches FabricPC classes.

API classification (FabricPC 0.6.0):
  public documented   : setup_jax, nodes.{Linear, IdentityNode, SkipConnection}, core.topology.Edge,
                        graph_assembly.{graph, TaskMap, GraphCycleError}, core.inference.{InferenceSGD, run_inference},
                        core.inference_epc.EPCInference, core.energy.{GaussianEnergy, graph_energy},
                        core.activations.{TanhActivation, IdentityActivation}, core.initializers.{ZerosInitializer, ...},
                        graph_initialization.{initialize_params, initialize_graph_state}, training.{build_clamps,
                        pc_weight_gradients, grad_denominator}, core.types.{GraphParams, NodeParams}
  public, weakly documented : GraphStructure.schedule / node_order / config fields (documented in types.py docstrings)
  private             : none used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ....contracts import ContractViolation, UnsupportedCapability, content_hash

import jax  # noqa: E402  (this module is imported only after bootstrap)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

import fabricpc  # noqa: E402
from fabricpc.core.activations import IdentityActivation, TanhActivation  # noqa: E402
from fabricpc.core.energy import GaussianEnergy, graph_energy  # noqa: E402
from fabricpc.core.inference import InferenceSGD, run_inference  # noqa: E402
from fabricpc.core.inference_epc import EPCInference  # noqa: E402
from fabricpc.core.initializers import KaimingInitializer, NormalInitializer, ZerosInitializer  # noqa: E402
from fabricpc.core.topology import Edge  # noqa: E402
from fabricpc.core.types import GraphParams, NodeParams  # noqa: E402
from fabricpc.graph_assembly import GraphCycleError, TaskMap, graph  # noqa: E402
from fabricpc.graph_initialization import GlobalStateInit, initialize_graph_state, initialize_params  # noqa: E402
from fabricpc.nodes import IdentityNode, Linear  # noqa: E402
from fabricpc.training import build_clamps, grad_denominator, pc_weight_gradients  # noqa: E402

from ..translate import InferencePlan  # noqa: E402

FABRICPC_VERSION = fabricpc.__version__
INPUT, READOUT = "input", "readout"


@dataclass(frozen=True, slots=True)
class WorkspaceGraphSpec:
    in_dim: int
    hidden: tuple[int, ...]
    out_dim: int
    activation: str  # tanh | identity
    plan: InferencePlan
    seed: int
    readout_init: str = "zeros"  # zeros (baseline parity) | kaiming (probes / non-residual use)


def _activation(name: str):
    return {"tanh": TanhActivation(), "identity": IdentityActivation()}[name]


def make_inference(plan: InferencePlan):
    if plan.solver == "none":
        return InferenceSGD(eta_infer=plan.eta_infer, infer_steps=0)  # graph() requires an inference object; never run
    if plan.fabricpc_class == "InferenceSGD":
        return InferenceSGD(eta_infer=plan.eta_infer, infer_steps=plan.infer_steps, latent_decay=plan.latent_decay)
    if plan.fabricpc_class == "EPCInference":
        return EPCInference(eta_infer=plan.eta_infer, infer_steps=plan.infer_steps, latent_decay=plan.latent_decay)
    raise UnsupportedCapability(f"no FabricPC 0.6 solver for plan {plan.canonical()}")


def build_structure(spec: WorkspaceGraphSpec):
    """input(Identity) -> hidden Linear(act)* -> readout Linear(identity, zero-init weights).
    With ``plan.cyclic`` the last hidden feeds back into the first hidden (unroll=U)."""
    if not spec.hidden:
        raise ContractViolation("the FabricPC workspace needs at least one hidden layer")
    inp = IdentityNode(shape=(spec.in_dim,), name=INPUT)
    hidden = [Linear(shape=(h,), activation=_activation(spec.activation), name=f"h{i}", weight_init=KaimingInitializer()) for i, h in enumerate(spec.hidden)]
    out = Linear(shape=(spec.out_dim,), activation=IdentityActivation(), energy=GaussianEnergy(), name=READOUT,
                 weight_init=ZerosInitializer() if spec.readout_init == "zeros" else KaimingInitializer())
    nodes = [inp, *hidden, out]
    edges = [Edge(source=inp, target=hidden[0].slot("in"))]
    for a, b in zip(hidden[:-1], hidden[1:]):
        edges.append(Edge(source=a, target=b.slot("in")))
    edges.append(Edge(source=hidden[-1], target=out.slot("in")))
    unroll = None
    if spec.plan.cyclic:
        if len(hidden) < 2:
            raise ContractViolation("fabricpc_spc_recurrent needs >= 2 hidden layers for a lateral feedback edge")
        edges.append(Edge(source=hidden[-1], target=hidden[0].slot("in")))
        unroll = spec.plan.unroll
    state_init = None
    if spec.plan.state_init == "global_normal":
        state_init = GlobalStateInit(initializer=NormalInitializer(mean=0.0, std=spec.plan.state_init_std))
    try:
        return graph(nodes=nodes, edges=edges, task_map=TaskMap(x=inp, y=out), inference=make_inference(spec.plan), unroll=unroll,
                     graph_state_initializer=state_init)
    except GraphCycleError as exc:  # pragma: no cover - guarded by plan validation
        raise ContractViolation(str(exc)) from exc


def structure_digest(structure, spec: WorkspaceGraphSpec) -> dict[str, Any]:
    inf = structure.config["inference"]
    return {
        "nodes": [(n, type(node).__name__, list(node.node_info.shape), type(node.node_info.activation).__name__, type(node.node_info.energy).__name__)
                  for n, node in structure.nodes.items()],
        "edges": sorted(structure.edges), "schedule": list(structure.schedule), "unroll": structure.config.get("unroll"),
        "inference": {"class": type(inf).__name__, "config": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in inf.config.items()}},
        "state_initializer": type(structure.config["graph_state_initializer"]).__name__, "fabricpc": FABRICPC_VERSION,
    }


def init_params(structure, seed: int) -> GraphParams:
    return initialize_params(structure, jax.random.PRNGKey(int(seed)))


# ------------------------------------------------------------------ params <-> restricted arrays
def params_to_arrays(params: GraphParams) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for node, np_ in params.nodes.items():
        for k, v in np_.weights.items():
            out[f"{node}|weights|{k}"] = np.asarray(v, dtype=np.float64)
        for k, v in np_.biases.items():
            out[f"{node}|biases|{k}"] = np.asarray(v, dtype=np.float64)
    return out


def arrays_to_params(structure, arrays: dict[str, np.ndarray]) -> GraphParams:
    nodes: dict[str, dict[str, dict[str, Any]]] = {n: {"weights": {}, "biases": {}} for n in structure.nodes}
    for key, arr in arrays.items():
        node, kind, name = key.split("|", 2)
        if node not in nodes or kind not in ("weights", "biases"):
            raise ContractViolation(f"unknown FabricPC parameter key {key!r} for this structure")
        nodes[node][kind][name] = jnp.asarray(arr)
    return GraphParams(nodes={n: NodeParams(weights=d["weights"], biases=d["biases"]) for n, d in nodes.items()})


def params_from_state_arrays(structure, arrays: dict[str, Any]) -> GraphParams:
    return arrays_to_params(structure, arrays)


# ------------------------------------------------------------------ prediction / learning primitives
def _settle(structure, params, clamps, batch_size: int, settle: bool, key):
    state = initialize_graph_state(structure, batch_size, key, clamps=clamps, params=params)
    e0 = graph_energy(state, structure)
    if settle:
        state = run_inference(params, state, clamps, structure)
    return state, e0


def free_prediction(structure, params, x, settle: bool, key=None):
    """Target-free: clamp the input only; the readout is free. Returns (readout z_mu, diag)."""
    x = jnp.asarray(x)
    clamps = build_clamps({"x": x}, structure, clamp_target=False)
    state, e0 = _settle(structure, params, clamps, int(x.shape[0]), settle, key if key is not None else jax.random.PRNGKey(0))
    return state.nodes[READOUT].z_mu, {"energy_initial": e0, "energy_final": graph_energy(state, structure)}


def clamped_local_gradients(structure, params, x, target, settle: bool, key=None):
    """PC learning: clamp input AND target, settle, local weight gradients (means per prediction)."""
    x, target = jnp.asarray(x), jnp.asarray(target)
    clamps = build_clamps({"x": x, "y": target}, structure, clamp_target=True)
    state, e0 = _settle(structure, params, clamps, int(x.shape[0]), settle, key if key is not None else jax.random.PRNGKey(0))
    denom = grad_denominator(structure, clamps)
    grads = pc_weight_gradients(params, state, structure, clamps)
    energy = graph_energy(state, structure) / denom
    return grads, {"objective": energy, "energy_initial": e0 / denom, "denominator": denom}


def bp_through_inference(structure, params, x, loss_fn: Callable[[Any], Any], settle: bool, key=None):
    """Reverse-mode gradient of ``loss_fn(prediction)`` through the target-free inference."""
    x = jnp.asarray(x)

    def objective(p):
        pred, diag = free_prediction(structure, p, x, settle, key)
        return loss_fn(pred), diag

    (loss, diag), grads = jax.value_and_grad(objective, has_aux=True)(params)
    return loss, grads, diag


def grads_to_arrays(grads: GraphParams) -> dict[str, np.ndarray]:
    return params_to_arrays(grads)


def mesh_for(devices):
    return jax.make_mesh((len(devices),), ("data",), devices=list(devices))


def shard_batch(mesh, arrays: dict[str, Any]) -> dict[str, Any]:
    sharding = NamedSharding(mesh, P("data"))
    return {k: jax.device_put(jnp.asarray(v), sharding) for k, v in arrays.items()}


def replicate(mesh, tree):
    return jax.device_put(tree, NamedSharding(mesh, P()))


def jit(fn):
    return jax.jit(fn)


def energy_of(structure, state) -> float:
    return float(graph_energy(state, structure))


def device_of(array) -> list[int]:
    try:
        return sorted(int(d.id) for d in array.sharding.device_set)
    except Exception:  # pragma: no cover
        return []
