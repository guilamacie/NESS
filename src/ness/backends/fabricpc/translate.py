"""NESS ``AlgorithmSpec`` for FabricPC executions.

NESS owns the scientific algorithm identity as a *tuple* (model, state, initialization,
energy, clamps, solver, derivative, readout, learning). The FabricPC adapter translates a
complete spec into one concrete execution only when the semantics match, and refuses
otherwise. A FabricPC string such as ``algorithm="pc"`` is never taken as the identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...contracts import UnsupportedCapability, ValidationError, content_hash

SOLVERS = {
    "fabricpc_feedforward": {"solver": "none", "fabricpc_class": None},
    "fabricpc_spc": {"solver": "state_based_sgd", "fabricpc_class": "InferenceSGD"},
    "fabricpc_epc": {"solver": "error_parameterised", "fabricpc_class": "EPCInference"},
    "fabricpc_spc_recurrent": {"solver": "state_based_sgd", "fabricpc_class": "InferenceSGD", "cyclic": True},
}


STATE_INITS = ("feedforward", "global_normal")


@dataclass(frozen=True, slots=True)
class InferencePlan:
    profile: str
    solver: str
    fabricpc_class: str | None
    eta_infer: float
    infer_steps: int
    latent_decay: float
    unroll: int | None
    cyclic: bool
    state_init: str = "feedforward"   # feedforward: z_latent = z_mu at init (free settle on a DAG is then a no-op)
    state_init_std: float = 0.05      # global_normal: latents ~ N(0, std); settling then genuinely relaxes

    def canonical(self) -> dict:
        return {"profile": self.profile, "solver": self.solver, "fabricpc_class": self.fabricpc_class, "eta_infer": self.eta_infer,
                "infer_steps": self.infer_steps, "latent_decay": self.latent_decay, "unroll": self.unroll, "cyclic": self.cyclic,
                "state_init": self.state_init, "state_init_std": self.state_init_std}


def plan_inference(cfg: dict[str, Any]) -> InferencePlan:
    profile = cfg.get("profile", "fabricpc_feedforward")
    if profile not in SOLVERS:
        raise ValidationError(f"fabricpc inference.profile must be one of {sorted(SOLVERS)}, got {profile!r}")
    s = SOLVERS[profile]
    steps = int(cfg.get("infer_steps", 0 if s["solver"] == "none" else 20))
    if s["solver"] == "none" and steps != 0:
        raise ValidationError("fabricpc_feedforward has no settling; infer_steps must be 0")
    if s["solver"] != "none" and steps < 1:
        raise ValidationError(f"{profile} requires infer_steps >= 1")
    cyclic = bool(s.get("cyclic", False))
    unroll = cfg.get("unroll")
    if cyclic and (unroll is None or int(unroll) < 1):
        raise ValidationError("fabricpc_spc_recurrent requires inference.unroll >= 1 (FabricPC cyclic graphs need an explicit unroll)")
    state_init = cfg.get("state_init", "feedforward")
    if state_init not in STATE_INITS:
        raise ValidationError(f"inference.state_init must be one of {STATE_INITS}")
    if state_init != "feedforward" and s["solver"] == "none":
        raise ValidationError("fabricpc_feedforward requires state_init: feedforward (there is no settling to relax a random init)")
    return InferencePlan(profile, s["solver"], s["fabricpc_class"], float(cfg.get("eta_infer", 0.05)), steps, float(cfg.get("latent_decay", 0.0)),
                         int(unroll) if unroll is not None else None, cyclic, state_init, float(cfg.get("state_init_std", 0.05)))


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """Complete identity of one FabricPC-backed computation (PDF §11.1 tuple)."""

    model: dict[str, Any]           # graph digest: nodes, shapes, activations, energies, edges
    state: tuple[str, ...]          # temporary inference variables
    initialization: str             # feedforward
    energy: str                     # per-node gaussian energies; readout gaussian
    clamps_prediction: tuple[str, ...]
    clamps_learning: tuple[str, ...]
    inference: dict[str, Any]
    derivative: str                 # how parameter gradients are obtained
    readout: str                    # what is read as the prediction
    learning_rule: str
    loss: str
    reductions: str
    parameter_masks: str
    backend: dict[str, Any] = field(default_factory=dict)

    def canonical(self) -> dict:
        def j(v):
            return list(v) if isinstance(v, tuple) else v
        return {k: j(getattr(self, k)) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]

    @property
    def spec_hash(self) -> str:
        return content_hash(self.canonical())


def prediction_spec(model_digest: dict[str, Any], plan: InferencePlan, backend: dict[str, Any]) -> AlgorithmSpec:
    return AlgorithmSpec(
        model=model_digest, state=("z_latent[hidden*]", "z_latent[readout]") if plan.solver != "none" else (),
        initialization=("FeedforwardStateInit (z_latent = z_mu at every unclamped node; a target-free settle on a DAG starts at equilibrium)"
                        if plan.state_init == "feedforward" else f"GlobalStateInit(Normal(0,{plan.state_init_std})) with request-local RNG; settling relaxes from noise"),
        energy="GaussianEnergy per node; readout GaussianEnergy(precision=1)",
        clamps_prediction=("input",), clamps_learning=(),
        inference=plan.canonical(), derivative="none (target-free prediction)", readout="z_mu[readout] after the last inference step",
        learning_rule="none", loss="none", reductions="none", parameter_masks="readout zero-initialised; all workspace weights update-eligible", backend=backend)


def learning_spec(base: AlgorithmSpec, rule: str) -> AlgorithmSpec:
    if rule == "fabricpc_pc_local":
        return AlgorithmSpec(base.model, base.state, base.initialization, base.energy, base.clamps_prediction, ("input", "readout=target_correction"),
                             base.inference, "FabricPC local weight gradients of the total clamped energy (pc_weight_gradients), batch-summed then / prediction count",
                             base.readout, "fabricpc_pc_local (PDF workspace_pc_local / workspace_epc_local depending on solver)",
                             "clamped total energy per prediction (not the task loss)", "sum over batch, divide once by N=B", base.parameter_masks, base.backend)
    if rule == "fabricpc_bp_through_inference":
        return AlgorithmSpec(base.model, base.state, base.initialization, base.energy, base.clamps_prediction, ("input",),
                             base.inference, "jax reverse-mode gradient of the NESS task loss through initialize_graph_state + run_inference (finite unroll, target free)",
                             base.readout, "fabricpc_bp_through_inference (PDF ff_bp when no settling, workspace_bp_unroll otherwise)",
                             "NESS task loss", "sum numerators / sum denominators over the batch", base.parameter_masks, base.backend)
    raise UnsupportedCapability(f"no FabricPC translation for learning rule {rule!r}")
