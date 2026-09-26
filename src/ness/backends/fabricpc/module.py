"""``fabricpc_residual_cap``: a NESS cap whose consumer is a FabricPC predictive-coding graph.

Prediction is target-free: only the packed evidence is clamped; the readout is free and the
correction is its ``z_mu`` after the declared inference (none / sPC / ePC / cyclic sPC).
The forecast is ``baseline + correction`` (residual point writer; zero-initialised readout
=> exact baseline parity at initialisation). Learning is a separate rule plugin."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import (
    ContractViolation,
    Differentiability,
    FieldRole,
    ModuleDescriptor,
    ModuleState,
    ParameterGroupSpec,
    PointForecast,
    PortSchema,
    PortSpec,
    StateSnapshot,
    content_hash,
    horizon_schema,
)
from ...plugin_api.base import BaseModule
from ...plugin_api.module import ModuleOutputs, PortValue, RuntimeContext
from ...runtimes.bootstrap import ensure_bootstrapped
from . import ADAPTER_VERSION
from .checkpoint import STATE_SCHEMA, check_restorable
from .translate import AlgorithmSpec, plan_inference, prediction_spec
from .version import load_adapter


class FabricPCResidualCap(BaseModule):
    plugin_id = "fabricpc_residual_cap"
    plugin_version = "1.0.0"
    module_kind = "cap"
    runtime = "fabricpc"
    requires = ("fabricpc", "jax")
    state_schema_id = STATE_SCHEMA

    def validate_config(self) -> None:
        c = self.config
        self.channels = tuple(c.get("channels", ("y0", "y1")))
        h = c.get("horizons", 4)
        self.horizons = tuple(range(1, int(h) + 1)) if isinstance(h, int) else tuple(int(x) for x in h)
        feats = c.get("features", {})
        if not feats:
            raise ContractViolation(f"{self.plugin_id}: config.features required (input port -> dim)")
        self.features = {k: int(feats[k]) for k in sorted(feats)}
        self.hidden = tuple(int(x) for x in c.get("hidden", (16,)))
        self.activation = c.get("activation", "tanh")
        if self.activation not in ("tanh", "identity"):
            raise ContractViolation("activation must be tanh|identity")
        self.plan = plan_inference(dict(c.get("inference", {})))
        self.seed = int(c.get("seed", 0))
        self.in_dim = sum(self.features.values())
        self.out_dim = len(self.horizons) * len(self.channels)
        self._adapter = None
        self._structure = None
        self._digest: dict[str, Any] | None = None

    # ------------------------------------------------------------------ adapter access (lazy, post-bootstrap)
    def adapter(self):
        if self._adapter is None:
            ensure_bootstrapped("fabricpc")
            self._adapter, self._version_info = load_adapter()
        return self._adapter

    def structure(self):
        if self._structure is None:
            ad = self.adapter()
            spec = ad.WorkspaceGraphSpec(self.in_dim, self.hidden, self.out_dim, self.activation, self.plan, self.seed)
            self._structure = ad.build_structure(spec)
            self._digest = ad.structure_digest(self._structure, spec)
        return self._structure

    def digest(self) -> dict[str, Any]:
        self.structure()
        assert self._digest is not None
        return self._digest

    def inference_profile(self) -> str:
        return self.plan.profile

    def settles(self) -> bool:
        return self.plan.solver != "none"

    # ------------------------------------------------------------------ descriptor
    def describe(self) -> ModuleDescriptor:
        H, D = len(self.horizons), len(self.channels)
        target = horizon_schema(self.channels, self.horizons)
        return self._descriptor(
            input_ports=(PortSpec("baseline", PortSchema("point_forecast", (H, D)), "point_forecast", FieldRole.DERIVED, Differentiability.STOP, target.schema_id,
                                  accepts_semantic_types=("point_forecast",)),)
            + tuple(PortSpec(n, PortSchema("dense", (d,)), "packed_evidence", FieldRole.DERIVED, Differentiability.STOP, accepts_semantic_types=("*",)) for n, d in self.features.items()),
            output_ports=(PortSpec("forecast", PortSchema("point_forecast", (H, D)), "point_forecast", FieldRole.DERIVED, Differentiability.STOP, target.schema_id),
                          PortSpec("correction", PortSchema("dense", (H * D,)), "point_correction", FieldRole.DERIVED, Differentiability.STOP)),
            parameter_groups=(ParameterGroupSpec("workspace", "trainable", "fabricpc", False, "FabricPC graph weights/biases (Linear nodes); readout zero-initialised"),),
            capabilities={"backend": "fabricpc", "adapter_version": ADAPTER_VERSION, "inference": self.plan.canonical(), "architecture": f"dense{list(self.hidden)}",
                          "activation": self.activation, "writer": "point_residual", "baseline_preserving_at_zero": True, "gradient_boundary": "stop (no cross-runtime derivative)",
                          "masked_objective": "unsupported for fabricpc_pc_local (all-true masks only); supported for fabricpc_bp_through_inference",
                          "parity_twin": "jax_dense_workspace_cap"},
        )

    def algorithm_spec(self) -> dict[str, Any]:
        info = getattr(self, "_version_info", None)
        backend = {"fabricpc": info.installed_version if info else "unbootstrapped", "adapter": ADAPTER_VERSION, "status": info.status if info else "unknown"}
        return prediction_spec(self.digest(), self.plan, backend).canonical()

    # ------------------------------------------------------------------ state
    def initialize(self, rng: np.random.Generator) -> ModuleState:
        ad = self.adapter()
        params = ad.init_params(self.structure(), self.seed)
        arrays = ad.params_to_arrays(params)
        return ModuleState(params={"workspace": arrays},
                           meta={"structure_digest_hash": content_hash(self.digest()), "fabricpc_version": ad.FABRICPC_VERSION,
                                 "fabricpc_family": ".".join(ad.FABRICPC_VERSION.split(".")[:2]), "adapter_version": ADAPTER_VERSION,
                                 "algorithm_spec_hash": content_hash(self.algorithm_spec())})

    def restore_state(self, snapshot: StateSnapshot) -> ModuleState:
        state = super().restore_state(snapshot)
        ad = self.adapter()
        check_restorable(state.meta, content_hash(self.digest()), ".".join(ad.FABRICPC_VERSION.split(".")[:2]))
        ad.arrays_to_params(self.structure(), state.params["workspace"])  # validates keys against the rebuilt structure
        return state

    def graph_params(self, state: ModuleState):
        return self.adapter().arrays_to_params(self.structure(), state.params["workspace"])

    # ------------------------------------------------------------------ packing (shared with the rules)
    def pack_features(self, dense_inputs: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([np.asarray(dense_inputs[n], dtype=np.float64).reshape(-1) for n in self.features])

    def pack_from_record(self, record: Any, compiled_node: Any) -> tuple[np.ndarray, np.ndarray]:
        """(features [F], baseline [H, D]) for one recorded prediction."""
        dense: dict[str, np.ndarray] = {}
        baseline = None
        for port, ri in compiled_node.inputs.items():
            if ri.is_passthrough():
                s = ri.sources[0]
                pv = record.port_values[(s.producer_node, s.producer_port.name)]
            else:
                pv = record.port_values[(compiled_node.spec.node_id, f"{port}#assembled")]
            if port == "baseline":
                baseline = np.asarray(pv.dense(), dtype=np.float64)
            else:
                dense[port] = np.asarray(pv.dense(), dtype=np.float64)
        assert baseline is not None
        return self.pack_features(dense), baseline

    # ------------------------------------------------------------------ forward (target-free)
    def forward(self, inputs: dict[str, PortValue], state: ModuleState, ctx: RuntimeContext) -> ModuleOutputs:
        base = inputs["baseline"].payload
        if not isinstance(base, PointForecast):
            raise ContractViolation(f"{self.plugin_id}: baseline must be a PointForecast")
        ad = self.adapter()
        x = self.pack_features({n: inputs[n].dense() for n in self.features})[None, :]
        params = self.graph_params(state)
        key = None
        if self.plan.state_init != "feedforward":  # request-local, deterministic latent initialisation
            import jax
            key = jax.random.PRNGKey(int(ctx.rng.integers(0, 2**31 - 1)))
        corr, diag = ad.free_prediction(self.structure(), params, x, self.settles(), key)
        corr = np.asarray(corr, dtype=np.float64).reshape(base.values.shape)
        fc = PointForecast(base.values + corr, base.target, base.functional)
        return ModuleOutputs(ports={"forecast": fc, "correction": corr.reshape(-1)},
                             diagnostics={"correction_norm": float(np.linalg.norm(corr)), "energy_initial": float(diag["energy_initial"]),
                                          "energy_final": float(diag["energy_final"]), "inference": self.plan.canonical(), "realized_devices": ad.device_of(corr) if hasattr(corr, "sharding") else []},
                             cost=float(self.in_dim * (self.plan.infer_steps + 1)))
