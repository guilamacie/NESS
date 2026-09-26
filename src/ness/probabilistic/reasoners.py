"""Reference probabilistic reasoners.

* ``ExactFiniteRegimeReasoner`` - exact enumeration over a finite latent (oracle baseline).
* ``ImportanceSamplingRegimeReasoner`` - same model, self-normalised importance sampling
  from the prior; approximate, seeded, with effective-sample-size diagnostics.

Both implement the ``ProbabilisticReasoner`` contract (describe/compile/infer) *and*
the ``MacroModule`` execution contract so they can be wired into a composition graph.
Outputs lower to ``HypothesisEvidence`` (+ ``FeatureEvidence`` posterior moments and an
``ExecutionTrace``). Neither ever sees withheld outcomes: inputs are graph ports only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..contracts import (
    ApproximationStatus,
    ContractViolation,
    Differentiability,
    ExecutionTrace,
    FeatureEvidence,
    FieldRole,
    HypothesisEvidence,
    ModuleDescriptor,
    ModuleState,
    PortSchema,
    PortSpec,
    ProbabilisticReasonerCapabilities,
    ProducerRef,
    Provenance,
)
from ..plugin_api.base import BaseModule
from ..plugin_api.module import ModuleOutputs, PortValue, RuntimeContext
from .model import FiniteLatentModel


@dataclass(frozen=True, slots=True)
class ReasonerResult:
    posterior: np.ndarray
    log_evidence: float | None
    diagnostics: dict[str, Any]
    approximation: ApproximationStatus


@runtime_checkable
class ProbabilisticReasoner(Protocol):
    def describe_reasoner(self) -> ProbabilisticReasonerCapabilities: ...
    def compile(self, model_spec: dict[str, Any], input_schema: dict[str, int], output_schema: tuple[str, ...]) -> FiniteLatentModel: ...
    def infer(self, compiled: FiniteLatentModel, features: np.ndarray, known: np.ndarray, rng: np.random.Generator, budget: int) -> ReasonerResult: ...


class _FiniteLatentReasonerBase(BaseModule):
    module_kind = "reasoner"
    runtime = "numpy"
    state_schema_id = "ness.reasoner.finite_latent/1"
    inference_method = "abstract"
    approximation = "exact"

    def validate_config(self) -> None:
        self.model = FiniteLatentModel.from_config(self.config.get("model", {}))
        ports = self.config.get("feature_ports")
        if not isinstance(ports, dict) or not ports:
            raise ContractViolation(f"{self.plugin_id}: config.feature_ports must map input port name -> feature dim")
        self.feature_ports: dict[str, int] = {k: int(v) for k, v in ports.items()}
        total = sum(self.feature_ports.values())
        if total != self.model.feature_dim:
            raise ContractViolation(
                f"{self.plugin_id}: feature ports total dim {total} != emission feature dim {self.model.feature_dim}")
        self.sample_budget = int(self.config.get("samples", 4000))

    # --- ProbabilisticReasoner ------------------------------------------------
    def describe_reasoner(self) -> ProbabilisticReasonerCapabilities:
        return ProbabilisticReasonerCapabilities(
            runtime="numpy", inference_methods=(self.inference_method,), exact_enumeration=self.approximation == "exact",
            supports_discrete_latents=True, supports_continuous_latents=False, supports_amortized_guide=False,
            supports_sampling=True, supports_log_prob=True, supports_differentiable_path=False, gradient_boundary="stop",
            supports_online_state=False, approximation=self.approximation,
            latent_schema=f"categorical[{self.model.n_states}]", output_schema="hypothesis+feature+trace",
            serialization_version="1")

    def compile(self, model_spec: dict[str, Any], input_schema: dict[str, int], output_schema: tuple[str, ...]) -> FiniteLatentModel:
        model = FiniteLatentModel.from_config(model_spec)
        if sum(input_schema.values()) != model.feature_dim:
            raise ContractViolation("input schema does not match emission dimension")
        return model

    def infer(self, compiled: FiniteLatentModel, features: np.ndarray, known: np.ndarray, rng: np.random.Generator, budget: int) -> ReasonerResult:  # pragma: no cover
        raise NotImplementedError

    # --- MacroModule ----------------------------------------------------------
    def describe(self) -> ModuleDescriptor:
        inputs = tuple(
            PortSpec(name, PortSchema("dense", (dim,)), "features", FieldRole.DERIVED, Differentiability.STOP,
                     accepts_semantic_types=("*",), description=f"feature block of dim {dim}")
            for name, dim in self.feature_ports.items()
        )
        outputs = (
            PortSpec("posterior", PortSchema("hypothesis", (self.model.n_states,)), "regime_posterior", FieldRole.DERIVED, Differentiability.STOP),
            PortSpec("moments", PortSchema("feature", (self.model.n_states,)), "posterior_moments", FieldRole.DERIVED, Differentiability.STOP),
        )
        return self._descriptor(input_ports=inputs, output_ports=outputs, capabilities=self.describe_reasoner())

    def initialize(self, rng: np.random.Generator) -> ModuleState:
        return ModuleState(meta={"model_hash": self.model.model_hash})

    def forward(self, inputs: dict[str, PortValue], state: ModuleState, ctx: RuntimeContext) -> ModuleOutputs:
        blocks, knowns = [], []
        for name, dim in self.feature_ports.items():
            pv = inputs[name]
            x = np.asarray(pv.dense(), dtype=np.float64).reshape(-1)
            if x.shape != (dim,):
                raise ContractViolation(f"{self.plugin_id}: port {name} delivered {x.shape}, declared ({dim},)")
            k = pv.knownness()
            knowns.append(np.ones(dim, dtype=bool) if k is None else np.asarray(k, dtype=bool).reshape(-1))
            blocks.append(x)
        features = np.concatenate(blocks)
        known = np.concatenate(knowns)
        res = self.infer(self.model, features, known, ctx.rng, self.sample_budget)
        producer = ProducerRef(ctx.node_id, self.plugin_id, self.plugin_version)
        deps = tuple(pv.evidence_id for pv in inputs.values())
        base_prov = Provenance(producer, deps, frozenset({FieldRole.DERIVED}), None,
                               tuple((n, pv.evidence_id) for n, pv in inputs.items()), f"{self.inference_method}")
        semantics = "exact_posterior" if res.approximation == ApproximationStatus.EXACT else "approximate_posterior"
        posterior = HypothesisEvidence(
            evidence_id=f"{ctx.node_id}/posterior@{ctx.request.request_id}", provenance=base_prov, semantic_type="regime_posterior",
            approximation=res.approximation, alternatives=self.model.states, weights=res.posterior,
            weight_semantics=semantics, omitted_mass=0.0 if res.approximation == ApproximationStatus.EXACT else None)
        moments = FeatureEvidence(
            evidence_id=f"{ctx.node_id}/moments@{ctx.request.request_id}", provenance=base_prov, semantic_type="posterior_moments",
            approximation=res.approximation, value=res.posterior, interpretation="posterior probability per state")
        trace = ExecutionTrace(
            evidence_id=f"{ctx.node_id}/trace@{ctx.request.request_id}", provenance=base_prov, semantic_type="reasoner_trace",
            approximation=res.approximation, status="ok",
            details={"inference_method": self.inference_method, "log_evidence": res.log_evidence, **res.diagnostics,
                     "known_features": int(known.sum()), "feature_dim": int(known.size)})
        return ModuleOutputs(ports={"posterior": posterior, "moments": moments}, evidence=(trace,),
                             diagnostics={"inference_method": self.inference_method, **res.diagnostics}, cost=float(res.diagnostics.get("cost", 1.0)))


class ExactFiniteRegimeReasoner(_FiniteLatentReasonerBase):
    plugin_id = "exact_finite_regime_reasoner"
    plugin_version = "1.0.0"
    inference_method = "exact_enum"
    approximation = "exact"

    def infer(self, compiled: FiniteLatentModel, features: np.ndarray, known: np.ndarray, rng: np.random.Generator, budget: int) -> ReasonerResult:
        log_joint = np.log(compiled.prior) + compiled.log_likelihood(features, known)
        m = log_joint.max()
        log_z = m + np.log(np.exp(log_joint - m).sum())
        post = np.exp(log_joint - log_z)
        return ReasonerResult(post / post.sum(), float(log_z), {"cost": float(compiled.n_states)}, ApproximationStatus.EXACT)


class ImportanceSamplingRegimeReasoner(_FiniteLatentReasonerBase):
    plugin_id = "importance_sampling_regime_reasoner"
    plugin_version = "1.0.0"
    inference_method = "importance_sampling"
    approximation = "monte_carlo"

    def infer(self, compiled: FiniteLatentModel, features: np.ndarray, known: np.ndarray, rng: np.random.Generator, budget: int) -> ReasonerResult:
        n = max(int(budget), 1)
        # Deterministic given the request-local RNG stream handed in by the runtime.
        z = rng.choice(compiled.n_states, size=n, p=compiled.prior)
        ll = compiled.log_likelihood(features, known)
        logw = ll[z]
        w = np.exp(logw - logw.max())
        w /= w.sum()
        post = np.array([w[z == k].sum() for k in range(compiled.n_states)])
        post = np.clip(post, 1e-12, None)
        post /= post.sum()
        ess = float(1.0 / np.sum(w**2))
        return ReasonerResult(post, None, {"samples": n, "effective_sample_size": ess, "cost": float(n)}, ApproximationStatus.APPROXIMATE)
