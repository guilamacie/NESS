# Probabilistic reasoners

A reasoner is a `MacroModule` of kind `reasoner` that also implements the
`ProbabilisticReasoner` contract: `describe_reasoner() -> ProbabilisticReasonerCapabilities`,
`compile(model_spec, input_schema, output_schema)`, `infer(compiled, features, known, rng,
budget) -> ReasonerResult`.

Capabilities it must declare: `runtime`, `inference_methods` (`exact_enum`, `variational`,
`importance_sampling`, `smc`, `hmc`, `nuts`, `belief_propagation`, `custom`),
`exact_enumeration`, discrete/continuous latents, amortized guide, sampling, log_prob,
`supports_differentiable_path`, `gradient_boundary`, `supports_online_state`, `approximation`,
latent/output schemas and `serialization_version`.

Outputs lower to typed evidence: `HypothesisEvidence` (posterior over alternatives; weight
semantics `exact_posterior` | `approximate_posterior` | `normalized_beam` |
`deterministic_assignment` | `score`), optional `FeatureEvidence` moments,
`PredictiveEvidence` only for a real forecast in a declared space, and an `ExecutionTrace`
with method, samples/ESS/ELBO, convergence, truncation and cost. A normalised retained beam is
never labelled a calibrated posterior.

Inputs are declared feature ports (`feature_ports: {name: dim}`; **order is significant** - it
is the emission feature layout); the composition graph routes raw observations, semantic
features, program outputs, neural states or memory evidence into them with boundary transforms
as needed. Missing features arrive with `knownness=False` and the reference model marginalises
them (`missing_policy: marginalize`) or fails (`fail`).

Reference implementations: `exact_finite_regime_reasoner` (oracle-quality baseline) and
`importance_sampling_regime_reasoner` (same model; approximate; ESS diagnostics). The example
plugin adds `deterministic_threshold_reasoner`. `tests/test_probabilistic.py` shows the
substitution/agreement test (T37) and target isolation (T38). A Pyro/NumPyro backend follows the
same shape: compile the model once, run inference in `infer`, keep model/guide/optimizer/RNG
state in separate `ModuleState` groups, and advertise a differentiable path only after contract
tests exercise it through the public training route.
