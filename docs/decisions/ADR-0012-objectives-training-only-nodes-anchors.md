# ADR-0012: Learning objectives, training-only nodes, auxiliary targets, parameter anchors

**Decision**
* An arm may declare `learning.objectives`: a list of `{id, kind, weight, applies_to}` with kinds
  `task` (a task loss on its output port), `port_target` (a registry loss between a
  differentiable source port and a constant target) and `parameter_anchor`
  (`0.5 * w * ||theta - theta_ref||^2` over owned groups, gradient `w * (theta - theta_ref)`).
  Without the key the v0.2 behaviour holds exactly (one implicit task objective per task weighted
  by the rule's `task_weights`). Declaring any `task` objective replaces the implicit set.
* Each objective is reduced as sum of numerators over sum of denominators over the items it
  applies to (non-matching items contribute zero to both), weighted, summed. A weight-0 objective
  is evaluated for logging but excluded from the differentiated sum, so adding it is bitwise
  neutral. Per update the runner logs value, weight, numerator, denominator and items per
  objective. Objectives are in the protocol hash (they are arm configuration) and in the
  manifest's `algorithm_specs.objectives` (present only when declared).
* `applies_to`: `group: {key: value | [values]}` over `request.group` (streams are
  `group.stream` by convention) and `has_outcome: [task ids]`.
* Targets are constants to every rule: a `port_target` target is either a port of a node outside
  the differentiable region (typically a **training-only node**) or a revealed **auxiliary
  target** `{auxiliary: <field>}`, and the source must be a differentiable port of the region
  (`GradientBoundaryError` otherwise).
* **Training-only nodes** (`training_only: true` on a node) are frozen; the compiler proves they
  can never reach a prediction (no task output reads one; only training-only nodes may read one);
  the executor never runs them for a prediction; the system runs them per batch item, on the
  prediction record's values, only when an active objective needs their output. They appear in
  the manifest (`training_only_nodes`, component list) and in `ness validate` / `ness graph`.
* **Auxiliary targets** are `outcome_only` observation fields the scenario declares and places
  in the raw bundle. The existing role machinery keeps them out of every permitted view and
  rejects wiring them into nodes; the causal check perturbs and drops them; `reveal()` copies the
  fields an objective needs into the learning item. An optional `mask: {auxiliary: <field>}`
  masks positions.
* Losses come from a registry: built-ins `mse`, `kl_last_axis` (KL(target || source) per
  position, spaces `logits | log_probs | probs`, optional temperature, no T^2 rescaling),
  `cross_entropy` (integer ids); installed packages add `LossSpec`s through the `ness.losses`
  entry point (clashes fail closed).
* Anchors populate `CreditMap.anchors` (`(group, "l2_to_reference:initial:<id>", weight)`),
  apply with every learning rule, snapshot `theta_ref` at build (`reference: initial`;
  `NessSystem.set_anchor_reference()` re-snapshots) and store it in learner checkpoints next to
  the optimizer state. Learning from a system restored without it (a serving checkpoint) fails
  closed.
* `task` / `port_target` objectives are evaluated by `bp_direct` in per-item mode; batched /
  data-parallel evaluation and FabricPC rules owning groups in the same arm are rejected
  (`UnsupportedCapability`) until verified.

**Why** The alternative for frozen targets, delivering teacher distributions as scenario
outcomes, costs O(positions x classes) float64 per request in the outcome stream (~0.4 MiB per
scored position at 50k classes) and moves model computation into the data provider. A
training-only node keeps the teacher a versioned, manifest-identified component evaluated only
when learning needs it, while the compiler guarantees it cannot influence a prediction.
Out-of-band optimizer steps would bypass NESS's optimizer state, checkpoints and manifests.

**Tests** `tests/test_objectives.py` (A1 hand-written JAX gradient to 1e-9 and weight-0 bitwise;
A2 compile rejections and identical predictions; A3 filters; A4 anchor gradient and checkpoint
round trip; A5 auxiliary targets; fail-closed cases; loss closed forms).
