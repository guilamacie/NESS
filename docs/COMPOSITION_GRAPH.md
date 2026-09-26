# Composition graph (`ness.composition/1`)

The composition graph is the collaborator-facing description of *which scientifically
meaningful macro modules read which earlier ports*. It is not a tensor IR.

## Elements

* **Node**: `{id, plugin, config, inputs, memory_queries, enabled}`. Disabled nodes do not
  instantiate, import or run; a reference to a disabled node is a compile error.
* **Input wiring**: `port: selector` | `{from, boundary: [...]}` |
  `{merge, sources: [...], merge_params, post: [...]}`. More than one source requires an explicit
  `merge`.
* **Source selector**: `scheme://node/port`; scheme must match the producer kind.
* **Boundary transforms** (per source, then `post` after the merge): `identity`, `linear{out_dim,
  init}` (learned), `layer_norm{affine,eps}`, `rms_norm{affine,eps}` (affine = learned),
  `standardize{mean,scale}` (declared constants), `mean_pool{axis}`, `last_step{axis}`, `flatten`.
* **Merges**: `concat` (last axis), `add`, `gated_add{per_dim,gate_init}` (learned sigmoid gate on
  every source after the first), `weighted_sum` (learned softmax weights), `select{index}`,
  `attention_pool` (learned query over a `[T, d]` source), `cross_attention{model_dim}` (query
  `[d_q]`, memory `[T, d_kv]`).
* **Output**: `{task, from}`; the port kind must match the task's forecast type and the
  coordinate schema must equal the task target.
* **Recurrent regions**: `recurrent_regions: [{id, nodes, state_variables, energy, solver,
  iterations, derivative}]`. Cycles are legal only inside a region. No solver is verified in this
  release, so executing a graph with a region raises `UnsupportedCapability`.

## Validation performed by `compile_graph`

1. plugins instantiated (only those used); `describe()` collected;
2. topological order (Kahn) with regions collapsed; cycle -> `CompositionError`;
3. per edge: producer port exists, scheme/kind agree, destination accepts the semantic type,
   **availability role permitted by the task policy** (`AccessPolicyViolation` otherwise),
   shape rules for each transform and merge, coordinate compatibility for add-like merges,
   learned ops only into a differentiable destination, declared destination shape compatible;
4. differentiable region = nodes in the single differentiable runtime; an edge is
   differentiable only if destination and producer are in that runtime and the producer port is
   `differentiable`; all other edges are recorded as stop edges;
5. outputs validated against task spaces;
6. `composition_hash` = hash(spec, descriptors, edge-parameter ids, observation schema);
   `resolved_wiring()` gives node -> port -> [selectors] for manifests and diagnostics.

## Semantics guaranteed at execution

* pass-through inputs keep their typed payload (`PointForecast`, `HypothesisEvidence`,
  `FeatureEvidence` with knownness);
* any transform/merge produces a dense derived value recorded as evidence and as
  `port_values[(node, "port#assembled")]`; knownness survives only a plain `concat`;
* provenance of every output lists dependencies, roles, availability and the resolved sources;
* the same `assemble_dense` runs in the eager and differentiable paths.

## Enabling/disabling a residual, swapping a provider, rerouting a semantic module

All are edits to `inputs`/`plugin` fields only. Examples are in
`examples/vertical_slice_timeseries/configs/vertical_slice.yaml` and `tests/test_vertical_slice.py`.
