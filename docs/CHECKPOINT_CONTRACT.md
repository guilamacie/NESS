# Checkpoint contract

## Identity

`PredictorManifest` (`ness.manifest/1`) names: protocol and arm, `composition_hash`, one
`ComponentRef` per node (plugin id/version/kind, config hash, **state hash**, state schema id,
runtime) plus `composition:boundaries` (edge parameters), task and scenario refs, memory
snapshot id, inference and learning profiles, credit-map hash, RNG state hash, dependency lock
and `n_updates`. `manifest_id` is the content hash of all of that; every `PredictionResult`
carries it.

## Snapshot

`SystemSnapshot` = manifest + per-component `StateSnapshot` (arrays + JSON) + edge parameters +
optimizer state (learner checkpoints only) + RNG state + memory snapshot export + the arm/experiment
specification (insertion order preserved: some mappings are order-significant).

## Storage and publication

`CheckpointStore(root)`: `stage(snapshot)` writes content-addressed `.npy` blobs
(`allow_pickle=False`) and an index JSON to `staging/`, validates that every referenced blob
exists; `publish(staged, arm_id, expected_parent)` moves the index to `manifests/` and updates
`HEAD.<arm>` by compare-and-swap (conflict -> `CheckpointError`). Blob reads verify checksums.

## Restore

`NessSystem.from_snapshot(snapshot, registry)`: re-parse the stored spec, **recompile** the
composition, refuse if `composition_hash` differs, instantiate plugins **by manifest** (never
from code in the checkpoint), restore each component with its plugin's `restore_state`
(fail-closed on plugin id, major version, state schema id and config hash; `migrate_state` is the
explicit escape), restore edge parameters, optimizer, RNG and memory, then refuse if the
reconstructed `manifest_id` differs. Tests: `tests/test_checkpoint.py`.

## Backend state (FabricPC)

FabricPC workspace parameters are stored as restricted arrays keyed `<node>|weights|<edge>` /
`<node>|biases|<name>` with JSON metadata (graph digest hash, FabricPC version and family,
adapter version, algorithm-spec hash). Restore rebuilds the FabricPC structure from NESS
configuration, checks digest and family, and loads the arrays through the version-routed
adapter. No FabricPC runtime object is pickled; FabricPC's own checkpoint tooling is not used.
Manifests record backend, jax/jaxlib/fabricpc versions and adapter version, so a checkpoint
produced under a different FabricPC family fails closed (explicit migration required).

## Plugin obligations

`snapshot_state` must contain everything that affects predictions (params, buffers such as
running statistics or fixed readouts, meta such as weight digests). Version bumps that change
state layout increment the major version and either ship a `migrate_state` or fail closed.

## Learner state added in 0.3 (ADR-0012, ADR-0013)

The learner snapshot's optimizer blob also carries the schedule and `param_groups` rules (data)
and, for `parameter_anchor` objectives, the reference parameters (arrays keyed
`a|<objective>|<group>|<name>`, listed in `data.anchors`). Restoring refuses a checkpoint whose
optimizer configuration differs from the arm's; a system restored from a serving checkpoint
has no anchor reference and refuses to learn until one is set.
