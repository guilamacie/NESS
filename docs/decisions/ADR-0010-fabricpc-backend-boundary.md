# ADR-0010: FabricPC integration boundary

**Status** accepted (2026-09-26). **Context** The first cycle deferred FabricPC (only an
unrelated historical fork was found locally). The official `trueagi-io/FabricPC` 0.6.0 is now the
first compatibility target. FabricPC's public API is a functional graph/inference toolkit
(`graph`, node classes, `InferenceSGD`/`EPCInference`, `run_inference`, `initialize_graph_state`,
`pc_weight_gradients`, `train`/`evaluate`, mesh data parallelism) with a `setup_jax()` that must
run before JAX backend initialisation.

**Decision**
1. FabricPC is a *numerical backend* behind `ness.backends.fabricpc`; observations, access
   policy, evidence, programs, memory, tasks, protocols, manifests and provenance stay in NESS.
2. Only `compat/v<family>.py` imports FabricPC classes; version routing (`version.py`) selects
   the module; unknown families fail closed. FabricPC types never appear in contracts, plugin
   protocols, checkpoint schemas or composition interfaces.
3. Integration point = a NESS **cap** (`fabricpc_residual_cap`) whose consumer is a FabricPC
   graph, plus **learning rules** (`fabricpc_pc_local`, `fabricpc_bp_through_inference`). The
   deployed prediction is target-free (input clamped, readout free); the PC training clamp is
   confined to the learning rule.
4. NESS controls the training loop using FabricPC's public lower-level functions; FabricPC's
   `train(algorithm=...)` labels are not used as algorithm identity. A complete `AlgorithmSpec`
   is recorded per node and per rule.
5. Checkpoints store FabricPC parameters as restricted arrays plus JSON metadata; restore
   rebuilds structures from NESS config and fails closed on digest/family mismatch.
6. Multi-device: data parallelism via JAX `NamedSharding` over a `("data",)` mesh applied to
   FabricPC's own public functions (the same technique FabricPC's `make_train_step(mesh=...)`
   uses); model parallelism and multi-host are unsupported and separately represented.

**Alternatives rejected** using FabricPC's trainer end-to-end (couples loss normalisation,
optimizer and algorithm labels; violates NESS reductions/credit ownership); vendoring FabricPC;
making FabricPC nodes compose freely with jax nodes through a gradient bridge (no verified
bridge exists).

**Consequences** A future FabricPC family needs a new compat module and matrix rows, not NESS
changes. Scientific fact surfaced and tested: a target-free settle on an acyclic workspace with
feedforward initialisation is the feedforward pass; only learning differs.

**Revisit** if FabricPC exposes a stable checkpoint format worth mirroring as a backend
artifact, a nudge/EP phase, or model parallelism.
