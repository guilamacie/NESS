# FabricPC workspace arms

`configs/fabricpc_arms.yaml` runs, on the vertical-slice scenario:

| arm | backend | deployed inference | learning rule (PDF name) |
|---|---|---|---|
| `native_baseline` | numpy | frozen provider forecast | none |
| `jax_dense_bp` | NESS JAX | dense workspace (parity twin) | `bp_direct` (ff_bp) |
| `fabricpc_feedforward_bp` | FabricPC | feedforward | `fabricpc_bp_through_inference` (ff_bp) |
| `fabricpc_spc_pc_local` | FabricPC | sPC settle (target free) | `workspace_pc_local` |
| `fabricpc_spc_bp_unroll` | FabricPC | sPC settle (target free) | `workspace_bp_unroll` |
| `fabricpc_epc_pc_local` | FabricPC | ePC settle (target free) | `workspace_epc_local` |
| `hybrid_jax_upper_fabricpc_cap` | JAX + FabricPC | sPC settle | `bp_direct` for the jax upper, `fabricpc_pc_local` for the cap (hybrid credit) |

```bash
pip install -e ".[fabricpc]"
ness validate examples/fabricpc_workspace/configs/fabricpc_arms.yaml
ness run examples/fabricpc_workspace/configs/fabricpc_arms.yaml --out runs/fabricpc
ness inspect runs/fabricpc/checkpoints <manifest-id>        # shows the AlgorithmSpec per node/rule
```

Reference run (CPU, 12 updates of batch 8, 64 test requests, 2026-09-26): all FabricPC arms
beat the frozen baseline slightly (test MSE 0.31-0.35 vs 0.36 baseline); this is an engineering
demonstration, not a research result. Note `fabricpc_feedforward_bp` and `fabricpc_spc_bp_unroll`
coincide: on an acyclic workspace with feedforward initialisation the target-free settle is a
no-op (see docs/FABRICPC_BACKEND.md), so both arms are BP through the same deployed computation.
