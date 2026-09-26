# FabricPC upgrade and compatibility maintenance

A new FabricPC release is supported only after this protocol completes. Import success is not
support. The compatibility table (`ness/backends/fabricpc/version.py::SUPPORTED_FAMILIES`) and
the matrix (`compat/fabricpc_compatibility_matrix.json`) are widened in the **last** step.

1. **Detect** the release (PyPI `pip index versions fabricpc`; the weekly `fabricpc-canary` CI job
   installs the newest release and upstream `main` non-blockingly and warns on breakage).
2. **Read** `CHANGELOG.md` and the migration tables; list API changes touching:
   `setup_jax`, `graph`/`Edge`/`TaskMap`, `NodeBase` contract, `InferenceSGD`/`EPCInference`/
   `run_inference`, `initialize_graph_state`, `build_clamps`, `pc_weight_gradients`,
   `grad_denominator`, `graph_energy`, `GraphParams`/`NodeParams`, mesh semantics.
3. **Install** in an isolated environment (`python -m venv`, `pip install "fabricpc==X.Y.Z"`),
   record the artifact sha256 (`pip download --no-deps fabricpc==X.Y.Z && sha256sum`) and the
   upstream tag revision.
4. **Native smoke**: `pytest tests/fabricpc_backend/test_fabricpc_native_smoke.py` with
   `NESS_FABRICPC_ALLOW_UNVERIFIED=1` (distinguishes FabricPC changes from adapter bugs).
5. **Probes**: `NESS_FABRICPC_ALLOW_UNVERIFIED=1 ness doctor --backend fabricpc --json doctor.json`.
6. **Adapter tests**: `pytest tests/fabricpc_backend`. If a new API family broke them, add
   `compat/vX_Y.py` (copy `v0_6.py`, adapt, keep the same function surface) and a table row
   with `tested_versions: []` first.
7. **Parity**: `test_fabricpc_parity.py` (jax twin vs FabricPC forward/gradient/update).
8. **Gradients**: probes `bp_through_deployed_finite_inference`, `local_pc_parameter_updates`.
9. **Checkpoints**: `test_fabricpc_checkpoint.py`; if the graph digest or key layout changed,
   write a `migrate_state` in the module and a migration test, or fail closed for old snapshots.
10. **GPU**: `pip install -U "ness[fabricpc]" "fabricpc[cudaNN]"`; `ness doctor --platform gpu`;
    `pytest -m "gpu or multidevice" tests/fabricpc_backend`.
11. **Matrix**: append an entry per environment exercised (versions, artifact hash, revision,
    adapter version, platform, accelerator, device counts, verified PC/BP algorithms, checkpoint
    compatibility, test artifact id, known issues).
12. **Widen** `SUPPORTED_FAMILIES`/`tested_versions` and the `fabricpc` extra range in
    `pyproject.toml`; add the version to the `fabricpc-cpu` CI matrix.

Private-API rule: if an adapter must use a private FabricPC symbol, isolate it in the family
module, document why, add a focused compatibility test, mark the NESS capability experimental,
and open an upstream issue/PR. The 0.6 adapter uses no private FabricPC API (the only private
symbol NESS touches is `jax._src.xla_bridge.backends_are_initialized`, diagnostics only, imported
defensively exactly as FabricPC does).
