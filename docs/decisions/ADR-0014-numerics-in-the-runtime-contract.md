# ADR-0014: Numerics in the runtime contract

**Decision**
* `runtime.numerics: {matmul_precision: default | high | highest, deterministic: bool}`. The
  bootstrap applies it before the first JAX computation: `jax_default_matmul_precision` is set;
  `deterministic: true` merges `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`
  into the user's `XLA_FLAGS` (read at backend initialisation). A conflicting user flag or
  `JAX_DEFAULT_MATMUL_PRECISION`, or a backend initialised too early to apply the flags, is a
  `ContractViolation`, never a silent override. FabricPC's `setup_jax` respects flags already set.
* `RuntimeReport.numerics` records the declared values, the effective matmul precision, the
  user's numeric XLA flags, the flags NESS applied, the effective numeric flags (including those
  FabricPC adds) and `JAX_DEFAULT_MATMUL_PRECISION` / `NVIDIA_TF32_OVERRIDE`.
* Predictor identity: the manifest's `extra.runtime.numerics` holds the declared numerics, the
  effective precision, the user's numeric flags and environment. It is present only when numerics
  are declared or the user's environment sets something numeric, so manifests of v0.2-style
  configurations keep the v0.2 shape. FabricPC's automatic flags are a function of the FabricPC
  version, which identity already records.

**Why** On GPUs the default float32 matrix product is TF32-class (relative error ~3e-4); two
systems whose predictions differ must not share a manifest id, and the bootstrap is the only
place allowed to set such configuration.

**Consequences** Manifest ids change once at 0.3.0 because the NESS version is part of the
dependency lock (as at every release); otherwise unchanged for configurations without numerics
in an environment without numeric variables.

**Tests** `tests/test_numerics_runtime.py` (C1 identity differs by precision, C2 v0.2 shape,
user environment, deterministic flags merged, conflicts fail closed).
