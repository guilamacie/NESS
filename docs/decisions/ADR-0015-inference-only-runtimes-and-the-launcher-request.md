# ADR-0015: Inference-only runtimes and the launcher request

**Decision**
* Module runtimes map to the backend a process must bootstrap: `numpy`/`host` -> numpy, `jax` ->
  jax, `jax_inference` -> jax, `fabricpc` / `fabricpc_inference` -> fabricpc; unknown runtimes
  (e.g. `torch`) need none. Inference-only runtimes are never part of a differentiable region:
  their outputs are constants to every rule, like numpy modules, and they need no `apply`.
* One rule for both launch paths (`NessSystem.build` and `run_experiment`): a node needs the more
  capable of its runtime's backend and its declared `requires`; the runner compiles each arm to
  apply it (falling back to `requires` for an arm that does not compile, which then fails and is
  recorded).
* `ensure_bootstrapped(backend)` returns an existing report that already satisfies the backend;
  when a launcher request exists but names a less capable backend it re-bootstraps with the
  *recorded request* (platform, devices, x64, profile, numerics) and only the backend raised,
  noting it. Defaults are used only when no launcher request exists at all.
* A prediction during which a node forced such an upgrade fails closed (and keeps failing for
  that system): its manifest would otherwise misstate the runtime.

**Why** v0.2 ranked unknown runtimes as numpy, so an arm of frozen JAX providers bootstrapped
numpy and the first JAX call bootstrapped JAX with defaults (`x64: false` became true) while the
manifest said numpy.

**Tests** `tests/test_numerics_runtime.py` (C3 inference-only arm with x64 false on the requested
platform, report and manifest say jax, runner rule agrees; C4 on-demand upgrade honours the
request; a node hiding its backend fails closed).
