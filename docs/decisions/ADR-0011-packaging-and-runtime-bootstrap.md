# ADR-0011: Three-level packaging and a single runtime bootstrap owner

**Decision**
* Extras: `ness` (numpy, pyyaml), `ness[jax]` (`jax>=0.7.0,<0.11`), `ness[fabricpc]`
  (`fabricpc>=0.6.0,<0.7`). NESS never pins `jaxlib` or CUDA plugin wheels; GPU installs compose
  with FabricPC's hardware extras in one resolver call (`pip install -U "ness[fabricpc]" "fabricpc[cuda12]"`).
* `ness.runtimes.bootstrap.bootstrap(RuntimeRequest)` is the only code that may initialise a
  numerical backend: it respects `JAX_PLATFORMS`/`XLA_FLAGS` already in the environment, calls
  `fabricpc.setup_jax` before the first JAX computation, sets x64, resolves the `DeviceSpec`
  (requested / visible / selected / realized) and records a `RuntimeReport`. `NessSystem.build`,
  the runner, the CLI and `ness doctor` call it before constructing numerical plugins; backends
  call `ensure_bootstrapped` (recorded as implicit) when a caller skipped the launcher.
* Import-time rules, enforced by subprocess tests: `import ness` imports neither JAX nor FabricPC
  and mutates no XLA environment; importing a backend module does not initialise the backend;
  a late `setup_jax` is detected and fails closed for `profile: scientific`; a `jax` bootstrap
  followed by a `fabricpc` request is an explicit error.
* Experiment identity: predictor manifests record backend, jax/jaxlib/fabricpc versions, adapter
  version, platform and x64; device ids/kinds/env flags go to reports and diagnostics.

**Why** FabricPC's `setup_jax` binds at backend initialisation silently; scattered device
queries at import would defeat it and make GPU behaviour depend on import order. Independent
`jaxlib`/CUDA pins conflict with the coupled wheel set JAX manages.

**Consequences** One process runs one backend configuration; multi-backend experiments bootstrap
the most capable backend first (fabricpc satisfies jax requests). `ness doctor` reports the
outcome unambiguously (text + JSON).
