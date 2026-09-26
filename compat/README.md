# Compatibility matrix

`fabricpc_compatibility_matrix.json` is the machine-readable record of every environment in which
the FabricPC backend was exercised. Fields per entry: NESS version, FabricPC version, artifact
hash and upstream revision, adapter version, Python/JAX/jaxlib/optax, platform, accelerator
runtime and model, device counts tested, data-parallel / model-parallel / multi-host status, PC
and BP algorithms verified, checkpoint compatibility, test artifact, known issues.

Current entries (2026-09-26): FabricPC 0.6.0 + JAX 0.10.2 on Linux x86_64 CPU (1 and 2 forced
host devices) and on 2 x NVIDIA RTX 3090 (CUDA 12). JAX 0.7.0 (minimum of the declared range)
with FabricPC 0.6.0 on CPU is recorded once its test run is attached.

Launchers write the resolved environment into every experiment report (`runtime`) and the
identity-relevant part (backend, jax/jaxlib/fabricpc versions, adapter, platform, x64) into
every predictor manifest.
