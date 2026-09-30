# Installing NESS

NESS is distributed from GitHub (not yet on PyPI). Wherever a command below says
`pip install "ness[extra]"`, install from the repository instead:

```bash
pip install "ness @ git+https://github.com/guilamacie/NESS.git"                # core
pip install "ness[jax] @ git+https://github.com/guilamacie/NESS.git"           # + NESS JAX backend
pip install "ness[fabricpc] @ git+https://github.com/guilamacie/NESS.git"      # + FabricPC 0.6.x backend
pip install "ness[report] @ git+https://github.com/guilamacie/NESS.git"        # + report tooling
pip install "ness @ git+https://github.com/guilamacie/NESS.git@v0.3.0"         # a tagged release
```

NESS has three installation levels. Each level is a superset of the previous one.

| level | command | brings | use when |
|---|---|---|---|
| A. core | `pip install ness` | numpy, pyyaml | contracts, symbolic programs, memory, reasoners, config validation, checkpoint inspection, numpy-runtime plugins |
| B. JAX backend | `pip install "ness[jax]"` | + jax (CPU build) | trainable NESS modules on the NESS JAX backend, `bp_direct`, NESS-owned data parallelism |
| C. FabricPC backend | `pip install "ness[fabricpc]"` | + fabricpc 0.6.x (which itself requires jax) | PC / ePC / recurrent workspaces, FabricPC-backed learning rules |
| report tooling | `pip install "ness[report]"` | + matplotlib, pandas | `ness report <run_dir>` (figures + Markdown from saved artifacts) |

Facts to keep straight:

* Importing `ness` never imports or initialises JAX, CUDA or FabricPC. Only `ness.backends.jax.*`
  and `ness.backends.fabricpc.compat.*` import them, and only after the runtime bootstrap.
* NESS never pins `jaxlib`, `jax-cuda*-plugin` or `jax-cuda*-pjrt`. The `jax` package owns its
  coupled wheel set; FabricPC's own hardware extras select the CUDA variant.
* "FabricPC without JAX" does not exist. Level C implies Level B.
* Supported ranges: `jax>=0.7.0,<0.11` (tested at 0.7.0 and 0.10.2), `fabricpc>=0.6.0,<0.7`
  (verified at 0.6.0; see `compat/fabricpc_compatibility_matrix.json`). A FabricPC outside the
  range fails closed; `NESS_FABRICPC_ALLOW_UNVERIFIED=1` runs it as *experimental*, never verified.

## GPU (Linux, NVIDIA)

Compose NESS with FabricPC's hardware extra in **one** resolver call and use `-U` so the coupled
JAX wheels upgrade together (FabricPC's documented rule):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -U "ness[fabricpc]" "fabricpc[cuda12]"     # CUDA 12
pip install -U "ness[fabricpc]" "fabricpc[cuda13]"     # CUDA 13 (driver >= 580)
ness doctor --backend fabricpc --platform gpu           # confirms JAX actually sees the GPUs
```

Verified on 2026-09-26 with two RTX 3090 (driver 595.84): `pip install -U "ness[fabricpc]"
"fabricpc[cuda12]"` resolved jax/jaxlib/jax-cuda12-plugin/jax-cuda12-pjrt 0.10.2 together.
`nvidia-smi` showing a GPU is not evidence; `ness doctor` reports what JAX sees. A configuration
that requests `platform: gpu` when JAX sees only CPU is a hard error unless it declares
`devices.fallback: cpu` (recorded in the report).

CPU-only machines: `pip install -U "ness[fabricpc]"` (the base `jax` dependency is the CPU build).
Windows/macOS: CPU only (JAX publishes CUDA wheels for Linux only).

## Verify

```bash
ness doctor --backend numpy       # Level A
ness doctor --backend jax         # Level B
ness doctor --backend fabricpc --json doctor.json   # Level C, with executable capability probes
pytest -q                         # tests skip what the environment lacks and never fake support
```

## Environment variables respected (never overridden silently)

`JAX_PLATFORMS`, `XLA_FLAGS`, `XLA_PYTHON_CLIENT_PREALLOCATE`, `CUDA_VISIBLE_DEVICES`,
`FABRICPC_SKIP_XLA_FLAGS`, `FABRICPC_DISABLE_TRITON_GEMM`. The bootstrap records their values;
a value already present in your shell wins over a NESS/FabricPC default.

## Developers

```bash
pip install -e ".[dev]"                     # jax CPU + pytest
pip install -e ".[dev-fabricpc]"            # fabricpc 0.6.x + pytest
pip install -e examples/external_plugin_example
XLA_FLAGS=--xla_force_host_platform_device_count=2 pytest -q tests/fabricpc_backend/test_fabricpc_multidevice.py -m multidevice
```
