# Contributing to NESS

NESS is a plugin platform. The most valuable contributions are **new plugins** (substrates,
upper modules, semantic adapters, programs, reasoners, memory stores, caps, tasks, datasets),
**new experiment configurations**, and **compatibility records** for backends and hardware.
Core changes (contracts, composition compiler, runtimes, checkpointing) go through an
architecture decision record in `docs/decisions/`.

## Where to start

| you want to | read |
|---|---|
| run and configure your first experiments | `docs/QUICKSTART.md`, `examples/quickstart/` |
| understand the architecture and interfaces | `docs/ARCHITECTURE.md` |
| plug your own models into the layers | `docs/AGENT_HANDOFF.md` (written for humans and AI coding agents) |
| write and test a plugin package | `docs/CONTRIBUTING_PLUGINS.md`, `examples/external_plugin_example/` |
| wire modules together | `docs/COMPOSITION_GRAPH.md` |
| use or extend the FabricPC backend | `docs/FABRICPC_BACKEND.md`, `docs/FABRICPC_UPGRADE.md`, `compat/` |
| know what is verified vs. experimental vs. design-only | `IMPLEMENTATION_STATUS.md` |

## Development setup

```bash
git clone https://github.com/guilamacie/NESS.git && cd NESS
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"                          # core + NESS JAX backend (CPU) + report tooling + pytest
pip install -e ".[dev-fabricpc]"                 # optionally: the FabricPC 0.6.x backend
pip install -e examples/external_plugin_example  # the external plugin example (used by tests)
python -m pytest -p no:cacheprovider             # jax/fabricpc/GPU tests skip themselves when unavailable
```

## Rules that keep the platform honest

* A plugin never reads a field whose role it is not permitted to see; the compiler and the
  causal check (`assert_target_free`) enforce this, and tests must keep passing.
* New plugins ship with the testkit contract tests (`ness.plugin_api.testkit`) and a
  checkpoint round-trip; they register through the `ness.plugins` entry point, never by
  editing the core registry.
* Do not pin `jaxlib` or CUDA plugin wheels; do not widen the FabricPC compatibility range
  without a new entry in `compat/fabricpc_compatibility_matrix.json` backed by a test run.
* Never report a positive scientific result that the saved artifacts do not support; failed
  runs and substitutions stay in the logs and the report.
* Keep test file basenames unique across `tests/` subdirectories.

## Pull requests

Open an issue or a draft PR early for core changes. A PR should state which acceptance tests
(`docs/ARCHITECTURE.md` §Acceptance) it exercises, include tests, and keep `IMPLEMENTATION_STATUS.md`
accurate. CI runs the core (no JAX), JAX-CPU and FabricPC-CPU layers on every push and pull
request; GPU jobs run on a self-hosted runner when one is available.

## Licence

By contributing you agree that your contributions are licensed under the Apache License 2.0
(see `LICENSE`).
