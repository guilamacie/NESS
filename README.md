# NESS

[![ci](https://github.com/guilamacie/NESS/actions/workflows/ci.yml/badge.svg)](https://github.com/guilamacie/NESS/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)

A composable research platform for the **Neural-Symbolic Sandwich** (NESS) architecture:
frozen neural substrates, trainable upper modules, semantic adapters, typed symbolic programs,
probabilistic reasoners, pinned analogue memory and baseline-preserving caps, composed **by
configuration** into a validated macro graph, learned through the NESS JAX backend or the
[FabricPC](https://github.com/trueagi-io/FabricPC) 0.6 predictive-coding backend, identified by
whole-system manifests, and checkpointed and restored as one system. Every layer is a plugin:
bring your own models and swap them without touching the core.

The platform implements the NESS/FabricPC cross-domain design specification and its
coding-agent handoff addendum. Those two documents are not distributed in this repository
(they are available from the project lead); references such as "PDF §16" or "addendum §3.4"
in the documentation point to them. Everything needed to *use* and *extend* the platform is
in `docs/`.

* **Quick start (configure and run your first experiments):** [`docs/QUICKSTART.md`](docs/QUICKSTART.md)
  with runnable examples in [`examples/quickstart/`](examples/quickstart/).
* **Read first for the design:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (summary, layout, interfaces, how-to).
* **Implementing your own NESS instance with your own models:** [`docs/AGENT_HANDOFF.md`](docs/AGENT_HANDOFF.md).
* **Writing a plugin:** [`docs/CONTRIBUTING_PLUGINS.md`](docs/CONTRIBUTING_PLUGINS.md), then
  [`docs/COMPOSITION_GRAPH.md`](docs/COMPOSITION_GRAPH.md),
  [`docs/PROBABILISTIC_REASONERS.md`](docs/PROBABILISTIC_REASONERS.md),
  [`docs/CHECKPOINT_CONTRACT.md`](docs/CHECKPOINT_CONTRACT.md).
* **What is implemented vs. design-only:** [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md);
  audit of inputs, dependencies and deviations: [`IMPLEMENTATION_AUDIT.md`](IMPLEMENTATION_AUDIT.md);
  decisions: [`docs/decisions/`](docs/decisions/).

## Install and run

From GitHub, no clone needed (pick the level you need; see `docs/INSTALL.md`):

```bash
pip install "ness @ git+https://github.com/guilamacie/NESS.git"                 # Level A: core (numpy, pyyaml)
pip install "ness[jax] @ git+https://github.com/guilamacie/NESS.git"            # Level B: + NESS JAX backend (CPU)
pip install "ness[fabricpc] @ git+https://github.com/guilamacie/NESS.git"       # Level C: + FabricPC 0.6.x backend
pip install -U "ness[fabricpc] @ git+https://github.com/guilamacie/NESS.git" "fabricpc[cuda12]"   # GPU (Linux)
```

From a clone (development):

```bash
git clone https://github.com/guilamacie/NESS.git && cd NESS
python -m venv .venv && . .venv/bin/activate
pip install -e .                                  # Level A: core (numpy, pyyaml) - no JAX, no FabricPC
pip install -e ".[jax]"                           # Level B: NESS JAX backend
pip install -e ".[fabricpc]"                      # Level C: FabricPC 0.6.x backend (implies JAX)
pip install -U -e ".[fabricpc]" "fabricpc[cuda12]" # GPU: compose with FabricPC's hardware extra (Linux)
pip install -e examples/external_plugin_example   # optional third-party plugin example
ness doctor --backend fabricpc --json doctor.json  # versions, bootstrap order, devices, probed capabilities
ness validate examples/vertical_slice_timeseries/configs/vertical_slice.yaml
ness run examples/vertical_slice_timeseries/configs/vertical_slice.yaml --out runs/vertical_slice
ness run examples/fabricpc_workspace/configs/fabricpc_arms.yaml --out runs/fabricpc
pytest -q
```

Details and GPU notes: [`docs/INSTALL.md`](docs/INSTALL.md). The FabricPC backend:
[`docs/FABRICPC_BACKEND.md`](docs/FABRICPC_BACKEND.md); upgrade protocol:
[`docs/FABRICPC_UPGRADE.md`](docs/FABRICPC_UPGRADE.md); compatibility matrix:
[`compat/`](compat/). Everything runs on CPU in a few minutes; GPU is optional.

## The vertical slice

`examples/vertical_slice_timeseries/configs/toy_vertical_slice.yaml` is the reference study:
an interpretable two-regime forecasting problem (target, auxiliary channel, known events,
hidden regime with recorded change points), arms A0-A5 (frozen baseline, neural cap, + symbolic
programs, + exact probabilistic regime reasoner, + pinned memory, same-fact neural control) over
three seeds, ten configuration-only substitution proofs, in-run checkpoint-restore identity and
causal checks. `ness report <run>` regenerates twelve figures and a Markdown report from the
saved artifacts; the write-up is `reports/toy_vertical_slice_report.pdf` (+ `.tex`). It is an
engineering demonstration of the interfaces and controls, not a research result.

## Status in one paragraph

Verified: contracts, composition compiler, plugin registry/discovery, test kit, reference
interpreter, in-memory snapshot store, exact/IS reasoners, the NESS JAX backend (BP, batched and
data-parallel gradients), the FabricPC 0.6.0 backend (target-free sPC/ePC/feedforward workspaces,
PC-local and BP-through-inference rules, hybrid credit, restricted-data checkpoints, probe-based
capabilities on CPU and on 2 GPUs), single runtime bootstrap + `ness doctor`, whole-system
checkpoints, transaction, experiment runner, CLI, vertical slice and substitution proof
(204 tests). Experimental: recurrent FabricPC workspace, masked objectives under the PC
rule. Unsupported (explicit errors): Hyperon memory, TimesFM / vision providers, nudge/EP rules,
model parallelism, multi-host, the reference44 port. See `IMPLEMENTATION_STATUS.md` and
`docs/FABRICPC_INTEGRATION_REPORT.md`.

## License and acknowledgements

Apache License 2.0 (see `LICENSE`). NESS, the Neural-Symbolic Sandwich architecture, is due to
Dr Ben Goertzel; FabricPC is developed by [TrueAGI](https://github.com/trueagi-io/FabricPC).
Cite this software with `CITATION.cff`. Contributions: `CONTRIBUTING.md`.
