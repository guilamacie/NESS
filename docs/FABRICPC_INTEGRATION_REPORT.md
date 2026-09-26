# FabricPC integration report (2026-09-26)

## Verified (contract tests passed through the NESS public runtime path)

* **Core isolation (Level A).** `pip install ness` in a fresh venv without JAX or FabricPC:
  `import ness` imports neither; `ness plugins`, `ness audit`, `ness doctor --backend numpy`,
  `ness validate --schema-only` and the core test files pass; requesting a FabricPC arm or
  backend fails with an explicit `PluginDependencyMissing`.
* **NESS JAX backend (Level B).** `bp_direct` per-item, batched and data-parallel modes agree
  (< 1e-9); finite-difference gradient checks; JAX 0.7.0 (range minimum) and 0.10.2 both pass.
* **FabricPC backend (Level C), FabricPC 0.6.0 / adapter 0.6-adapter/1.0.0.**
  * bootstrap: `setup_jax` runs before backend initialisation (asserted); user `XLA_FLAGS` /
    `JAX_PLATFORMS` win; a late call is detected and refused under `profile: scientific`;
    `jax`-then-`fabricpc` in one process is an explicit error.
  * a real FabricPC graph executes through `NessSystem.predict` with the target free; a real
    PC inference path (sPC and ePC) executes; BP paths execute (feedforward and through the
    finite settle); unsupported/unknown versions and algorithms fail closed.
  * forward, gradient and one-update parity with the NESS JAX dense twin (1e-8..1e-10).
  * PC local rule lowers the clamped energy; BP rule lowers the task loss.
  * hybrid credit: jax upper (`bp_direct`) + FabricPC cap (`fabricpc_pc_local`) in one arm;
    cross-runtime ownership refused.
  * checkpoints: restricted `.npy` arrays + JSON; round trip bitwise; different
    graph/inference configuration or FabricPC family refused.
  * capabilities (probes): basic graph, custom node, skip connections, cyclic graph, sPC, ePC,
    ordinary backprop, BP through deployed finite inference, local PC updates, dtypes, jit,
    checkpoint/restore, deterministic replay, solver diagnostics - verified on CPU and GPU;
    single GPU and data parallelism verified on 2 x RTX 3090 (sharded vs single-device gradients
    max diff 5.6e-17) and on 2 forced CPU devices.
  * end-to-end: seven arms (`examples/fabricpc_workspace`) run through the runner with paired
    reports and checkpoints.

## Experimental

`fabricpc_spc_recurrent` (cyclic workspace, unroll), masked objectives under the PC rule,
FabricPC `InferenceSchedule`, `NESS_FABRICPC_ALLOW_UNVERIFIED=1` override.

## Unsupported (explicit errors)

Nudge / equilibrium-propagation rules (`workspace_ep_centered`), model/tensor parallelism,
multi-host execution, cross-runtime gradient bridges, composition-level recurrent regions,
FabricPC versions outside 0.6.x, GPU requests when JAX sees only CPU (unless `fallback: cpu`).

## Exact environments

| environment | Python | jax / jaxlib | fabricpc | accelerator | tests |
|---|---|---|---|---|---|
| CPU primary | 3.11.10 | 0.10.2 | 0.6.0 (sha256 50766c7c...) | XLA CPU, 1 and 2 forced devices | full suite: 204 collected, 3 skipped |
| CPU minimum JAX | 3.11.10 | 0.7.0 | 0.6.0 | XLA CPU | backend + learning + slice + checkpoint files |
| GPU | 3.11.10 | 0.10.2 + jax-cuda12-plugin/pjrt 0.10.2 | 0.6.0 | 2 x NVIDIA GeForce RTX 3090, driver 595.84, nvidia-cublas-cu12 12.9.2.10 | `ness doctor --platform gpu`; backend suites incl. 2-GPU data parallelism |

Install command verified literally for GPU: `pip install -U "ness[fabricpc]" "fabricpc[cuda12]"`.

## Upstream API limitations encountered

1. Free (target-unclamped) settling on an acyclic graph with `FeedforwardStateInit` is a no-op;
   to make settling matter one needs a non-feedforward state initialiser (now a declared NESS
   option `state_init: global_normal`) or cycles.
2. FabricPC's trainer skips ragged batches under a mesh; NESS refuses them instead.
3. Energies are per sample; coordinate-wise outcome masks cannot be expressed for the PC local
   rule.
4. No nudged/EP phase exists in 0.6.
5. `setup_jax` warns but cannot repair a late call; detection relies on a private JAX symbol
   (`jax._src.xla_bridge.backends_are_initialized`), used defensively as FabricPC does.
6. The `predict()` contract forbids reading `state.z_latent`, which constrains how a NESS node
   could express latent-dependent gating (energy terms only).

## Recommended upstream contributions

* A public, documented `backend_initialized()` helper (or `setup_jax(strict=True)` raising)
  so downstream bootstraps do not depend on `jax._src`.
* A public free-prediction helper (`predict(params, structure, inputs, settle=...)`) mirroring
  `evaluate`'s clamping without metrics, and a documented statement of the feedforward-init
  equilibrium fact in the inference guide.
* Per-sample weights / masks in `graph_energy` and `pc_weight_gradients` (padded positions),
  so masked objectives can reach the local rule.
* A stable, documented checkpoint format for `GraphParams` (or a documented promise that
  `NodeParams` weight keys `source->target:slot` are stable across releases).
* An explicit `ragged_batches="error"` option for mesh training.
