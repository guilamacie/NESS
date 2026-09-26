# IMPLEMENTATION_AUDIT.md

Audit performed 2026-09-26 before implementation, as required by the coding-agent prompt.

## 1. Existing code and assets found

| item | status |
|---|---|
| `NESS_FabricPC_CrossDomain_Design.pdf` (68 pp., 25 Sep 2026) | read completely; authoritative; not distributed in this repository (available from the project lead) |
| `NESS_Coding_Agent_Handoff_Addendum.md` | read completely; not distributed in this repository (available from the project lead) |
| `NESS_Initial_Coding_Agent_Prompt.md` | read completely; not distributed in this repository (available from the project lead) |
| existing NESS code in the working folder | **none** (only `.venv` with pip/setuptools) |
| companion assets named by the PDF: `requirements.json`, `configs/reference44.design.json`, `dependency-lock.example.json`, `trusted_helper.program.json`, `temporal_shared_event.program.json`, `vision_shared_entity.program.json`, checker scripts, source register, `IMPLEMENTATION_AUDIT.md` (24 Sep 2026), historical NESS fixtures/checkpoints (v43-v45) | **missing** from the supplied material |
| NESS papers 2-4, GT report | **missing** (referenced only) |

Consequence: R0 reference reconstruction (M0) and the reference44 numerical port (M1) cannot
start: there are no trusted checkpoints, fixtures, parameter maps or historical numbers, and
this implementation does not invent them. The trusted-helper program of PDF §13.2/B.2 was
transcribed from the PDF text as a golden test; its historical alias `v43:681` is retained as
provenance only.

## 2. Dependency and revision status

| dependency | found | revision | status |
|---|---|---|---|
| Python | 3.11.10 (`.venv`) | - | locked in `dependency_lock()` |
| numpy | 2.4.6 | - | core dependency |
| PyYAML | 6.0.3 | - | core dependency (config files) |
| JAX / jaxlib (CPU) | 0.10.2 | - | optional extra `[jax]`; reference differentiable runtime |
| FabricPC | *not installed in this project's venv*. A checkout existed in a sibling local project (package `fabricpc` 0.3.1, commit `256b77bf2751640641a2948f8b95bf93bfb3342a`, fork `iCog-Labs-Dev/ContinualPC-Transformers`, not `trueagi-io/FabricPC` as cited by the PDF) | pinnable but **not adopted** | recorded as `unresolved`; no FabricPC code imported |
| Hyperon (hyperon-experimental) | not found anywhere on the machine | - | `unresolved`; memory contract implemented with the local snapshot backend first |
| TimesFM-3, CNN/ViT/text checkpoints | not present | - | not required for G0; no adapter written |

No moving-main dependency was installed for a scientific run; the only installs are pinned
package versions recorded above (`ness audit` prints the lock).

## 3. Contradictions and ambiguities resolved (documented in ADRs)

* PDF cites `trueagi-io/FabricPC`; the locally available FabricPC is a fork under a different
  organisation with an evolved node API. Integration deferred; when adopted, the exact commit
  and a `BackendCapabilities` record with contract-test evidence are required (PDF §2.3, §10.3).
* Addendum §13 example config has a separate `substrates:` section without inputs; here
  substrates are nodes with explicit input wiring (ADR-0004) so every read is policy-checked.
* Addendum §3.2 sketches `WorkspaceModule.initialize(spec, rng, dependencies)`; implemented as
  `initialize(rng)` with config given at construction and dependencies resolved by the compiler.
* Addendum §5.2 `ProbabilisticReasoner.infer(compiled, evidence, request, state, budget, rng)`;
  implemented with features/knownness arrays extracted from declared ports (the graph does the
  routing) and the same information in `ExecutionTrace`.
* PDF §9.5 "zero outputs reproduce the native grid exactly" required an algebraically
  equivalent writer form to be bitwise exact (ADR-0008).

## 4. Proposed ADR list (written: docs/decisions/)

ADR-0001 one execution verb; ADR-0002 JAX reference runtime + two-pass differentiable region;
ADR-0003 frozen-dataclass contracts + canonical hashing; ADR-0004 substrates as explicitly wired
nodes; ADR-0005 order-preserving specs / canonical cap layout; ADR-0006 memory pinning and
publication; ADR-0007 fail-closed profile vocabularies; ADR-0008 exact-zero writers; ADR-0009
recorded deviations. Still to write when their work starts (PDF §23.4): coarse-vs-fine FabricPC
graph boundary; exact finite-inference BP; program truth/completeness for Hyperon views; no host
callbacks in the default solver; source revision pinning; sparse probability semantics;
post-outcome publication ordering (partially covered by ADR-0006).

## 5. Implementation plan mapped to PDF milestones and tests

| milestone | this cycle | acceptance |
|---|---|---|
| Step B contracts + plugin API (addendum §14) | done | `tests/test_contracts.py`, `test_composition.py`, `test_plugins_external.py` |
| Step C vertical slice + substitution (G0 subset) | done | `test_vertical_slice.py`, `test_runtime_system.py`, `test_learning.py`, `test_checkpoint.py` |
| M0 / R0 reference import | blocked (no fixtures/checkpoints) | T05-T07 numerical parity unstarted |
| M1 / R1 FabricPC coarse port + BP-through-finite-inference | not started; seam: `ness.runtimes` | T05/T06/T13 |
| M2 / R1 symbolic + journal/snapshots + Hyperon view | reference interpreter + local snapshot store done; Hyperon adapter not started | T02/T03/T09 pass on the local backend |
| MX-TS / G-TS TimesFM forecast adapter | not started; quantile task/cap/writer ready | T18 (algebra) passes; T19/T25/T28 unstarted |
| MX-VL / G-VL | not started | - |
| M4-M7 | not started | - |

---

# Addendum (2026-09-26, FabricPC integration iteration)

History preserved: the initial audit above found **no official FabricPC** installed for this
project (only an unrelated historical fork checkout in a sibling workspace) and **no Hyperon**;
integration was correctly deferred and recorded as `unresolved`.

## Resolution

| item | outcome |
|---|---|
| Upstream | `trueagi-io/FabricPC`, PyPI package `fabricpc`. Published versions at audit time: 0.4.0, 0.5.0, 0.5.1, 0.5.2, **0.6.0**. |
| Pinned target | FabricPC **0.6.0**: wheel `fabricpc-0.6.0-py3-none-any.whl`, sha256 `50766c7cda6dbe1325eb7e673410a88ad0bb0cd0e5020ab76f303575c006b46f`; source tag `v0.6.0` = commit `8406e6a838442391fd3089958e1a2c1b44c57eda` (2026-09-15). Dependencies declared by FabricPC: `jax>=0.7.0`, optax, orbax-checkpoint, chex, jaxtyping, numpy, tqdm; extras `cpu`, `cuda12`, `cuda13`, `tfds`, `experiments`, `viz`, `all`, `dev`. |
| Material read | README, `docs/user_guides/01,02,04,06,08,12,17`, CHANGELOG (0.5.0-0.6.0 migration tables), `fabricpc/__init__.py`, `jax_config.py` (`setup_jax`), `core/types.py`, `core/topology.py`, `graph_assembly/graph_construction.py`, `core/inference.py`, `core/inference_epc.py` (via docs), `core/learning.py`, `graph_initialization/*`, `nodes/base.py`, `nodes/linear.py`, `nodes/identity.py`, `nodes/skip_connection.py`, `training/trainer.py`, `tests/test_sharding.py`, `tests/test_jax_config.py`, `tests/test_external_custom_node.py`, `examples/mnist_multi_gpu.py`. |
| Native example run | `scratch native_smoke.py` (now `tests/fabricpc_backend/test_fabricpc_native_smoke.py`): sPC settle (energy 20.29 -> 3.75), ePC settle (-> 3.60), backprop and PC `make_train_step`, skip connection, cyclic graph (`GraphCycleError` without `unroll`; settles with `unroll=2`), free (target-unclamped) prediction, and a forced 2-device CPU mesh step - all succeeded on FabricPC 0.6.0 / JAX 0.10.2. |
| Observed behaviours that shaped the adapter | (1) `setup_jax` binds at *backend initialisation*, warns if late, respects `JAX_PLATFORMS`/`XLA_FLAGS` already set, writes `--xla_gpu_deterministic_ops`, `--xla_gpu_enable_triton_gemm=false`, `--xla_gpu_autotune_level=1`. (2) The 0.6 node contract is `predict()`/`energy()`; error pairing is base-owned. (3) `train`/`evaluate` clamp input+target for both algorithms during training and derive the loss from the output node's energy functional; the backprop objective is the clamped target energy per prediction (N = prediction count). (4) Evaluation (`clamp_target=False`) leaves the target free. (5) `FeedforwardStateInit` puts every unclamped node at zero energy, so a target-free settle on a DAG is a no-op. (6) Multi-device = jit + `NamedSharding` over a `("data",)` mesh; ragged training batches are *skipped* by FabricPC's trainer (NESS refuses them instead). (7) `GraphCycleError` unless `graph(..., unroll=U)`; ePC on cycles minimises the unrolled approximation. (8) No orbax usage inside FabricPC 0.6.0 core; checkpointing of `GraphParams` is the caller's job. |
| API classification used by the adapter | public documented: `setup_jax`, `graph`, `Edge`, `TaskMap`, `GraphCycleError`, `Linear`, `IdentityNode`, `SkipConnection`, `InferenceSGD`, `EPCInference`, `run_inference`, `GaussianEnergy`, `graph_energy`, activations, initializers, `initialize_params`, `initialize_graph_state`, `GlobalStateInit`, `build_clamps`, `pc_weight_gradients`, `grad_denominator`, `GraphParams`/`NodeParams`. Public but weakly documented: `GraphStructure.schedule/node_order/config`. **Private: none** (the only private symbol NESS touches is `jax._src.xla_bridge.backends_are_initialized`, diagnostics only, imported defensively as FabricPC itself does). |
| Environments | CPU: Python 3.11.10, jax/jaxlib 0.10.2, optax 0.2.8. Minimum of the declared JAX range: jax/jaxlib 0.7.0 with FabricPC 0.6.0 (CPU). GPU: same versions + `jax-cuda12-plugin`/`jax-cuda12-pjrt` 0.10.2, `nvidia-cublas-cu12 12.9.2.10`, driver 595.84, 2 x NVIDIA GeForce RTX 3090, installed with `pip install -U "ness[fabricpc]" "fabricpc[cuda12]"` (verified literally). |
| Hyperon | still absent; still `unresolved`. |

## Deviations / interpretations recorded (see ADR-0010, ADR-0011)

* FabricPC's `train(algorithm=...)` is **not** used for NESS learning: its objective
  normalisation, optimizer coupling and algorithm labels do not match NESS's reductions, credit
  ownership and AlgorithmSpec identity. NESS drives FabricPC's public lower-level functions.
* The PDF's `workspace_pc_local`, `workspace_epc_local` and `workspace_bp_unroll` are
  implemented by the FabricPC backend (`fabricpc_pc_local`, `fabricpc_bp_through_inference`) and
  their status moved from `unsupported` to `verified` with the tests named in
  `ness/learning/profiles.py`. `workspace_ep_centered` stays unsupported (no nudge phase in
  FabricPC 0.6 and none implemented in NESS).
* Composition-level recurrent regions remain unsupported; recurrence is available inside a
  FabricPC workspace node (`fabricpc_spc_recurrent`, experimental, explicit opt-in).
