# The FabricPC backend

```
                    NESS contracts (numpy only)
                            |
           +----------------+----------------+
           |                                 |
   ness.backends.jax               ness.backends.fabricpc
   (DifferentiableRegion,          (version routing, translate,
    BPDirectRule, sharding)         compat/v0_6, module, rules)
           |                                 |
          JAX                            FabricPC 0.6.x
                                             |
                                            JAX
```

FabricPC is a numerical graph / inference backend. NESS owns observations, access policy,
evidence, programs, memory, tasks, experiment protocols, manifests and provenance; the
adapter packs bounded numeric evidence into a FabricPC graph and translates back.

## What the backend provides

| plugin | kind | what |
|---|---|---|
| `fabricpc_residual_cap` | cap | a dense PC workspace (input -> tanh hidden layers -> zero-init linear readout; optional lateral feedback with `unroll`) whose **target-free** readout `z_mu` is added to the baseline forecast (residual point writer) |
| `fabricpc_pc_local` | learning rule | clamp input + target correction, settle with the node's solver, FabricPC local weight gradients (means per prediction). PDF `workspace_pc_local` (sPC) / `workspace_epc_local` (ePC) |
| `fabricpc_bp_through_inference` | learning rule | reverse-mode gradient of the NESS task loss through `initialize_graph_state + run_inference` with the target free. PDF `ff_bp` (feedforward) / `workspace_bp_unroll` (settled) |

Inference profiles (declared on the cap's `inference.profile` **and** the arm's `inference.profile`):
`fabricpc_feedforward`, `fabricpc_spc` (InferenceSGD), `fabricpc_epc` (EPCInference), and the
experimental `fabricpc_spc_recurrent` (cyclic hidden graph, `unroll: U`, `allow_experimental: true`).
`inference.state_init` is `feedforward` (default) or `global_normal` (latents drawn from N(0, std)
with a request-local key).

## Semantics that are not negotiable

* **Prediction is target-free**: the adapter clamps the packed evidence only
  (`build_clamps(..., clamp_target=False)`); the readout is free. The PC *training* clamp
  (input and target) happens only inside `fabricpc_pc_local`.
* **A settled DAG at feedforward init is the feedforward pass.** With `FeedforwardStateInit`
  every unclamped node starts at `z_latent = z_mu` (zero energy), so free sPC/ePC settling on an
  acyclic workspace changes nothing. `fabricpc_spc` and `fabricpc_feedforward` therefore have the
  same deployed predictor and differ only in learning; the manifest ids still differ because the
  algorithm identity differs. For settling to change the deployed prediction use
  `state_init: global_normal` or the recurrent profile (`tests/fabricpc_backend/test_fabricpc_parity.py`).
* **No cross-runtime gradient.** A FabricPC node is a stop-gradient consumer of any jax node
  and vice versa; the credit map refuses a rule that would own parameters in another runtime
  (`GradientBoundaryError`). Learned boundary transforms cannot feed a FabricPC node.
* **AlgorithmSpec, not a string.** Every FabricPC node records a complete spec (graph digest,
  state variables, initialisation, energies, clamps at prediction and at learning, solver with
  rate/steps/unroll, derivative, readout, learning rule, loss, reductions, parameter masks,
  FabricPC version + adapter version) in `manifest.extra["algorithm_specs"]`. FabricPC's
  `algorithm="pc"|"backprop"` trainer labels are never used as identity; NESS uses FabricPC's
  public lower-level functions in a NESS-controlled loop.
* **Checkpoints hold data.** FabricPC parameters are stored as `.npy` arrays keyed
  `<node>|weights|<edge>` / `<node>|biases|<name>` plus JSON metadata (graph digest hash,
  FabricPC version/family, adapter version). Restore rebuilds the FabricPC structure from NESS
  config, checks the digest and family, then loads arrays. No FabricPC object is ever pickled.

## Version routing

`ness.backends.fabricpc.version` classifies the installed FabricPC: `verified` (tested
version, e.g. 0.6.0), `qualified` (untested patch in the 0.6 family), `unsupported` (other
family; fails closed), `experimental_override` (`NESS_FABRICPC_ALLOW_UNVERIFIED=1`). The
`scientific` runtime profile accepts only `verified`. One adapter module per family
(`compat/v0_6.py`); supporting 0.7 means adding `compat/v0_7.py` and a table row after the
upgrade protocol, not touching NESS core.

## Capabilities

`ness doctor --backend fabricpc` runs executable probes through the adapter and prints a
status per capability. Recorded evidence: `compat/fabricpc_compatibility_matrix.json`.
Summary for FabricPC 0.6.0 / JAX 0.10.2:

| capability | status |
|---|---|
| basic graph execution, custom nodes, skip connections, cyclic graphs (unroll) | verified |
| state-based PC inference (sPC), error-parameterised PC (ePC) | verified |
| ordinary backprop (feedforward), BP through the deployed finite inference | verified |
| local PC parameter updates | verified |
| dtypes (float64 under NESS x64), jit, checkpoint/restore, deterministic replay, solver diagnostics | verified |
| single GPU, data parallelism (2 GPUs; 2 forced CPU devices) | verified |
| InferenceSchedule (ePC -> sPC composition), masked objectives (BP rule yes, PC rule no) | experimental |
| nudge / equilibrium-propagation rules, model parallelism, multi-host | unsupported |

## Data parallelism

Set `rule_config: {data_parallel: true}` on a FabricPC rule (or `bp_direct`). The batch is
sharded over a `("data",)` mesh built from the *selected* devices; parameters are replicated;
the objective is sum-of-numerators / sum-of-denominators, never a mean of local means. A
batch not divisible by the device count is refused (no silent drop). Gradients, loss and the
post-update predictions match the single-device path to < 1e-9 (tests, 2 CPU devices and 2
GPUs). Model/tensor parallelism and multi-host execution are represented separately in the
capability table and are unsupported.

## Configuration example

See `examples/fabricpc_workspace/configs/fabricpc_arms.yaml` (native baseline, jax dense twin,
FabricPC feedforward+BP, sPC+PC-local, sPC+BP-unroll, ePC+PC-local, hybrid jax-upper + FabricPC-cap).
