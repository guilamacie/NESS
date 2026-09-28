# Quick-start examples

Seven small runnable files (one idea each) plus two architecture examples, explained in
[`docs/QUICKSTART.md`](../../docs/QUICKSTART.md):

| file | idea |
|---|---|
| `01_baseline_only.yaml` | the frozen provider's own forecast (the control) |
| `02_neural_cap.yaml` | trainable upper module + residual cap, `learning:` |
| `03_symbolic_probabilistic.yaml` | semantic adapter, typed program, exact reasoner |
| `04_with_memory.yaml` | pinned analogue memory attached to an arm |
| `05_external_plugin_substitution.yaml` | third-party plugins and substitution proofs |
| `06_fabricpc_workspace.yaml` | the cap on the FabricPC backend (needs `ness[fabricpc]`) |
| `07_quantiles.yaml` | quantile prediction space and monotone quantile cap |
| `B_two_providers_quantiles.yaml` | architecture B: two providers, one workspace, sampled reasoner, memory, quantile cap (runs) |
| `C_bring_your_own_models.yaml` | architecture C: a template with placeholder `my_*` plugins; draw it with `ness graph --spec-only` |

```bash
ness validate examples/quickstart/02_neural_cap.yaml
ness run examples/quickstart/02_neural_cap.yaml --out runs/qs/02
ness evaluate runs/qs/02
```

```bash
ness graph examples/quickstart/B_two_providers_quantiles.yaml --out graphs/B      # diagrams of every arm
```

`tests/test_quickstart_configs.py` keeps every file valid.
