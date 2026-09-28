# Quick-start examples

Seven small, runnable experiment files, one per idea, explained in
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

```bash
ness validate examples/quickstart/02_neural_cap.yaml
ness run examples/quickstart/02_neural_cap.yaml --out runs/qs/02
ness evaluate runs/qs/02
```

`tests/test_quickstart_configs.py` keeps every file valid.
