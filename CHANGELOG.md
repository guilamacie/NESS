# Changelog

## 0.2.0 (2026-09-28)

* `ness graph <config>`: diagrams of an experiment for humans: composition graph per arm (nodes by
  layer, per-port wiring, gradient boundaries, trainable nodes), sandwich view, arms matrix and
  data-access map; works from the configuration alone (`--spec-only`, unknown plugins allowed) or
  from the compiled graph. New package `ness.visualize` (matplotlib, `ness[report]`).
* Quick start rewritten (`docs/QUICKSTART.md`): concepts, the experiment file section by section,
  every reference plugin with its ports and configuration keys, three architectures with diagrams
  (`examples/quickstart/`), and the architecture document extended (`docs/ARCHITECTURE.md`).
* Training protocol: every arm, frozen or trainable, sees the same `batch_size x updates`
  training requests (from 0.1.1's reference run onwards).

## 0.1.1 (2026-09-26)

* Tests: the eight composition-compile tests that instantiate JAX-backed reference plugins now skip
  when JAX is absent, so the core-without-JAX CI layer is green (no runtime change).

## 0.1.0 (2026-09-26)

First public release: typed-evidence contracts, configuration-compiled macro composition graph,
plugin registry and testkit, numpy / NESS-JAX / FabricPC 0.6 runtimes behind one bootstrap owner,
content-addressed whole-system checkpoints, experiment runner with substitution proofs and causal
checks, `ness report`, and the executed toy vertical slice with its LaTeX/PDF write-up.
