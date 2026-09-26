# Changelog

## 0.1.1 (2026-09-26)

* Tests: the eight composition-compile tests that instantiate JAX-backed reference plugins now skip
  when JAX is absent, so the core-without-JAX CI layer is green (no runtime change).

## 0.1.0 (2026-09-26)

First public release: typed-evidence contracts, configuration-compiled macro composition graph,
plugin registry and testkit, numpy / NESS-JAX / FabricPC 0.6 runtimes behind one bootstrap owner,
content-addressed whole-system checkpoints, experiment runner with substitution proofs and causal
checks, `ness report`, and the executed toy vertical slice with its LaTeX/PDF write-up.
