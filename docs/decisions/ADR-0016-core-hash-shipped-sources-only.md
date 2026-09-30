# ADR-0016: The core hash covers shipped sources only

**Decision** `core_source_hash` skips every path component that is hidden (starts with `.`,
e.g. `.ipynb_checkpoints/`), `__pycache__`, the reference plugins and the baseline file: exactly
the sources a wheel ships. An editor autosave in a development tree can no longer change it.

**Why** A Jupyter autosave (`src/ness/memory/.ipynb_checkpoints/store-checkpoint.py`, identical
to `store.py`) made the development tree fail its own integrity test while installed copies were
unaffected.

**Consequences** The rule change itself does not alter the hash of a clean tree (no hidden files
were shipped); the baseline changes at 0.3.0 because core sources changed.

**Tests** `tests/test_core_hash.py::test_editor_and_tool_artefacts_never_enter_the_hash`.
