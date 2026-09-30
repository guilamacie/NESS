"""NESS: a cross-domain research platform for typed-evidence predictive systems.

The package is organised by dependency direction (see docs/ARCHITECTURE.md):

    contracts  <-  plugin_api  <-  composition  <-  runtime  <-  experiments / cli
                                     ^               ^
                 symbolic, memory, probabilistic, tasks, scenarios, learning,
                 checkpoint, observability (all depend only on contracts/plugin_api)
                 runtimes.jax (optional; the only place that imports JAX)

``ness.contracts`` imports neither FabricPC, Hyperon, JAX nor any modality framework.
"""

__version__ = "0.3.0"

CONTRACT_VERSION = "ness.contracts/1"
COMPOSITION_SCHEMA = "ness.composition/1"
EXPERIMENT_SCHEMA = "ness.experiment/3"
PROGRAM_SCHEMA = "ness.program/1"
MANIFEST_SCHEMA = "ness.manifest/1"
