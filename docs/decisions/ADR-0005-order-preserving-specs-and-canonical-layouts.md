# ADR-0005: Specification mappings are order-preserving; cap feature layout is canonical

**Context** A reasoner's `feature_ports` order defines its emission layout; the cap's `features`
order defined its parameter layout. A checkpoint written with sorted JSON keys changed both and
broke restore. **Decision** Checkpoint indices preserve insertion order; caps concatenate
features in sorted port-name order so their layout never depends on config key order; reasoner
`feature_ports` order remains significant and documented. **Consequence** restore recompiles to
the identical composition hash (tested).
