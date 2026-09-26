# ADR-0006: Memory is pinned per prediction and published between updates

**Decision** `NessSystem.predict` opens a `MemoryView` (snapshot id + `AvailabilityCut(origin)`
+ task namespaces) and passes it in `ctx.memory_views`; modules never receive the store to write.
Learner appends go to a forked branch; `PredictionTransaction.learn_step` seals and publishes a
new snapshot after the optimizer update. Eligibility filtering happens on the view before any
ranking. Frozen evaluation never publishes. **Why** PDF §14 and addendum §6/§17.2. **Tests**
`test_memory.py`, `TestMemoryArm`.
