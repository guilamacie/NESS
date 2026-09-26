# ADR-0009: Recorded deviations and interpretations

1. **`typed_program` generic node** instead of one plugin per program: programs are data
   (`ness.program/1` IR in config). Compatibility: identical semantics; program identity via
   `program_hash` in capabilities and state meta.
2. **Cap feature knownness**: the reference caps lower unknown feature entries to 0 and do not
   consume knownness masks (declared in capabilities). A mask-aware cap is a plugin change.
3. **Single information boundary per request**: tasks with different views in one request raise
   `UnsupportedCapability` (separate state solves are future work).
4. **Recurrent regions** compile (T36) but cannot execute (no verified solver).
5. **Evidence graph holds dense port values as `FeatureEvidence`** (interpretation `dense_port`)
   so lineage is complete; large payloads are hashed lazily.
6. **Vertical-slice frozen providers self-pretrain** deterministically from their own seed
   (independent synthetic process) so the baseline forecast is meaningful and the frozen state
   hash identifies the provider like an external weights digest.
7. **License** field in `pyproject.toml` set to Apache-2.0 as a placeholder decision; confirm.
