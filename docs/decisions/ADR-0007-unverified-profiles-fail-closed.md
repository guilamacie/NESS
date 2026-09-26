# ADR-0007: Profile vocabularies carry status; unverified profiles fail at build time

**Decision** `LEARNING_PROFILES` and `INFERENCE_PROFILES` list every PDF profile with
`verified | unsupported | experimental` and a pointer to evidence. `NessSystem.build` resolves
profiles and raises `UnsupportedCapability` for anything unverified. Dependency lock records
`fabricpc`/`hyperon` as `unresolved`. **Why** REQ-A01/REQ-X01/T01: a name is not a capability;
no silent fallback.
