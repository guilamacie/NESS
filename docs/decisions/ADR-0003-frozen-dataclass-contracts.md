# ADR-0003: Frozen dataclasses + canonical JSON hashing for contracts

**Decision** Contracts are frozen `dataclasses` with explicit `__post_init__` validation and a
`canonical()` method feeding a sha256 over canonical JSON (arrays hashed by dtype/shape/bytes).
No pydantic/msgspec dependency. **Why** zero dependencies for plugin authors, hashable identities
under our control, restricted-data serialisation. **Consequence** validation is hand-written and
tested; configs are validated by the compiler, not by a schema library.
