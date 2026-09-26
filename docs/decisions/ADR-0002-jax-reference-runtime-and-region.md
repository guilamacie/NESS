# ADR-0002: JAX as the reference differentiable runtime; two-pass differentiable region

**Context** FabricPC is JAX-based but not integrated in this cycle; the vertical slice needs a
trainable upper module and cap with exact gradients, running on CPU in seconds.
**Decision** `runtime = "jax"` is the reference differentiable runtime. Training re-runs only the
jax nodes as a pure function of trainable parameters, feeding all host/frozen/stop-port inputs as
constants recorded during the eager pass (`DifferentiableRegion`). Module code and merges are
written against an array namespace so eager numpy and traced jax share code.
**Alternatives** hand-written numpy backward (poor contributor ergonomics); PyTorch (would make
the FabricPC path a second differentiable runtime from day one).
**Consequences** Exactly the declared semantics: cross-runtime edges are stop-gradient leaves;
forward parity between paths is a test (T05). Per-request tracing is slow at scale; a jitted
batched path is future work. A FabricPC runtime plugs in as another runtime with its own region.
