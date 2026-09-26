# ADR-0008: Baseline-preserving writers must be bitwise identical at zero correction

**Decision** `point_residual_writer` returns `baseline + a`; `monotone_quantile_writer` uses the
algebraically equivalent form `Q0_k + a + cumsum(gap_j * expm1(d_j))` so `expm1(0) == 0` gives the
native grid exactly, instead of reassembling the grid by cumulative sums (1-ulp drift).
**Why** PDF §9.5 / T18: zero outputs reproduce the native grid *exactly*; parity claims must be
testable with equality.
