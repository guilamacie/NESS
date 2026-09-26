"""Rolling-origin helpers."""

from __future__ import annotations


def rolling_origins(n_steps: int, context: int, horizon: int, stride: int = 1) -> tuple[int, ...]:
    """Origins o such that [o-context, o) is observed and [o, o+horizon) is the target."""
    return tuple(range(context, n_steps - horizon + 1, stride))


def chronological_split(origins: tuple[int, ...], train_fraction: float, horizon: int) -> dict[str, tuple[int, ...]]:
    """Chronological split with a gap >= horizon so no training outcome overlaps a test
    observation window's future. Later evaluation origins may observe earlier realised
    values; that is the declared rolling protocol, not leakage."""
    n = len(origins)
    cut = int(n * train_fraction)
    train = origins[:cut]
    test = tuple(o for o in origins[cut:] if not train or o >= train[-1] + horizon)
    return {"train": train, "test": test}
