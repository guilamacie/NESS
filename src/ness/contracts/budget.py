"""Deterministic budgets and cost accounting."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ExecutionBudget:
    fuel: int = 10_000        # interpreter steps
    max_rows: int = 100_000   # intermediate set cardinality
    max_witnesses: int = 1_000
    max_depth: int = 32


@dataclass(frozen=True, slots=True)
class QueryBudget:
    max_candidates: int = 10_000
    max_results: int = 16
    fuel: int = 100_000


@dataclass
class CostLedger:
    """Realised costs, by component and category. Requested budgets are separate."""

    entries: dict[str, float] = field(default_factory=dict)

    def charge(self, key: str, amount: float) -> None:
        self.entries[key] = self.entries.get(key, 0.0) + float(amount)

    def total(self, prefix: str = "") -> float:
        return float(sum(v for k, v in self.entries.items() if k.startswith(prefix)))

    def snapshot(self) -> dict[str, float]:
        return dict(sorted(self.entries.items()))
