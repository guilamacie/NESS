"""Reference interpreter: set semantics, fuel budgets, witness traces, and orthogonal
truth / completeness / error statuses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..contracts import BudgetExceeded, Completeness, ContractViolation, ExecutionBudget, Knownness, TruthStatus
from .ir import ProgramIR
from .operators import PrimitiveRegistry, default_registry
from .worlds import Fact, RelationWorld


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    value: Any
    output_type: str
    truth_status: TruthStatus
    completeness: Completeness
    status: str  # ok | budget_exhausted | error
    knownness: Knownness
    witnesses: tuple[Fact, ...] = ()
    trace: tuple[str, ...] = ()
    fuel_used: int = 0
    error: str | None = None


class _Fuel:
    def __init__(self, budget: ExecutionBudget) -> None:
        self.budget = budget
        self.used = 0

    def charge(self, n: int) -> None:
        self.used += n
        if self.used > self.budget.fuel:
            raise BudgetExceeded(f"fuel {self.budget.fuel} exhausted")


@dataclass
class _Ctx:
    program: ProgramIR
    bindings: dict[str, Any]
    world: RelationWorld | None
    fuel: _Fuel
    registry: PrimitiveRegistry
    trace: list[str] = field(default_factory=list)
    witnesses: list[Fact] = field(default_factory=list)
    complete: bool = True
    knownness: Knownness = Knownness.KNOWN


class ReferenceInterpreter:
    """Executes ``ProgramIR`` against typed bindings and an optional relation world."""

    interpreter_id = "reference_interpreter"
    interpreter_version = "1.0.0"

    def __init__(self, registry: PrimitiveRegistry | None = None) -> None:
        self.registry = registry or default_registry()

    def execute(self, program: ProgramIR, bindings: dict[str, Any], world: RelationWorld | None = None,
                budget: ExecutionBudget | None = None) -> ExecutionResult:
        budget = budget or ExecutionBudget()
        missing = set(program.inputs) - set(bindings)
        if missing:
            raise ContractViolation(f"program {program.name}: missing bindings {sorted(missing)}")
        ctx = _Ctx(program, bindings, world, _Fuel(budget), self.registry)
        try:
            value = self._eval(program.body, ctx, depth=0)
        except BudgetExceeded as exc:
            return ExecutionResult(None, program.output, TruthStatus.ERROR, Completeness.PARTIAL, "budget_exhausted",
                                   Knownness.UNAVAILABLE_BUDGET, tuple(ctx.witnesses), tuple(ctx.trace), ctx.fuel.used, str(exc))
        except ContractViolation:
            raise
        except Exception as exc:  # transport/runtime failure is ERROR, not unknown
            return ExecutionResult(None, program.output, TruthStatus.ERROR, Completeness.PARTIAL, "error",
                                   Knownness.UNAVAILABLE_BUDGET, tuple(ctx.witnesses), tuple(ctx.trace), ctx.fuel.used, repr(exc))
        completeness = Completeness.COMPLETE if ctx.complete else Completeness.PARTIAL
        truth = self._truth(program.output, value, completeness)
        return ExecutionResult(value, program.output, truth, completeness, "ok", ctx.knownness,
                               tuple(ctx.witnesses), tuple(ctx.trace), ctx.fuel.used)

    # --- semantics -----------------------------------------------------------
    @staticmethod
    def _truth(output_type: str, value: Any, completeness: Completeness) -> TruthStatus:
        if output_type != "truth_result":
            return TruthStatus.UNKNOWN if value is None else TruthStatus.TRUE
        if value is True:
            return TruthStatus.TRUE  # a valid witness establishes truth even under truncation
        if value is False:
            return TruthStatus.FALSE if completeness == Completeness.COMPLETE else TruthStatus.UNKNOWN
        return TruthStatus.UNKNOWN

    def _eval(self, node: dict[str, Any], ctx: _Ctx, depth: int) -> Any:
        if depth > ctx.fuel.budget.max_depth:
            raise BudgetExceeded("max depth exceeded")
        op = node["op"]
        ctx.fuel.charge(1)
        if op == "input":
            return ctx.bindings[node["name"]]
        if op == "const":
            return node["value"]
        if op == "follow":
            if ctx.world is None:
                raise ContractViolation("follow requires a relation world")
            root = ctx.bindings[node["root"]]
            current = frozenset([root]) if isinstance(root, str) else frozenset(root)
            for rel in node["path"]:
                targets, complete = ctx.world.follow(current, rel)
                ctx.witnesses.extend(ctx.world.witnesses(current, rel))
                ctx.fuel.charge(len(targets) + 1)
                if len(targets) > ctx.fuel.budget.max_rows:
                    raise BudgetExceeded("max_rows exceeded")
                if not complete:
                    ctx.complete = False
                ctx.trace.append(f"follow {sorted(current)} -[{rel}]-> {sorted(targets)} complete={complete}")
                current = targets
            return current
        if op in ("intersect", "union"):
            sets = [frozenset(self._eval(it, ctx, depth + 1)) for it in node["items"]]
            ctx.fuel.charge(sum(len(s) for s in sets))
            out = frozenset.intersection(*sets) if op == "intersect" else frozenset.union(*sets)
            ctx.trace.append(f"{op} -> {sorted(out)}")
            return out
        if op == "exists":
            value = self._eval(node["value"], ctx, depth + 1)
            result = len(value) > 0
            ctx.trace.append(f"exists -> {result}")
            return result
        if op == "count":
            value = self._eval(node["value"], ctx, depth + 1)
            return int(len(value))
        if op == "call":
            spec = ctx.registry.get(node["primitive"])
            args: dict[str, Any] = {}
            for k, a in node.get("args", {}).items():
                args[k] = self._eval(a, ctx, depth + 1) if isinstance(a, dict) and "op" in a else a
            missing = set(spec.input_types) - set(args)
            if missing:
                raise ContractViolation(f"primitive {spec.name}: missing args {sorted(missing)}")
            ctx.fuel.charge(spec.cost)
            res = spec.fn(**args)
            if res.knownness != Knownness.KNOWN:
                ctx.knownness = res.knownness
            ctx.trace.append(f"call {spec.name} -> knownness={res.knownness.value} {res.detail}".rstrip())
            return res.value
        raise ContractViolation(f"unknown opcode {op!r}")  # unreachable after IR validation; fail closed
