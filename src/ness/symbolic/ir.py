"""Typed program IR (``ness.program/1``).

Deliberately small: entity roots, typed relation traversal (with inverses), endpoint
intersection/union, existential nonemptiness, bounded count, and calls to registered
domain-neutral primitives. Every op declares types; unknown ops fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contracts import ContractViolation, content_hash, require_schema

PROGRAM_SCHEMA = "ness.program/1"

TYPES = ("entity", "entity_set", "truth_result", "numeric", "numeric_vector", "numeric_series", "person", "object", "int")
OPS = ("input", "follow", "intersect", "union", "exists", "count", "const", "call")


@dataclass(frozen=True, slots=True)
class ProgramIR:
    name: str
    semantics: str
    inputs: dict[str, str]     # input name -> type
    output: str                # output type
    body: dict[str, Any]       # nested op tree
    schema_version: str = PROGRAM_SCHEMA
    semantic_version: str = "1"
    relation_schema_version: str = "1"
    historical_alias: str | None = None
    example_status: str = "reference"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, PROGRAM_SCHEMA)
        for n, t in self.inputs.items():
            if t not in TYPES:
                raise ContractViolation(f"program {self.name}: input {n} has unknown type {t!r}")
        if self.output not in TYPES:
            raise ContractViolation(f"program {self.name}: unknown output type {self.output!r}")
        _validate_ops(self.body, self.inputs)

    def canonical(self) -> dict:
        """Canonical form: input names replaced by positional placeholders so renaming
        an input does not change identity."""
        order = list(self.inputs.keys())
        rename = {n: f"${i}" for i, n in enumerate(order)}
        return {
            "schema_version": self.schema_version,
            "semantics": self.semantics,
            "semantic_version": self.semantic_version,
            "relation_schema_version": self.relation_schema_version,
            "inputs": [self.inputs[n] for n in order],
            "output": self.output,
            "body": _rename(self.body, rename),
        }

    @property
    def program_hash(self) -> str:
        return content_hash(self.canonical())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProgramIR":
        return cls(
            name=d["name"],
            semantics=d["semantics"],
            inputs=dict(d["inputs"]),
            output=d["output"],
            body=d["body"],
            schema_version=d.get("schema_version", PROGRAM_SCHEMA),
            semantic_version=str(d.get("semantic_version", "1")),
            relation_schema_version=str(d.get("relation_schema_version", "1")),
            historical_alias=d.get("historical_alias"),
            example_status=d.get("example_status", "reference"),
        )


def _validate_ops(node: Any, inputs: dict[str, str], depth: int = 0) -> None:
    if depth > 64:
        raise ContractViolation("program nesting too deep")
    if not isinstance(node, dict) or "op" not in node:
        raise ContractViolation(f"malformed IR node: {node!r}")
    op = node["op"]
    if op not in OPS:
        raise ContractViolation(f"unknown opcode {op!r}; unknown opcodes fail closed")
    if op == "input":
        if node.get("name") not in inputs:
            raise ContractViolation(f"input {node.get('name')!r} not declared")
    elif op == "follow":
        if node.get("root") not in inputs:
            raise ContractViolation(f"follow root {node.get('root')!r} must be a declared input")
        path = node.get("path")
        if not isinstance(path, list) or not path or not all(isinstance(r, str) and r for r in path):
            raise ContractViolation("follow.path must be a non-empty list of relation names")
    elif op in ("intersect", "union"):
        items = node.get("items")
        if not isinstance(items, list) or len(items) < 1:
            raise ContractViolation(f"{op}.items must be a non-empty list")
        for it in items:
            _validate_ops(it, inputs, depth + 1)
    elif op in ("exists", "count"):
        _validate_ops(node["value"], inputs, depth + 1)
    elif op == "const":
        if "value" not in node:
            raise ContractViolation("const requires value")
    elif op == "call":
        if not isinstance(node.get("primitive"), str):
            raise ContractViolation("call requires primitive name")
        for _, a in sorted(node.get("args", {}).items()):
            if isinstance(a, dict) and "op" in a:
                _validate_ops(a, inputs, depth + 1)


def _rename(node: Any, rename: dict[str, str]) -> Any:
    if isinstance(node, dict):
        out = {}
        for k, v in sorted(node.items()):
            if k in ("name", "root") and isinstance(v, str) and v in rename:
                out[k] = rename[v]
            else:
                out[k] = _rename(v, rename)
        return out
    if isinstance(node, list):
        return [_rename(v, rename) for v in node]
    return node
