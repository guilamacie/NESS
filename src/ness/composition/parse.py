"""Parse the YAML/JSON composition dialect (with its ergonomic sugar) into the spec."""

from __future__ import annotations

from typing import Any

from ..contracts import CompositionError
from .selectors import SourceSelector
from .spec import (
    COMPOSITION_SCHEMA,
    BoundaryTransformSpec,
    CompositionGraphSpec,
    InferenceWorkspaceSpec,
    InputWiring,
    NodeSpec,
    OutputSpec,
    SourceRef,
)


def _parse_boundary(items: Any) -> tuple[BoundaryTransformSpec, ...]:
    if items is None:
        return ()
    if isinstance(items, (str, dict)):
        items = [items]
    out = []
    for it in items:
        if isinstance(it, str):
            out.append(BoundaryTransformSpec(it))
        elif isinstance(it, dict):
            it = dict(it)
            kind = it.pop("kind", None)
            if not kind:
                raise CompositionError(f"boundary transform needs a kind: {it!r}")
            out.append(BoundaryTransformSpec(kind, it))
        else:
            raise CompositionError(f"malformed boundary transform {it!r}")
    return tuple(out)


def _parse_source(item: Any) -> SourceRef:
    if isinstance(item, str):
        return SourceRef(SourceSelector.parse(item))
    if isinstance(item, dict):
        if "from" not in item:
            raise CompositionError(f"source needs 'from': {item!r}")
        return SourceRef(SourceSelector.parse(item["from"]), _parse_boundary(item.get("boundary")))
    raise CompositionError(f"malformed source {item!r}")


def parse_input(port: str, value: Any) -> InputWiring:
    """Accepted forms::

        port: scheme://node/port
        port: {from: ..., boundary: [...]}
        port: {merge: concat, sources: [...], merge_params: {...}, post: [...]}
    """
    if isinstance(value, str) or (isinstance(value, dict) and "from" in value):
        src = _parse_source(value)
        return InputWiring(port, (src,))
    if isinstance(value, dict) and "sources" in value:
        sources = tuple(_parse_source(s) for s in value["sources"])
        return InputWiring(port, sources, value.get("merge"), dict(value.get("merge_params", {})), _parse_boundary(value.get("post")))
    raise CompositionError(f"malformed input wiring for port {port!r}: {value!r}")


def parse_node(d: dict[str, Any]) -> NodeSpec:
    if "id" not in d or "plugin" not in d:
        raise CompositionError(f"node needs 'id' and 'plugin': {d!r}")
    inputs = tuple(parse_input(p, v) for p, v in dict(d.get("inputs", {})).items())
    to = d.get("training_only", False)
    if not isinstance(to, bool):
        raise CompositionError(f"node {d['id']}: training_only must be true/false, got {to!r}")
    return NodeSpec(d["id"], d["plugin"], dict(d.get("config", {})), inputs, tuple(d.get("memory_queries", ())), bool(d.get("enabled", True)), to)


def parse_composition(d: dict[str, Any]) -> CompositionGraphSpec:
    nodes = tuple(parse_node(n) for n in d.get("nodes", []))
    outs = d.get("output") or d.get("outputs")
    if isinstance(outs, dict):
        outs = [outs]
    if not outs:
        raise CompositionError("composition needs an 'output' (task + from)")
    outputs = tuple(OutputSpec(o["task"], SourceSelector.parse(o["from"])) for o in outs)
    regions = tuple(
        InferenceWorkspaceSpec(r["id"], tuple(r["nodes"]), tuple(r.get("state_variables", ())), r.get("energy", ""),
                               r.get("solver", ""), int(r.get("iterations", 0)), r.get("derivative", ""))
        for r in d.get("recurrent_regions", [])
    )
    return CompositionGraphSpec(nodes, outputs, regions, d.get("schema_version", COMPOSITION_SCHEMA))
