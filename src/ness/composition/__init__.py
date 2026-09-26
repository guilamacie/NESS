"""Macro composition graph: spec, selectors, parser and compiler."""

from .parse import parse_composition, parse_input, parse_node
from .selectors import OBSERVATION_NODE, SourceSelector
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

__all__ = ["parse_composition", "parse_input", "parse_node", "OBSERVATION_NODE", "SourceSelector", "COMPOSITION_SCHEMA",
           "BoundaryTransformSpec", "CompositionGraphSpec", "InferenceWorkspaceSpec", "InputWiring", "NodeSpec", "OutputSpec", "SourceRef"]
from .compiler import CompiledGraph, CompiledNode, EdgeParamSpec, ResolvedInput, ResolvedSource, compile_graph  # noqa: E402

__all__ += ["CompiledGraph", "CompiledNode", "EdgeParamSpec", "ResolvedInput", "ResolvedSource", "compile_graph"]
