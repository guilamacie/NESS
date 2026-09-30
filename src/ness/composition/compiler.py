"""Composition compiler/validator.

Resolves every source selector to a typed producer port, checks DAG-ness (cycles are
only legal inside declared recurrent regions), semantic/shape/coordinate compatibility
of merges and boundary transforms, task access policy on every edge, runtime and
gradient boundaries, and records the resolved graph (T43) plus a composition hash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..contracts import (
    AccessPolicyViolation,
    CompositionError,
    Differentiability,
    FieldRole,
    ModuleDescriptor,
    ObservationField,
    PortSchema,
    PortSpec,
    PREDICTION_TIME_ROLES,
    content_hash,
)
from ..plugin_api.module import MacroModule
from ..plugin_api.registry import PluginRegistry
from ..runtimes import DIFFERENTIABLE_RUNTIMES, ops
from .selectors import OBSERVATION_NODE, SourceSelector
from .spec import BoundaryTransformSpec, CompositionGraphSpec, InputWiring, NodeSpec, SourceRef

Shape = tuple[int | None, ...]

FORECAST_KIND_FOR_TYPE = {"point": "point_forecast", "quantile": "quantile_forecast"}


@dataclass(frozen=True, slots=True)
class EdgeParamSpec:
    group_id: str
    owner_node: str
    op_type: str  # boundary | merge
    kind: str
    params: dict[str, Any]
    in_shapes: tuple[Shape, ...]

    def param_shapes(self) -> dict[str, tuple[int, ...]]:
        if self.op_type == "boundary":
            return ops.boundary_param_shapes(self.kind, self.params, self.in_shapes[0])
        return ops.merge_param_shapes(self.kind, self.params, self.in_shapes)

    def init(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if self.op_type == "boundary":
            return ops.init_boundary_params(self.kind, self.params, self.in_shapes[0], rng)
        return ops.init_merge_params(self.kind, self.params, self.in_shapes, rng)


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    ref: SourceRef
    producer_node: str
    producer_port: PortSpec
    producer_plugin: str
    producer_version: str
    boundary_groups: tuple[str | None, ...]
    shape_after: Shape
    semantic_after: str
    coordinate_after: str | None
    differentiable: bool  # gradient may flow from the destination into this producer

    @property
    def selector(self) -> str:
        return str(self.ref.selector)


@dataclass(frozen=True, slots=True)
class ResolvedInput:
    port: PortSpec
    sources: tuple[ResolvedSource, ...]
    merge: str | None
    merge_params: dict[str, Any]
    merge_group: str | None
    post: tuple[BoundaryTransformSpec, ...]
    post_groups: tuple[str | None, ...]
    shape_after: Shape

    def is_passthrough(self) -> bool:
        """Single source, no transforms: typed evidence flows intact."""
        return len(self.sources) == 1 and not self.sources[0].ref.boundary and not self.post and self.merge in (None, "select")


@dataclass
class CompiledNode:
    spec: NodeSpec
    module: MacroModule
    descriptor: ModuleDescriptor
    inputs: dict[str, ResolvedInput]
    differentiable: bool  # runs in the differentiable runtime
    training_only: bool = False  # executes only to supply learning targets (never during prediction)


@dataclass
class CompiledGraph:
    spec: CompositionGraphSpec
    nodes: tuple[CompiledNode, ...]  # topological order
    outputs: dict[str, ResolvedSource]  # task_id -> source
    observation_fields: dict[str, ObservationField]
    edge_params: dict[str, EdgeParamSpec]
    differentiable_runtime: str | None
    stop_edges: tuple[str, ...]
    composition_hash: str
    warnings: tuple[str, ...] = ()
    training_only_nodes: tuple[str, ...] = ()

    def node(self, node_id: str) -> CompiledNode:
        for n in self.nodes:
            if n.spec.node_id == node_id:
                return n
        raise CompositionError(f"unknown compiled node {node_id!r}")

    def node_ids(self) -> tuple[str, ...]:
        return tuple(n.spec.node_id for n in self.nodes)

    def differentiable_nodes(self) -> tuple[str, ...]:
        return tuple(n.spec.node_id for n in self.nodes if n.differentiable)

    def resolved_wiring(self) -> dict[str, dict[str, list[str]]]:
        """node -> input port -> list of resolved source selectors (for manifests/diagnostics)."""
        return {n.spec.node_id: {p: [s.selector for s in ri.sources] for p, ri in n.inputs.items()} for n in self.nodes}

    def init_edge_params(self, rng: np.random.Generator) -> dict[str, dict[str, np.ndarray]]:
        return {gid: spec.init(rng) for gid, spec in sorted(self.edge_params.items())}


# --------------------------------------------------------------------------- helpers
def _observation_port(field: ObservationField) -> PortSpec:
    return PortSpec(
        name=field.name,
        schema=PortSchema("dense", tuple(field.shape)),
        semantic_type=field.semantic_type,
        availability_role=field.role,
        differentiability=Differentiability.STOP,
        coordinate_schema_id=field.coordinates.schema_id,
    )


def _toposort(spec: CompositionGraphSpec, producers: dict[str, set[str]]) -> list[str]:
    """Kahn's algorithm over enabled nodes. Nodes inside one declared recurrent region are
    collapsed into a super-node so their internal cycles are allowed; any other cycle is
    rejected (T36)."""
    region_of: dict[str, str] = {}
    for r in spec.recurrent_regions:
        for nid in r.node_ids:
            region_of[nid] = r.workspace_id
    rep = lambda n: region_of.get(n, n)  # noqa: E731
    nodes = [n.node_id for n in spec.enabled_nodes()]
    supers = sorted({rep(n) for n in nodes})
    indeg = {s: 0 for s in supers}
    adj: dict[str, set[str]] = {s: set() for s in supers}
    for dst, srcs in producers.items():
        for src in srcs:
            if src == OBSERVATION_NODE:
                continue
            a, b = rep(src), rep(dst)
            if a != b and b not in adj[a]:
                adj[a].add(b)
                indeg[b] += 1
    ready = sorted(s for s in supers if indeg[s] == 0)
    order: list[str] = []
    while ready:
        s = ready.pop(0)
        order.append(s)
        for t in sorted(adj[s]):
            indeg[t] -= 1
            if indeg[t] == 0:
                ready.append(t)
                ready.sort()
    if len(order) != len(supers):
        cyclic = sorted(s for s in supers if indeg[s] > 0)
        raise CompositionError(
            f"composition graph has a cycle through {cyclic}; recurrence must be declared as an InferenceWorkspace region")
    expanded: list[str] = []
    for s in order:
        members = [n for n in nodes if rep(n) == s]
        expanded.extend(sorted(members))
    return expanded


# --------------------------------------------------------------------------- compile
def compile_graph(
    spec: CompositionGraphSpec,
    registry: PluginRegistry,
    observation_fields: dict[str, ObservationField],
    task_spaces: dict[str, Any],
    allowed_roles: frozenset[FieldRole],
    *,
    instances: dict[str, MacroModule] | None = None,
) -> CompiledGraph:
    """Validate and resolve a composition. ``task_spaces`` maps task_id -> PredictionSpace.
    ``instances`` lets callers (e.g. checkpoint restore) supply pre-built modules."""

    if allowed_roles - PREDICTION_TIME_ROLES:
        raise AccessPolicyViolation("compile-time allowed roles include outcome/oracle roles")
    enabled = spec.enabled_nodes()
    node_specs = {n.node_id: n for n in enabled}

    # 1. instantiate modules (imports only the plugins this graph uses)
    modules: dict[str, MacroModule] = {}
    descriptors: dict[str, ModuleDescriptor] = {}
    for n in enabled:
        mod = instances[n.node_id] if instances and n.node_id in instances else registry.create(n.plugin, n.config)
        if not isinstance(mod, MacroModule):
            raise CompositionError(f"plugin {n.plugin!r} for node {n.node_id} does not implement MacroModule")
        modules[n.node_id] = mod
        descriptors[n.node_id] = mod.describe()

    # 2. dependency structure
    producers: dict[str, set[str]] = {}
    for n in enabled:
        producers[n.node_id] = set()
        for w in n.inputs:
            for s in w.sources:
                producers[n.node_id].add(s.selector.node_id)
                if s.selector.node_id != OBSERVATION_NODE and s.selector.node_id not in node_specs:
                    if s.selector.node_id in {x.node_id for x in spec.nodes}:
                        raise CompositionError(f"node {n.node_id} reads disabled node {s.selector.node_id!r}")
                    raise CompositionError(f"node {n.node_id} reads unknown node {s.selector.node_id!r}")
    order = _toposort(spec, producers)

    diff_runtimes = {descriptors[n].runtime for n in order if descriptors[n].runtime in DIFFERENTIABLE_RUNTIMES and not node_specs[n].training_only}
    if len(diff_runtimes) > 1:
        raise CompositionError(f"multiple differentiable runtimes in one graph are not supported: {sorted(diff_runtimes)}")
    diff_runtime = next(iter(diff_runtimes)) if diff_runtimes else None

    # 3. resolve inputs
    edge_params: dict[str, EdgeParamSpec] = {}
    stop_edges: list[str] = []
    warnings: list[str] = []
    compiled: dict[str, CompiledNode] = {}

    def producer_port(sel: SourceSelector, dest: str) -> tuple[PortSpec, str, str]:
        if sel.node_id == OBSERVATION_NODE:
            if sel.port not in observation_fields:
                raise CompositionError(f"node {dest}: unknown observation field {sel.port!r}; known {sorted(observation_fields)}")
            f = observation_fields[sel.port]
            return _observation_port(f), "observation", "scenario"
        d = descriptors[sel.node_id]
        if not sel.expects_kind(d.module_kind):
            raise CompositionError(f"node {dest}: selector {sel} uses scheme {sel.scheme!r} but node {sel.node_id} is a {d.module_kind}")
        try:
            return d.output_port(sel.port), d.plugin_id, d.plugin_version
        except KeyError:
            raise CompositionError(f"node {dest}: node {sel.node_id} has no output port {sel.port!r}; ports {[p.name for p in d.output_ports]}") from None

    for nid in order:
        nspec, desc = node_specs[nid], descriptors[nid]
        dest_diff = desc.runtime == diff_runtime and diff_runtime is not None and not nspec.training_only
        resolved_inputs: dict[str, ResolvedInput] = {}
        wired = {w.port for w in nspec.inputs}
        for p in desc.input_ports:
            if p.name not in wired and not p.optional:
                raise CompositionError(f"node {nid}: required input port {p.name!r} is not wired")
        for w in nspec.inputs:
            try:
                dport = desc.input_port(w.port)
            except KeyError:
                raise CompositionError(f"node {nid}: plugin {desc.plugin_id} has no input port {w.port!r}; ports {[p.name for p in desc.input_ports]}") from None
            sources: list[ResolvedSource] = []
            for si, sref in enumerate(w.sources):
                pport, pplugin, pversion = producer_port(sref.selector, nid)
                # access policy: every edge is a read under the task view (addendum §3.4, §17.1)
                if pport.availability_role not in allowed_roles:
                    raise AccessPolicyViolation(
                        f"node {nid} input {w.port}: source {sref.selector} has role {pport.availability_role.value}, "
                        f"not permitted by the task access policy {sorted(r.value for r in allowed_roles)}")
                if not dport.accepts(pport.semantic_type):
                    raise CompositionError(
                        f"node {nid} input {w.port} accepts {dport.accepts_semantic_types} but {sref.selector} has semantic type {pport.semantic_type!r}")
                shape: Shape = tuple(pport.schema.shape)
                coord = pport.coordinate_schema_id
                sem = pport.semantic_type
                groups: list[str | None] = []
                for bi, b in enumerate(sref.boundary):
                    learned = ops.boundary_is_learned(b.kind, b.params)
                    if learned and not dest_diff:
                        raise CompositionError(
                            f"node {nid} input {w.port}: learned boundary transform {b.kind!r} feeds a non-differentiable runtime "
                            f"({desc.runtime}); use a non-learned transform or a differentiable destination")
                    gid = f"{nid}.{w.port}#src{si}.b{bi}:{b.kind}" if learned else None
                    if gid:
                        edge_params[gid] = EdgeParamSpec(gid, nid, "boundary", b.kind, dict(b.params), (shape,))
                    shape = ops.boundary_out_shape(b.kind, b.params, shape)
                    groups.append(gid)
                    if b.kind in ("linear", "flatten", "mean_pool", "last_step"):
                        coord = None  # coordinates changed by projection/reduction
                        sem = f"{b.kind}({sem})"
                # gradient boundary: differentiable only if producer port says so, producer is in the
                # same differentiable runtime, and the destination is differentiable
                src_diff = (
                    dest_diff
                    and sref.selector.node_id != OBSERVATION_NODE
                    and not node_specs[sref.selector.node_id].training_only
                    and descriptors[sref.selector.node_id].runtime == diff_runtime
                    and pport.differentiability == Differentiability.DIFFERENTIABLE
                )
                if not src_diff:
                    stop_edges.append(f"{sref.selector}->{nid}.{w.port}")
                sources.append(ResolvedSource(sref, sref.selector.node_id, pport, pplugin, pversion, tuple(groups), shape, sem, coord, src_diff))
            # merge
            shapes = tuple(s.shape_after for s in sources)
            merge_group: str | None = None
            if len(sources) > 1 or w.merge == "select":
                assert w.merge is not None
                out_shape = ops.merge_out_shape(w.merge, w.merge_params, shapes)
                if w.merge in ("add", "gated_add", "weighted_sum"):
                    coords = {s.coordinate_after for s in sources if s.coordinate_after is not None}
                    if len(coords) > 1:
                        raise CompositionError(
                            f"node {nid} input {w.port}: {w.merge} over incompatible coordinate schemas {sorted(coords)}; add an explicit projection")
                if ops.merge_is_learned(w.merge):
                    if not dest_diff:
                        raise CompositionError(f"node {nid} input {w.port}: learned merge {w.merge!r} requires a differentiable destination")
                    merge_group = f"{nid}.{w.port}#merge:{w.merge}"
                    edge_params[merge_group] = EdgeParamSpec(merge_group, nid, "merge", w.merge, dict(w.merge_params), shapes)
            else:
                out_shape = shapes[0]
            post_groups: list[str | None] = []
            for pi, b in enumerate(w.post):
                learned = ops.boundary_is_learned(b.kind, b.params)
                if learned and not dest_diff:
                    raise CompositionError(f"node {nid} input {w.port}: learned post transform {b.kind!r} requires a differentiable destination")
                gid = f"{nid}.{w.port}#post{pi}:{b.kind}" if learned else None
                if gid:
                    edge_params[gid] = EdgeParamSpec(gid, nid, "boundary", b.kind, dict(b.params), (out_shape,))
                out_shape = ops.boundary_out_shape(b.kind, b.params, out_shape)
                post_groups.append(gid)
            # destination shape check (known dims only)
            if not dport.schema.compatible_shape(PortSchema("dense", out_shape)):
                raise CompositionError(
                    f"node {nid} input {w.port}: delivered shape {out_shape} incompatible with declared {dport.schema.shape}")
            passthrough_kinds_ok = len(sources) == 1 and not sources[0].ref.boundary and not w.post
            if not passthrough_kinds_ok and dport.schema.kind not in ("dense", "feature"):
                raise CompositionError(
                    f"node {nid} input {w.port}: declared kind {dport.schema.kind!r} cannot receive a merged/transformed dense value")
            resolved_inputs[w.port] = ResolvedInput(dport, tuple(sources), w.merge, dict(w.merge_params), merge_group, w.post, tuple(post_groups), out_shape)
        compiled[nid] = CompiledNode(nspec, modules[nid], desc, resolved_inputs, dest_diff, nspec.training_only)

    # 4. outputs
    outputs: dict[str, ResolvedSource] = {}
    for o in spec.outputs:
        if o.task_id not in task_spaces:
            raise CompositionError(f"output names unknown task {o.task_id!r}")
        space = task_spaces[o.task_id]
        pport, pplugin, pversion = producer_port(o.source, f"output:{o.task_id}")
        want_kind = FORECAST_KIND_FOR_TYPE.get(space.forecast_type)
        if pport.schema.kind != want_kind:
            raise CompositionError(
                f"output for task {o.task_id} expects a {space.forecast_type} forecast port, but {o.source} is {pport.schema.kind}")
        if pport.coordinate_schema_id and pport.coordinate_schema_id != space.target.schema_id:
            raise CompositionError(
                f"output for task {o.task_id}: port coordinates {pport.coordinate_schema_id} != task target {space.target.schema_id}")
        if pport.availability_role not in allowed_roles:
            raise AccessPolicyViolation(f"output {o.source} has forbidden role {pport.availability_role.value}")
        outputs[o.task_id] = ResolvedSource(SourceRef(o.source), o.source.node_id, pport, pplugin, pversion, (), tuple(pport.schema.shape), pport.semantic_type, pport.coordinate_schema_id, False)

    # 5. training-only nodes (ADR-0012): frozen, never on a path to a task output
    training_only = tuple(n for n in order if node_specs[n].training_only)
    for nid in training_only:
        desc = descriptors[nid]
        if desc.trainable_groups():
            raise CompositionError(f"training-only node {nid} has trainable parameter groups {[g.name for g in desc.trainable_groups()]}; "
                                   "training-only nodes must be frozen (stop-gradient target providers)")
        if desc.module_kind == "memory_query":
            raise CompositionError(f"training-only node {nid} is a memory query; memory is read only through pinned prediction-time views")
    for nid in order:
        if node_specs[nid].training_only:
            continue
        for w in node_specs[nid].inputs:
            for sref in w.sources:
                if sref.selector.node_id in training_only:
                    raise CompositionError(f"node {nid} reads training-only node {sref.selector.node_id!r}; only training-only nodes may consume "
                                           "a training-only node, so it can never influence a prediction")
    for o in spec.outputs:
        if o.source.node_id in training_only:
            raise CompositionError(f"output for task {o.task_id} reads training-only node {o.source.node_id!r}; training-only nodes never feed predictions")

    comp_hash = content_hash({
        "spec": spec.canonical(),
        "descriptors": {nid: descriptors[nid].canonical() for nid in order},
        "edge_params": sorted(edge_params),
        "observation_fields": {k: [v.role.value, v.semantic_type, v.coordinates.schema_id, list(v.shape)] for k, v in sorted(observation_fields.items())},
    })
    return CompiledGraph(spec, tuple(compiled[n] for n in order), outputs, dict(observation_fields), edge_params,
                         diff_runtime, tuple(stop_edges), comp_hash, tuple(warnings), training_only)
