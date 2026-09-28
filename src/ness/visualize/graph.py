"""Human-facing diagrams of an experiment configuration (``ness graph``).

Four drawings, all produced with matplotlib only (install ``ness[report]``):

* ``<arm>__composition``: the macro composition graph of one arm; nodes by layer, edges
  labelled ``input-port <- source-port`` with their boundary transforms and merges; solid edges
  carry gradients, dashed edges are stop-gradient reads (resolved mode only).
* ``<arm>__stack``: the "sandwich" view of the same arm: which layers are present, which
  plugin fills each, and the arm's learning / inference / memory declarations.
* ``experiment__arms``: one row per arm (and substitution), one column per layer: what each arm
  is made of, at a glance.
* ``experiment__data_access``: the scenario's observation fields with their availability roles
  and which nodes read them; withheld fields are shown as such.

Two levels of detail. *Spec level* uses only the configuration and the plugin registry's
descriptors (no plugin is instantiated, unknown plugin ids are allowed, so an architecture
sketch with not-yet-written plugins can be drawn). *Resolved level* compiles the arm through
the real compiler and adds runtimes, port shapes, trainable groups and gradient boundaries.
``render_experiment`` tries resolved first and falls back to spec level per arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..composition.selectors import OBSERVATION_NODE, SourceSelector
from ..composition.spec import CompositionGraphSpec, NodeSpec
from ..experiments.spec import ArmSpec, ExperimentSpec, parse_arm

# --------------------------------------------------------------------------- vocabulary
KIND_ORDER = ("observation", "substrate", "upper_module", "semantic_adapter", "program", "reasoner", "memory_query", "cap", "output")
KIND_LABEL = {
    "observation": "observations (scenario fields)", "substrate": "lower substrate", "upper_module": "upper module",
    "semantic_adapter": "semantic adapter", "program": "typed program", "reasoner": "probabilistic reasoner",
    "memory_query": "memory query", "cap": "cap (consumer + writer)", "output": "task output", "unknown": "unknown kind",
}
KIND_COLOR = {
    "observation": "#eeeeee", "substrate": "#cfe2f3", "upper_module": "#d9ead3", "semantic_adapter": "#fff2cc",
    "program": "#fce5cd", "reasoner": "#ead1dc", "memory_query": "#d0e0e3", "cap": "#f4cccc", "output": "#ffffff", "unknown": "#f3f3f3",
}
SCHEME_KIND = {"substrate": "substrate", "semantic": "semantic_adapter", "program": "program", "reasoner": "reasoner", "memory": "memory_query"}
LEARNED_BOUNDARIES = {"linear", "layer_norm", "rms_norm"}
LEARNED_MERGES = {"gated_add", "weighted_sum", "attention_pool", "cross_attention"}
ROLE_COLOR = {"observed": "#d9ead3", "known_future": "#cfe2f3", "derived": "#eeeeee", "outcome_only": "#f4cccc", "evaluation_oracle": "#f4cccc"}
WITHHELD_ROLES = {"outcome_only", "evaluation_oracle"}
COLUMN_X = 1.0  # horizontal distance between layers in composition drawings


# --------------------------------------------------------------------------- model
@dataclass
class NodeInfo:
    node_id: str
    plugin: str
    kind: str
    runtime: str | None = None
    trainable: bool | None = None
    input_ports: list[str] = field(default_factory=list)
    output_ports: list[str] = field(default_factory=list)   # "name shape" strings when resolved
    config_summary: str = ""
    enabled: bool = True


@dataclass
class EdgeInfo:
    src_node: str
    src_port: str
    dst_node: str
    dst_port: str
    boundary: list[str] = field(default_factory=list)   # e.g. ["linear(16)"]
    merge: str | None = None
    post: list[str] = field(default_factory=list)
    learned: bool = False
    differentiable: bool | None = None                   # None = not resolved
    shape: str = ""


@dataclass
class GraphModel:
    arm_id: str
    description: str
    nodes: list[NodeInfo]
    edges: list[EdgeInfo]
    observation_fields: dict[str, str]      # field -> role ("?" when unknown)
    output_task: str
    output_selector: str
    learning: dict[str, Any]
    inference: dict[str, Any]
    memory: dict[str, Any]
    resolved: bool
    note: str = ""

    def node(self, node_id: str) -> NodeInfo | None:
        return next((n for n in self.nodes if n.node_id == node_id), None)

    def kinds_present(self) -> dict[str, list[NodeInfo]]:
        out: dict[str, list[NodeInfo]] = {}
        for n in self.nodes:
            out.setdefault(n.kind, []).append(n)
        return out


def _fmt_boundary(b: Any) -> str:
    p = ", ".join(f"{k}={v}" for k, v in sorted(b.params.items()))
    return f"{b.kind}({p})" if p else b.kind


def _config_summary(cfg: dict[str, Any], limit: int = 60) -> str:
    parts = []
    for k, v in cfg.items():
        if isinstance(v, (dict, list, tuple)):
            continue
        parts.append(f"{k}={v}")
    s = ", ".join(parts)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _infer_kinds(spec: CompositionGraphSpec, registry: Any | None) -> dict[str, str]:
    """Kind per node id: from the registry descriptor when the plugin is known, else from the
    selector schemes other nodes use to read it (and the output selector), else 'unknown'."""
    kinds: dict[str, str] = {}
    by_scheme: dict[str, str] = {}
    output_nodes = {o.source.node_id for o in spec.outputs}
    for n in spec.nodes:
        for w in n.inputs:
            for s in w.sources:
                sel = s.selector
                if sel.scheme in SCHEME_KIND:
                    by_scheme[sel.node_id] = SCHEME_KIND[sel.scheme]
                elif sel.scheme == "module":
                    by_scheme.setdefault(sel.node_id, "upper_module")
    for n in spec.nodes:
        kind = None
        if registry is not None and registry.has(n.plugin):
            kind = registry.describe(n.plugin).kind
        if kind is None:
            kind = by_scheme.get(n.node_id)
            if n.node_id in output_nodes and kind in (None, "upper_module"):
                kind = "cap"
        kinds[n.node_id] = kind or "unknown"
    return kinds


def model_from_spec(arm: ArmSpec, registry: Any | None = None, observation_fields: dict[str, str] | None = None) -> GraphModel:
    spec = arm.composition
    kinds = _infer_kinds(spec, registry)
    nodes = [NodeInfo(n.node_id, n.plugin, kinds[n.node_id], config_summary=_config_summary(n.config), enabled=n.enabled,
                      input_ports=[w.port for w in n.inputs]) for n in spec.nodes]
    edges: list[EdgeInfo] = []
    fields: dict[str, str] = dict(observation_fields or {})
    for n in spec.nodes:
        for w in n.inputs:
            for s in w.sources:
                sel = s.selector
                src = OBSERVATION_NODE if sel.scheme == "observation" else sel.node_id
                if sel.scheme == "observation":
                    fields.setdefault(sel.port, "?")
                learned = any(b.kind in LEARNED_BOUNDARIES for b in s.boundary) or (w.merge in LEARNED_MERGES) or any(b.kind in LEARNED_BOUNDARIES for b in w.post)
                edges.append(EdgeInfo(src, sel.port, n.node_id, w.port, [_fmt_boundary(b) for b in s.boundary], w.merge if len(w.sources) > 1 else None,
                                      [_fmt_boundary(b) for b in w.post], learned))
    out = spec.outputs[0] if spec.outputs else None
    return GraphModel(arm.arm_id, arm.description, nodes, edges, fields, out.task_id if out else "?", str(out.source) if out else "?",
                      dict(arm.learning), dict(arm.inference), dict(arm.memory), resolved=False)


def model_from_compiled(arm: ArmSpec, compiled: Any) -> GraphModel:
    nodes: list[NodeInfo] = []
    for cn in compiled.nodes:
        d = cn.descriptor
        nodes.append(NodeInfo(cn.spec.node_id, d.plugin_id, d.module_kind, d.runtime,
                              any(g.mutability == "trainable" for g in d.parameter_groups),
                              [p.name for p in d.input_ports],
                              [f"{p.name} {tuple(p.schema.shape)}" for p in d.output_ports],
                              _config_summary(cn.spec.config), cn.spec.enabled))
    edges: list[EdgeInfo] = []
    for cn in compiled.nodes:
        for port, ri in cn.inputs.items():
            for rs in ri.sources:
                src = OBSERVATION_NODE if rs.producer_node == OBSERVATION_NODE else rs.producer_node
                learned = any(g is not None for g in rs.boundary_groups) or ri.merge_group is not None or any(g is not None for g in ri.post_groups)
                edges.append(EdgeInfo(src, rs.ref.selector.port, cn.spec.node_id, port, [_fmt_boundary(b) for b in rs.ref.boundary],
                                      ri.merge if len(ri.sources) > 1 else None, [_fmt_boundary(b) for b in ri.post], learned,
                                      bool(rs.differentiable), str(tuple(rs.shape_after))))
    fields = {name: f.role.value if hasattr(f.role, "value") else str(f.role) for name, f in compiled.observation_fields.items()}
    task_id, src = next(iter(compiled.outputs.items()))
    return GraphModel(arm.arm_id, arm.description, nodes, edges, fields, task_id, src.selector, dict(arm.learning), dict(arm.inference),
                      dict(arm.memory), resolved=True)


# --------------------------------------------------------------------------- layout helpers
def _layers(model: GraphModel) -> dict[str, tuple[int, int, int]]:
    """node -> (column, row, rows_in_column). Columns follow KIND_ORDER, then dependency depth
    breaks ties inside a kind so a node never sits left of something it reads."""
    deps: dict[str, set[str]] = {n.node_id: set() for n in model.nodes}
    for e in model.edges:
        if e.src_node != OBSERVATION_NODE and e.src_node in deps:
            deps[e.dst_node].add(e.src_node)
    depth: dict[str, int] = {}
    for _ in range(len(model.nodes) + 1):
        for n in model.nodes:
            depth[n.node_id] = 0 if not deps[n.node_id] else 1 + max(depth.get(d, 0) for d in deps[n.node_id])
    col_key: dict[str, tuple[int, int]] = {}
    for n in model.nodes:
        k = KIND_ORDER.index(n.kind) if n.kind in KIND_ORDER else len(KIND_ORDER)
        col_key[n.node_id] = (k, depth[n.node_id])
    distinct = sorted(set(col_key.values()))
    col_of = {key: i + 1 for i, key in enumerate(distinct)}  # column 0 = observations
    groups: dict[int, list[str]] = {}
    for nid, key in col_key.items():
        groups.setdefault(col_of[key], []).append(nid)
    out: dict[str, tuple[int, int, int]] = {OBSERVATION_NODE: (0, 0, 1)}
    for c, ids in groups.items():
        ids = sorted(ids, key=lambda i: (depth[i], i))
        for r, nid in enumerate(ids):
            out[nid] = (c, r, len(ids))
    return out


def _pos(layout: dict[str, tuple[int, int, int]], ncols: int) -> dict[str, tuple[float, float]]:
    pos = {}
    for nid, (c, r, n) in layout.items():
        y = 0.62 if n == 1 else 0.95 - 0.80 * r / (n - 1)
        pos[nid] = (c * COLUMN_X, y)
    return pos


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------------- drawings
def draw_composition(model: GraphModel, path: Path, dpi: int = 150) -> Path:
    """Boxes per node in layer columns; arrows show who reads whom; the wiring of every input
    port (source, boundary transforms, merge, shape) is listed under its destination box, so the
    arrows stay unlabelled and legible."""
    plt = _plt()
    from matplotlib.patches import FancyBboxPatch, Patch
    from matplotlib.lines import Line2D
    layout = _layers(model)
    ncols = max(c for c, _, _ in layout.values()) + 1
    max_rows = max(n for _, _, n in layout.values())
    pos = _pos(layout, ncols)
    row_pitch = 1.0 / max(1, max_rows)
    fig_w = 3.1 * ncols + 1.5
    fig_h = max(3.9, 2.4 * max_rows + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.62, (ncols - 1) * COLUMN_X + 0.62)
    ax.set_ylim(-0.08, 1.06)
    ax.axis("off")
    bw = 0.78
    bh = min(0.16, row_pitch * 0.38)
    wiring_by_node: dict[str, list[str]] = {}
    for e in model.edges:
        src = "observation" if e.src_node == OBSERVATION_NODE else e.src_node
        chain = " -> ".join(e.boundary + ([e.merge] if e.merge else []) + e.post)
        line = f"{e.dst_port} <- {src}/{e.src_port}" + (f"  [{chain}]" if chain else "") + (f"  {e.shape}" if model.resolved and e.shape else "")
        wiring_by_node.setdefault(e.dst_node, []).append(line)

    import textwrap
    extent: list[float] = []  # lowest y reached by any box or wiring block (for a tight vertical frame)

    def box(nid: str, text: str, color: str, lw: float = 0.9, ls: str = "-", under: list[str] | None = None):
        x, y = pos[nid]
        lines = text.split("\n")
        h = bh + 0.028 * max(0, len(lines) - 3)   # grow with the number of lines (observation fields)
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - h / 2), bw, h, boxstyle="round,pad=0.015,rounding_size=0.03", fc=color, ec="k", lw=lw, ls=ls, zorder=3))
        ax.text(x, y + h / 2 - 0.035, lines[0], ha="center", va="center", fontsize=8.2, fontweight="bold", zorder=4)
        ax.text(x, y - 0.02, "\n".join(lines[1:]), ha="center", va="center", fontsize=6.4, zorder=4, linespacing=1.2)
        low = y - h / 2
        if under:
            wrapped: list[str] = []
            for u in under:
                wrapped.extend(textwrap.wrap(u, width=44, subsequent_indent="    ") or [u])
            shown = wrapped[:9] + ([f"... {len(wrapped) - 9} more lines"] if len(wrapped) > 9 else [])
            ax.text(x - bw / 2 + 0.02, low - 0.012, "\n".join(shown), ha="left", va="top", fontsize=5.6, color="0.15", zorder=4,
                    family="monospace", linespacing=1.25, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.75", lw=0.5, alpha=0.95))
            low -= 0.012 + 0.03 * len(shown)
        extent.append(low)

    fields = list(model.observation_fields)
    shown = fields[:5] + (["..."] if len(fields) > 5 else [])
    box(OBSERVATION_NODE, "observations\n" + "\n".join(f"{f} [{model.observation_fields.get(f, '?')}]" if f != "..." else f for f in shown), KIND_COLOR["observation"])
    for n in model.nodes:
        detail = f"[{n.runtime}]" + (" trainable" if n.trainable else " frozen") if model.resolved else f"({KIND_LABEL.get(n.kind, n.kind)})"
        box(n.node_id, f"{n.node_id}\n{n.plugin}\n{detail}", KIND_COLOR.get(n.kind, KIND_COLOR["unknown"]), lw=1.7 if n.trainable else 0.9,
            ls="-" if n.enabled else ":", under=wiring_by_node.get(n.node_id))

    seen: set[tuple[str, str]] = set()
    k = 0
    for e in model.edges:
        if e.src_node not in pos or e.dst_node not in pos:
            continue
        key = (e.src_node, e.dst_node)
        x0, y0 = pos[e.src_node]
        x1, y1 = pos[e.dst_node]
        rad = (0.12, -0.12, 0.22, -0.22)[k % 4] if key in seen else 0.0 if abs(y1 - y0) < 1e-9 else (0.12 if k % 2 else -0.12)
        seen.add(key)
        ls = "-" if (e.differentiable or e.differentiable is None) else "--"
        color = "#1f5f8b" if e.learned else "0.35"
        ax.annotate("", xy=(x1 - bw / 2, y1), xytext=(x0 + bw / 2, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.2 if e.learned else 0.8, color=color, ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0), zorder=2)
        k += 1

    out_node = model.output_selector.split("://")[1].split("/")[0] if "://" in model.output_selector else None
    if out_node in pos:
        x, y = pos[out_node]
        ax.annotate("", xy=(x + bw / 2 + 0.36, y), xytext=(x + bw / 2, y), arrowprops=dict(arrowstyle="-|>", lw=1.3, color="k"), zorder=2)
        ax.text(x + bw / 2 + 0.38, y, f"task {model.output_task}\n<- {model.output_selector.split('/', 2)[-1]}", fontsize=6.6, va="center", ha="left")

    handles = [Patch(fc=KIND_COLOR[kk], ec="k", label=KIND_LABEL[kk]) for kk in KIND_ORDER if kk != "output" and (kk == "observation" or any(n.kind == kk for n in model.nodes))]
    handles += [Line2D([0], [0], color="#1f5f8b", lw=1.2, label="edge with learned parameters (boundary / merge)"),
                Line2D([0], [0], color="0.35", lw=0.8, label="gradient flows into the source" if model.resolved else "reads")]
    if model.resolved:
        handles.append(Line2D([0], [0], color="0.35", lw=0.8, ls="--", label="stop-gradient read (frozen / host / non-differentiable source)"))
        handles.append(Patch(fc="white", ec="k", lw=1.7, label="thick border = trainable parameters"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.4, frameon=False)
    meta = [f"learning: {model.learning.get('rule', 'none (frozen arm)')}", f"inference: {model.inference.get('profile', 'direct')}",
            f"memory: {(model.memory.get('store') or {}).get('plugin', model.memory.get('store')) if model.memory else 'none'}",
            "level: resolved (compiled)" if model.resolved else "level: specification only"]
    title = f"Arm {model.arm_id}" + (f": {model.description}" if model.description else "")
    fig.suptitle(title, fontsize=10, x=0.01, ha="left", y=0.995)
    ax.text(0.0, -0.005, "   |   ".join(meta) + (f"   |   {model.note}" if model.note else "")
            + "\n(under each box: its input ports as  port <- source/port  [boundary -> merge -> post]" + ("  (shape)" if model.resolved else "") + ")",
            transform=ax.transAxes, fontsize=6.6, color="0.25", va="top")
    ax.set_ylim(min(extent) - 0.04, max(y for _, y in pos.values()) + bh / 2 + 0.12)
    fig.subplots_adjust(top=0.80, bottom=0.06)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


STACK_ROWS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("task output", ("output",)),
    ("cap: consumer + baseline-preserving writer", ("cap",)),
    ("evidence: reasoner posterior, memory analogues", ("reasoner", "memory_query")),
    ("evidence: semantic features, typed programs", ("semantic_adapter", "program")),
    ("upper module (trainable)", ("upper_module",)),
    ("lower substrate(s) (frozen provider)", ("substrate",)),
    ("observations (scenario fields, role-tagged)", ("observation",)),
)


def draw_stack(model: GraphModel, path: Path, dpi: int = 150) -> Path:
    plt = _plt()
    from matplotlib.patches import FancyBboxPatch
    present = model.kinds_present()
    fig, ax = plt.subplots(figsize=(12.5, 1.05 * len(STACK_ROWS) + 1.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(STACK_ROWS))
    ax.axis("off")
    for i, (label, kinds) in enumerate(reversed(STACK_ROWS)):
        y = i + 0.5
        items: list[str] = []
        used = False
        for kk in kinds:
            if kk == "output":
                items.append(f"{model.output_task} <- {model.output_selector}")
                used = True
            elif kk == "observation":
                flds = [f"{f} [{r}]" for f, r in model.observation_fields.items()]
                items.extend(", ".join(flds[i:i + 3]) for i in range(0, len(flds), 3))
                used = True
            else:
                for n in present.get(kk, []):
                    items.append(f"{n.node_id}: {n.plugin}" + (f" [{n.runtime}{', trainable' if n.trainable else ''}]" if model.resolved else "")
                                 + (f"   {{{n.config_summary}}}" if n.config_summary else ""))
                    used = True
        color = KIND_COLOR.get(kinds[0], "#f3f3f3") if used else "#fafafa"
        ax.add_patch(FancyBboxPatch((0.3, y - 0.42), 7.2, 0.84, boxstyle="round,pad=0.02", fc=color, ec="k" if used else "0.6", lw=0.9, ls="-" if used else ":"))
        ax.text(0.5, y + 0.24, label, fontsize=8, fontweight="bold", va="center", color="k" if used else "0.55")
        body = "\n".join(items[:4]) + ("\n..." if len(items) > 4 else "") if items else "not used in this arm"
        ax.text(0.5, y - 0.12, body, fontsize=6.2, va="center", color="k" if used else "0.55", linespacing=1.15)
        if i < len(STACK_ROWS) - 1:
            ax.annotate("", xy=(3.9, y + 0.44), xytext=(3.9, y + 0.56), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.4"))
    side = [f"arm: {model.arm_id}", f"learning rule: {model.learning.get('rule', 'none (frozen)')}",
            f"optimizer: {_config_summary(model.learning.get('optimizer', {}))}" if model.learning.get("optimizer") else "optimizer: -",
            f"inference profile: {model.inference.get('profile', 'direct')}",
            f"memory store: {(model.memory.get('store') or {}).get('plugin', model.memory.get('store')) if model.memory else 'none'}",
            f"nodes: {len(model.nodes)}, edges: {len(model.edges)}", "level: " + ("resolved" if model.resolved else "specification")]
    ax.text(7.75, len(STACK_ROWS) - 0.4, "\n".join(side), fontsize=7.0, va="top", family="monospace")
    ax.set_title(f"Sandwich view of arm {model.arm_id}", fontsize=9.5, loc="left")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


MATRIX_COLUMNS = ("substrate", "upper_module", "semantic_adapter", "program", "reasoner", "memory_query", "cap")


def _wrap_id(text: str, width: int = 18) -> str:
    """Break a long plugin id at underscores so it fits a table cell."""
    if len(text) <= width:
        return text
    parts, lines, cur = text.split("_"), [], ""
    for part in parts:
        cand = f"{cur}_{part}" if cur else part
        if len(cand) > width and cur:
            lines.append(cur + "_")
            cur = part
        else:
            cur = cand
    lines.append(cur)
    return "\n".join(lines)


def draw_arms_matrix(exp: ExperimentSpec, models: list[GraphModel], path: Path, dpi: int = 150, n_arms: int | None = None) -> Path:
    plt = _plt()
    cols = [KIND_LABEL[c].replace(" (consumer + writer)", "") for c in MATRIX_COLUMNS] + ["learning", "inference", "memory"]
    rows, cell_colors, labels = [], [], []
    for i, m in enumerate(models):
        present = m.kinds_present()
        row, colors = [], []
        for c in MATRIX_COLUMNS:
            ns = present.get(c, [])
            row.append("\n".join(_wrap_id(n.plugin) for n in ns) if ns else "-")
            colors.append(KIND_COLOR[c] if ns else "#ffffff")
        row += [_wrap_id(m.learning.get("rule", "-")), m.inference.get("profile", "direct"), _wrap_id((m.memory.get("store") or {}).get("plugin", "-")) if m.memory else "-"]
        colors += ["#f3f3f3"] * 3
        rows.append(row)
        cell_colors.append(colors)
        labels.append(m.arm_id if (n_arms is None or i < n_arms) else f"[sub] {m.arm_id}")
    fig_h = 0.62 * (len(rows) + 1) + 0.8
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=rows, rowLabels=labels, colLabels=cols, cellColours=cell_colors, cellLoc="center", bbox=[0.16, 0.0, 0.84, 0.92])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
        if c == -1:
            cell.set_text_props(fontweight="bold", ha="right")
    ax.set_title(f"Experiment {exp.protocol_id}: what each arm is made of (rows below the line are substitution proofs)" if n_arms is not None and n_arms < len(models)
                 else f"Experiment {exp.protocol_id}: what each arm is made of", fontsize=9.5, loc="left")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_data_access(exp: ExperimentSpec, models: list[GraphModel], path: Path, dpi: int = 150) -> Path:
    plt = _plt()
    from matplotlib.patches import FancyBboxPatch, Patch
    fields: dict[str, str] = {}
    for m in models:
        for f, r in m.observation_fields.items():
            if fields.get(f, "?") == "?":
                fields[f] = r
    readers: dict[str, set[str]] = {f: set() for f in fields}
    for m in models:
        for e in m.edges:
            if e.src_node == OBSERVATION_NODE:
                readers.setdefault(e.src_port, set()).add(f"{e.dst_node}.{e.dst_port}")
    consumers = sorted({c for s in readers.values() for c in s})
    fig, ax = plt.subplots(figsize=(11, max(3.6, 0.5 * max(len(fields), len(consumers)) + 1.6)))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fy = {f: 0.92 - 0.84 * i / max(1, len(fields) - 1) if len(fields) > 1 else 0.5 for i, f in enumerate(fields)}
    cy = {c: 0.92 - 0.84 * i / max(1, len(consumers) - 1) if len(consumers) > 1 else 0.5 for i, c in enumerate(consumers)}
    for f, y in fy.items():
        role = fields[f]
        withheld = role in WITHHELD_ROLES
        ax.add_patch(FancyBboxPatch((0.4, y - 0.045), 3.0, 0.09, boxstyle="round,pad=0.01", fc=ROLE_COLOR.get(role, "#f3f3f3"), ec="k", lw=0.8, hatch="//" if withheld else None))
        ax.text(1.9, y, f"{f}  [{role}]" + ("  (withheld at prediction time)" if withheld else ""), ha="center", va="center", fontsize=7)
    for c, y in cy.items():
        ax.add_patch(FancyBboxPatch((6.6, y - 0.045), 3.0, 0.09, boxstyle="round,pad=0.01", fc="#ffffff", ec="k", lw=0.8))
        ax.text(8.1, y, c, ha="center", va="center", fontsize=7)
    for f, cs in readers.items():
        if f not in fy:
            continue
        for c in cs:
            ax.annotate("", xy=(6.6, cy[c]), xytext=(3.4, fy[f]), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.4", connectionstyle="arc3,rad=0.05"))
    handles = [Patch(fc=ROLE_COLOR[r], ec="k", label=r, hatch="//" if r in WITHHELD_ROLES else None) for r in ("observed", "known_future", "derived", "outcome_only", "evaluation_oracle") if r in fields.values()]
    if handles:
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=len(handles), fontsize=6.8, frameon=False)
    ax.set_title(f"Experiment {exp.protocol_id}: observation fields, their availability roles, and who reads them "
                 f"(union over arms; withheld fields cannot be wired, the compiler refuses)", fontsize=8.8, loc="left")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- driver
def build_models(exp: ExperimentSpec, registry: Any | None, arms: list[str] | None = None, spec_only: bool = False,
                 include_substitutions: bool = True) -> tuple[list[GraphModel], int, list[str]]:
    """Models for the selected arms (+ substitutions); resolved when the arm compiles, else spec
    level with a note. Returns (models, number of primary arms, notes)."""
    selected = [a for a in exp.arms if arms is None or a.arm_id in arms]
    subs = [parse_arm(d["id"], d) for d in exp.raw.get("substitutions", [])] if include_substitutions and arms is None else []
    notes: list[str] = []
    scenario = spaces = roles = None
    field_roles: dict[str, str] = {}
    if registry is not None and not spec_only:
        try:
            scenario = registry.create(exp.scenario.plugin, exp.scenario.config)
            tasks = {t.task_id: registry.create(t.plugin, {**t.config, "task_id": t.task_id}) for t in exp.tasks}
            roles = frozenset.intersection(*(t.allowed_roles() for t in tasks.values()))
            spaces = {tid: t.prediction_space() for tid, t in tasks.items()}
            field_roles = {n: (f.role.value if hasattr(f.role, "value") else str(f.role)) for n, f in scenario.observation_fields().items()}
        except Exception as exc:  # noqa: BLE001 - diagrams must degrade, never block
            notes.append(f"scenario/tasks not instantiated ({type(exc).__name__}: {exc}); drawing at specification level")
            scenario = None
    models: list[GraphModel] = []
    for a in selected + subs:
        m: GraphModel | None = None
        if scenario is not None and registry is not None and not spec_only:
            try:
                from ..composition import compile_graph
                compiled = compile_graph(a.composition, registry, scenario.observation_fields(), spaces, roles)
                m = model_from_compiled(a, compiled)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"arm {a.arm_id}: not compiled ({type(exc).__name__}: {str(exc)[:90]}); specification level")
        if m is None:
            m = model_from_spec(a, registry, field_roles)
            m.note = "not compiled: see notes" if not spec_only and registry is not None else ""
        models.append(m)
    return models, len(selected), notes


def render_experiment(config: str | Path, out_dir: str | Path | None = None, arms: list[str] | None = None, spec_only: bool = False,
                      fmt: str = "png", dpi: int = 150, include_substitutions: bool = True, registry: Any | None = None) -> tuple[list[Path], list[str]]:
    """Draw every diagram for an experiment file. Returns (written paths, notes)."""
    from ..config import load_experiment
    exp = load_experiment(config)
    if registry is None:
        from ..plugin_api import default_registry
        registry = default_registry()
    out = Path(out_dir) if out_dir else Path("graphs") / exp.protocol_id
    out.mkdir(parents=True, exist_ok=True)
    models, n_primary, notes = build_models(exp, registry, arms, spec_only, include_substitutions)
    written: list[Path] = []
    for m in models[:n_primary]:
        written.append(draw_composition(m, out / f"{m.arm_id}__composition.{fmt}", dpi))
        written.append(draw_stack(m, out / f"{m.arm_id}__stack.{fmt}", dpi))
    written.append(draw_arms_matrix(exp, models, out / f"experiment__arms.{fmt}", dpi, n_primary))
    written.append(draw_data_access(exp, models, out / f"experiment__data_access.{fmt}", dpi))
    return written, notes
