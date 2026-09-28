"""Static figures for docs/ARCHITECTURE.md: the package map with its dependency direction and
the prediction transaction. Regenerate with `python tools/draw_architecture_figures.py`.
Needs matplotlib (ness[report]). The content is deliberately hand-written: it documents the
intended architecture, which the import-order tests enforce."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "architecture"


def box(ax, x, y, w, h, title, body="", fc="#f3f3f3", fs=7.5, title_fs=8.5, ec="k", lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + 0.02, y + h - 0.03, title, fontsize=title_fs, fontweight="bold", va="top", ha="left", zorder=3)
    if body:
        ax.text(x + 0.02, y + h - 0.075, body, fontsize=fs, va="top", ha="left", zorder=3, linespacing=1.25)


def arrow(ax, p, q, text="", color="0.3", ls="-"):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", lw=1.0, color=color, ls=ls), zorder=4)
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.012, text, fontsize=6.3, ha="center", color=color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9))


def package_map(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # layer 0: contracts
    box(ax, 0.26, 0.835, 0.48, 0.105, "ness.contracts  (numpy only; imports no backend)",
        "observations & roles, access policy, evidence types, forecasts & prediction spaces, ports & descriptors,\ncapabilities, state snapshots, manifests, errors", fc="#eeeeee")
    # layer 1: plugin API, composition, symbolic, memory, probabilistic, tasks, scenarios
    box(ax, 0.02, 0.62, 0.22, 0.18, "ness.plugin_api", "MacroModule / BaseModule\nPortValue, RuntimeContext\nregistry (entry points)\ntestkit (contract tests)", fc="#fff2cc")
    box(ax, 0.26, 0.62, 0.22, 0.18, "ness.composition", "CompositionGraphSpec\nselectors scheme://node/port\nparser (YAML/JSON)\ncompiler (validate + resolve)", fc="#fff2cc")
    box(ax, 0.50, 0.62, 0.22, 0.18, "host-side subsystems", "ness.symbolic  (program IR, interpreter)\nness.memory  (MemoryStore, snapshots)\nness.probabilistic  (reasoners)\nness.tasks / ness.scenarios", fc="#fff2cc")
    box(ax, 0.74, 0.62, 0.24, 0.18, "ness.runtimes", "ops: boundary / merge vocabulary (xp)\nbootstrap: THE runtime owner\ndevices, doctor", fc="#fff2cc")
    # layer 2: runtime, learning, inference, checkpoint
    box(ax, 0.02, 0.38, 0.30, 0.18, "ness.runtime", "GraphExecutor (topological execution)\nNessSystem (build, manifest, snapshot, restore)\nPredictionTransaction (predict / reveal / learn)", fc="#d9ead3")
    box(ax, 0.34, 0.38, 0.22, 0.18, "ness.learning / ness.inference", "credit map (who owns which parameters)\noptimizers (adam, sgd)\nlearning & inference profile vocabularies", fc="#d9ead3")
    box(ax, 0.58, 0.38, 0.18, 0.18, "ness.checkpoint", "content-addressed blobs\nCheckpointStore (CAS publish)\nSystemSnapshot", fc="#d9ead3")
    box(ax, 0.78, 0.38, 0.20, 0.18, "ness.observability", "PredictionRecord\nExperienceEvent\ncost ledger", fc="#d9ead3")
    # layer 3: backends
    box(ax, 0.02, 0.14, 0.30, 0.18, "ness.backends.jax", "DifferentiableRegion (re-run jax nodes\nas a function of params)\nBPDirectRule; batched / sharded objective", fc="#cfe2f3")
    box(ax, 0.34, 0.14, 0.34, 0.18, "ness.backends.fabricpc", "version routing -> compat/v0_6 (the ONLY FabricPC importer)\ntranslate (AlgorithmSpec), module (workspace cap)\nrules (pc_local, bp_through_inference), capability probes", fc="#cfe2f3")
    box(ax, 0.70, 0.14, 0.28, 0.18, "ness.reference_plugins + external packages", "toy substrates, upper modules, semantic adapter,\ntyped program, reasoners, memory query, caps,\nwriters/consumers  (registered via entry points)", fc="#f4cccc")
    # layer 4: experiments, cli, visualize
    box(ax, 0.02, 0.005, 0.46, 0.105, "ness.experiments / ness.config", "ExperimentSpec (arms, protocol, substitutions); runner (paired arms, seeds,\ncheckpoints, causal checks, artifacts); toy_report; YAML/JSON loading", fc="#ead1dc")
    box(ax, 0.50, 0.005, 0.48, 0.105, "ness.cli / ness.visualize", "ness validate | run | evaluate | report | graph | inspect | audit | plugins | doctor\nvisualize: composition / stack / arms / data-access diagrams", fc="#ead1dc")
    # dependency arrows (point from dependant to dependency)
    for x in (0.13, 0.37, 0.61, 0.86):
        arrow(ax, (x, 0.80), (0.50, 0.835), "")
    arrow(ax, (0.17, 0.56), (0.13, 0.62)); arrow(ax, (0.22, 0.56), (0.37, 0.62)); arrow(ax, (0.28, 0.56), (0.61, 0.62)); arrow(ax, (0.30, 0.56), (0.86, 0.62))
    arrow(ax, (0.45, 0.56), (0.37, 0.62)); arrow(ax, (0.67, 0.56), (0.50, 0.835), ""); arrow(ax, (0.88, 0.56), (0.50, 0.835), "")
    arrow(ax, (0.17, 0.32), (0.13, 0.38)); arrow(ax, (0.24, 0.32), (0.45, 0.38)); arrow(ax, (0.51, 0.32), (0.45, 0.38)); arrow(ax, (0.60, 0.32), (0.86, 0.62))
    arrow(ax, (0.84, 0.32), (0.13, 0.62)); arrow(ax, (0.25, 0.11), (0.17, 0.14)); arrow(ax, (0.40, 0.11), (0.51, 0.14)); arrow(ax, (0.74, 0.11), (0.84, 0.14))
    ax.text(0.5, 0.995, "Package map and dependency direction (arrows point at what a package imports; everything rests on ness.contracts)",
            ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(0.5, 0.968, "Rules: contracts import no backend  |  only backends.jax.* and backends.fabricpc.compat.* import JAX / FabricPC  |  "
                        "one bootstrap owner  |  plugins register through the `ness.plugins` entry point, core never names them",
            ha="center", va="top", fontsize=7.0, color="0.25")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def transaction_flow(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("1  request", "scenario yields a\nPredictionRequest:\nrole-tagged bundle,\ntask queries; withheld\noutcomes are NOT inside", "#eeeeee"),
        ("2  pin", "manifest id, memory views\n(snapshot cut at the origin),\ninference & learning\nprofiles are fixed for\nthis prediction", "#d0e0e3"),
        ("3  permitted view", "task access policy selects\nobserved + known_future\nfields; outcome_only and\noracle fields are removed", "#d9ead3"),
        ("4  execute", "GraphExecutor runs nodes in\ntopological order; typed\nevidence flows through ports;\nmerges lower to dense arrays\nand record provenance", "#fff2cc"),
        ("5  record", "immutable PredictionRecord:\nforecast + evidence graph +\ncosts + diagnostics\n(pre-outcome, target-free)", "#fce5cd"),
        ("6  reveal", "outcome arrives; task scores\nit; idempotent\nExperienceEvent; matured\nepisode queued for memory", "#ead1dc"),
        ("7  learn", "one optimizer step per batch\nunder the credit map (each\nrule owns groups in its\nruntime); then memory\npublishes a NEW snapshot", "#f4cccc"),
    ]
    n = len(steps)
    pitch = 0.985 / n
    w = pitch - 0.012
    for i, (t, b, c) in enumerate(steps):
        x = 0.008 + i * pitch
        box(ax, x, 0.30, w, 0.48, t, b, fc=c, fs=5.7, title_fs=8.0)
        if i < n - 1:
            arrow(ax, (x + w, 0.54), (x + pitch, 0.54))
    ax.text(0.5, 0.95, "The prediction transaction (PredictionTransaction.predict -> reveal -> learn_step)", ha="center", fontsize=10, fontweight="bold")
    ax.text(0.5, 0.88, "prequential mode: 1-7 repeat and the system changes only at step 7   |   frozen mode: 1-6 only, no parameter, count or memory update of any kind",
            ha="center", fontsize=7.4, color="0.25")
    ax.text(0.01, 0.22, "Identity: every scored prediction resolves to one PredictorManifest (content hash of every component, config and state).\n"
                        "Isolation: in-flight predictions keep their pinned memory views even after step 7 publishes a new snapshot.\n"
                        "Causality: the causal check re-predicts a request with its withheld fields perturbed and removed; the prediction must not change.",
            fontsize=7.2, va="top", color="0.2")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    package_map(OUT / "package_map.png")
    transaction_flow(OUT / "transaction_flow.png")
    print("\n".join(str(p) for p in sorted(OUT.glob("*.png"))))
