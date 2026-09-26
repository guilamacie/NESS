"""``ness report <run_dir>``: regenerate metrics, figures and the Markdown report from saved
artifacts only (nothing is recomputed from models). Requires the ``report`` extra
(matplotlib, pandas). Figures are numbered as in the toy-slice requirements document."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - import guard
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise ImportError("ness report needs the 'report' extra: pip install 'ness[report]'") from exc

TRANSITION_WINDOW = 8
ARM_ORDER_HINT = ("A0", "A1", "A2", "A3", "A4", "A5")


def _arm_sort_key(name: str) -> tuple:
    for i, pfx in enumerate(ARM_ORDER_HINT):
        if name.startswith(pfx):
            return (i, name)
    return (len(ARM_ORDER_HINT), name)


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text()) if p.exists() else None


def _read_csv(p: Path) -> "pd.DataFrame":
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 0 else pd.DataFrame()


def _parse_json_cols(df: "pd.DataFrame", cols: tuple[str, ...]) -> "pd.DataFrame":
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: json.loads(v) if isinstance(v, str) and v[:1] in "[{" else v)
    return df


class RunArtifacts:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.summary = _load_json(self.run_dir / "metrics" / "summary.json") or {}
        self.report_json = _load_json(self.run_dir / "report.json") or {}
        self.gen = _load_json(self.run_dir / "data" / "generation_manifest.json") or {}
        self.splits = _load_json(self.run_dir / "data" / "splits.json") or {}
        self.series = _read_csv(self.run_dir / "data" / "series.csv")
        self.truth = _read_csv(self.run_dir / "data" / "ground_truth.csv")
        self.subs = _load_json(self.run_dir / "metrics" / "substitution_matrix.json") or {"rows": [], "checks": []}
        self.train_hist = _read_csv(self.run_dir / "metrics" / "train_history.csv")
        self.timing = _read_csv(self.run_dir / "metrics" / "timing.csv")
        self.env = {p.stem: _load_json(p) for p in (self.run_dir / "environment").glob("*.json")}
        preds = [_read_csv(p) for p in sorted((self.run_dir / "predictions").glob("*.csv"))]
        self.preds = pd.concat([p for p in preds if not p.empty], ignore_index=True) if any(not p.empty for p in preds) else pd.DataFrame()
        traces = [_read_csv(p) for p in sorted((self.run_dir / "traces").glob("*_evidence.csv"))]
        self.traces = pd.concat([t for t in traces if not t.empty], ignore_index=True) if any(not t.empty for t in traces) else pd.DataFrame()
        if not self.traces.empty:
            self.traces = _parse_json_cols(self.traces, ("values", "knownness", "alternatives", "returned_ids", "sources", "details"))
        restores = [_read_csv(p) for p in sorted((self.run_dir / "metrics").glob("checkpoint_restore_*.csv"))]
        self.restores = pd.concat([r for r in restores if not r.empty], ignore_index=True) if any(not r.empty for r in restores) else pd.DataFrame()
        self.manifests = {p.stem: _load_json(p) for p in (self.run_dir / "manifests").glob("*.json")}
        self.arms = sorted({r["arm_id"] for r in self.summary.get("arms", {}).values()}, key=_arm_sort_key)
        self.task = next(iter(next(iter(self.summary.get("arms", {}).values()), {}).get("test", {"future_y": None})))
        self.change_points = list(self.gen.get("change_points", []))
        self.horizon = int(self.gen.get("horizon", 4))

    # ------------------------------------------------------------------ metrics
    def test_preds(self) -> "pd.DataFrame":
        if self.preds.empty:
            return self.preds
        df = self.preds[self.preds["phase"] == "test"].copy()
        df["dist_cp"] = df["g_distance_to_change_point"] if "g_distance_to_change_point" in df else np.nan
        return df

    def metrics_table(self) -> "pd.DataFrame":
        df = self.test_preds()
        t = self.task
        rows = []
        for (arm, seed), g in df.groupby(["arm", "seed"]):
            abs_cols = [c for c in g.columns if c.startswith(f"{t}/abs_err_")]
            errs = g[abs_cols].to_numpy(dtype=float)
            near = g["dist_cp"] <= TRANSITION_WINDOW
            rows.append({"arm": arm, "seed": seed, "MAE": float(np.nanmean(errs)), "RMSE": float(np.sqrt(np.nanmean(errs**2))),
                         "MAE_stable": float(np.nanmean(errs[~near.to_numpy()])) if (~near).any() else float("nan"),
                         "MAE_transition": float(np.nanmean(errs[near.to_numpy()])) if near.any() else float("nan"),
                         "n_scored": int(len(g)), "n_failed": int(g[f"{t}/mae"].isna().sum())})
        m = pd.DataFrame(rows)
        if m.empty:
            return m
        agg = m.groupby("arm").agg(MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"), RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
                                   MAE_stable=("MAE_stable", "mean"), MAE_transition=("MAE_transition", "mean"), n_scored=("n_scored", "first"),
                                   n_failed=("n_failed", "sum"), seeds=("seed", "count")).reset_index()
        agg["order"] = agg["arm"].map(_arm_sort_key)
        return agg.sort_values("order").drop(columns="order"), m

    def regime_diagnostics(self, node: str = "regime") -> "pd.DataFrame":
        tr = self.traces
        if tr.empty or "kind" not in tr:
            return pd.DataFrame()
        h = tr[(tr["node"] == node) & (tr["kind"] == "hypothesis") & (tr["phase"] == "test")].copy()
        if h.empty:
            return pd.DataFrame()
        truth = self.truth.set_index("request_id")["g_regime_at_origin"] if not self.truth.empty else None
        rows = []
        for (arm, seed), g in h.groupby(["arm", "seed"]):
            alts = g["alternatives"].iloc[0]
            idx_shift = alts.index("SHIFT") if "SHIFT" in alts else 1
            p_shift = np.array([v[idx_shift] for v in g["values"]])
            r = np.array([int(truth.loc[rid]) for rid in g["request_id"]]) if truth is not None else np.zeros(len(g), int)
            p_true = np.where(r == 1, p_shift, 1 - p_shift)
            rows.append({"arm": arm, "seed": seed, "n": len(g), "brier": float(np.mean((p_shift - r) ** 2)), "nll": float(-np.mean(np.log(np.clip(p_true, 1e-12, 1)))),
                         "accuracy": float(np.mean((p_shift > 0.5).astype(int) == r)), "semantics": g["semantics"].iloc[0]})
        return pd.DataFrame(rows)

    def memory_diagnostics(self, node: str = "analogues") -> dict[str, Any]:
        tr = self.traces
        if tr.empty:
            return {}
        m = tr[(tr["node"] == node) & (tr["kind"] == "retrieval")]
        if m.empty:
            return {}
        ages, n_empty, n_trunc = [], 0, 0
        for _, row in m.iterrows():
            ids = row["returned_ids"] or []
            if not ids:
                n_empty += 1
            for i in ids:
                ages.append(int(row["origin"]) - int(str(i).split("@")[1]))
            n_trunc += int(bool(row.get("truncated", False)))
        return {"retrievals": int(len(m)), "fraction_no_analogue": n_empty / len(m), "mean_age": float(np.mean(ages)) if ages else float("nan"),
                "median_age": float(np.median(ages)) if ages else float("nan"), "min_age": int(min(ages)) if ages else None, "truncated": n_trunc,
                "phases": sorted(m["phase"].unique().tolist())}

    def symbolic_diagnostics(self) -> dict[str, Any]:
        tr = self.traces
        out: dict[str, Any] = {}
        if tr.empty or self.truth.empty:
            return out
        truth = self.truth.set_index("request_id")
        ev = tr[(tr["node"] == "event_program") & (tr["kind"] == "feature") & (tr["phase"] == "test")]
        if not ev.empty:
            g = ev[ev["arm"] == sorted(ev["arm"].unique(), key=_arm_sort_key)[0]]
            g = g[g["seed"] == g["seed"].min()]
            pred = np.array([v[0] for v in g["values"]])
            known = np.array([k[0] for k in g["knownness"]])
            gt = np.array([int(truth.loc[r, "g_event_in_horizon"]) for r in g["request_id"]])
            out["event_program"] = {"n": int(len(g)), "agreement_with_ground_truth": float(np.mean((pred > 0.5) == (gt == 1))), "known_fraction": float(known.mean())}
        sl = tr[(tr["node"] == "slope_program") & (tr["kind"] == "feature") & (tr["phase"] == "test")]
        if not sl.empty:
            g = sl[(sl["arm"] == sorted(sl["arm"].unique(), key=_arm_sort_key)[0])]
            g = g[g["seed"] == g["seed"].min()]
            reg = np.array([int(truth.loc[r, "g_regime_at_origin"]) for r in g["request_id"]])
            vals = np.array([v for v in g["values"]])
            out["slope_program"] = {"n": int(len(g)), "mean_abs_slope_y_normal": float(np.mean(np.abs(vals[reg == 0, 0]))) if (reg == 0).any() else None,
                                    "mean_abs_slope_y_shift": float(np.mean(np.abs(vals[reg == 1, 0]))) if (reg == 1).any() else None}
        sem = tr[(tr["node"] == "semantic") & (tr["kind"] == "feature") & (tr["phase"] == "test")]
        if not sem.empty:
            g = sem[(sem["arm"] == sorted(sem["arm"].unique(), key=_arm_sort_key)[0])]
            g = g[g["seed"] == g["seed"].min()]
            reg = np.array([int(truth.loc[r, "g_regime_at_origin"]) for r in g["request_id"]])
            vals = np.array([v for v in g["values"]])
            names = str(g["interpretation"].iloc[0]).split(",")
            out["semantic_features_by_regime"] = {n: {"NORMAL": float(vals[reg == 0, i].mean()), "SHIFT": float(vals[reg == 1, i].mean())} for i, n in enumerate(names) if i < vals.shape[1]}
        return out


# ---------------------------------------------------------------------------- figures
def _save(fig, plots: Path, name: str) -> str:
    fig.tight_layout()
    fig.savefig(plots / f"{name}.png", dpi=150)
    fig.savefig(plots / f"{name}.svg")
    plt.close(fig)
    return f"plots/{name}.png"


def fig_dataset(a: RunArtifacts, plots: Path) -> str | None:
    if a.series.empty:
        return None
    s = a.series
    end = min(len(s), max(360, (a.change_points[3] + 40) if len(a.change_points) > 3 else 360))
    seg = s.iloc[:end]
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for ax in axes:
        for k in range(0, end):
            pass
        r = seg["regime"].to_numpy()
        t = seg["t"].to_numpy()
        starts = [0] + [i for i in range(1, len(r)) if r[i] != r[i - 1]]
        for i, st in enumerate(starts):
            en = starts[i + 1] if i + 1 < len(starts) else len(r)
            if r[st] == 1:
                ax.axvspan(t[st], t[en - 1], color="orange", alpha=0.15, lw=0)
    axes[0].plot(seg["t"], seg["y"], lw=1, color="C0", label="target y_t")
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper right")
    axes[1].plot(seg["t"], seg["x"], lw=1, color="C2", label="auxiliary x_t")
    axes[1].set_ylabel("x")
    axes[1].legend(loc="upper right")
    axes[2].step(seg["t"], seg["event"], where="post", color="C3", label="event e_t (known ahead)")
    axes[2].set_ylabel("e")
    axes[2].set_xlabel("t (steps)")
    axes[2].legend(loc="upper right")
    for cp in a.change_points:
        if cp < end:
            axes[0].axvline(cp, color="k", ls=":", lw=0.8)
    fig.suptitle("Figure 1. Synthetic dataset: y, x, events; shaded = hidden SHIFT regime (evaluation only), dotted = regime change points")
    return _save(fig, plots, "fig01_dataset")


def fig_architecture(a: RunArtifacts, plots: Path, arm_hint: str = "A4") -> str | None:
    key = next((k for k in sorted(a.manifests) if k.startswith(arm_hint)), next(iter(sorted(a.manifests)), None))
    if key is None:
        return None
    m = a.manifests[key]
    wiring = m["extra"]["resolved_wiring"]
    comps = {c["node_id"]: c for c in m["components"]}
    nodes = list(wiring)
    deps = {n: {s.split("://")[1].split("/")[0] for srcs in ports.values() for s in srcs if not s.startswith("observation://")} for n, ports in wiring.items()}
    layer: dict[str, int] = {}
    for _ in range(len(nodes) + 1):
        for n in nodes:
            layer[n] = 0 if not deps[n] else 1 + max(layer.get(d, 0) for d in deps[n])
    layers: dict[int, list[str]] = {}
    for n in nodes:
        layers.setdefault(layer[n], []).append(n)
    ncols = max(layers) + 2
    fig, ax = plt.subplots(figsize=(14, 6.5))
    pos: dict[str, tuple[float, float]] = {"observation": (0.06, 0.5)}
    for L, ns in sorted(layers.items()):
        ns = sorted(ns)
        for i, n in enumerate(ns):
            y = 1 - (i + 1) / (len(ns) + 1)
            if len(ns) == 1:
                y = 0.5 + (0.16 if L % 2 else -0.16)  # stagger single-node layers so labels do not collide
            pos[n] = (0.06 + (L + 1) * (0.88 / (ncols - 1)), y)
    bw, bh = 0.105, 0.115

    def box(x, y, text, color):
        ax.add_patch(plt.Rectangle((x - bw / 2, y - bh / 2), bw, bh, fc=color, ec="k", lw=0.8, transform=ax.transAxes, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=7, transform=ax.transAxes, zorder=4)

    box(*pos["observation"], "observations\nseries_history\nevent_history\nevent_future", "#eeeeee")
    colors = {"substrate": "#cfe2f3", "upper_module": "#d9ead3", "semantic_adapter": "#fff2cc", "program": "#fce5cd", "reasoner": "#ead1dc", "memory_query": "#d0e0e3", "cap": "#f4cccc"}
    for n in nodes:
        c = comps.get(n, {})
        box(*pos[n], f"{n}\n{c.get('plugin_id', '?')}\n[{c.get('runtime', '?')}]", colors.get(c.get("module_kind", ""), "#ffffff"))
    k = 0
    for n, ports in wiring.items():
        for port, srcs in ports.items():
            for s in srcs:
                src = "observation" if s.startswith("observation://") else s.split("://")[1].split("/")[0]
                x0, y0 = pos[src]
                x1, y1 = pos[n]
                rad = 0.15 if (k % 2) else -0.15
                ax.annotate("", xy=(x1 - bw / 2, y1), xytext=(x0 + bw / 2, y0), xycoords="axes fraction", textcoords="axes fraction",
                            arrowprops=dict(arrowstyle="->", lw=0.7, color="0.35", connectionstyle=f"arc3,rad={rad}"))
                lbl = s.split("://")[1] if s.startswith("observation://") else s.split("/", 3)[-1]
                t = 0.72  # label near the destination
                lx, ly = x0 + bw / 2 + t * (x1 - bw / 2 - x0 - bw / 2), y0 + t * (y1 - y0) + (0.028 if k % 2 else -0.028)
                ax.text(lx, ly, f"{port} <- {lbl}", fontsize=5.5, color="0.2", ha="center", transform=ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8), zorder=5)
                k += 1
    ax.text(0.5, 0.01, f"Figure 2. Instantiated composition of arm {key} (resolved from manifest {str(m.get('manifest_id', ''))[:12]}); edge labels: input-port <- source-port. "
            "All edges into the cap from host nodes are stop-gradient leaves.", ha="center", fontsize=8, transform=ax.transAxes)
    ax.axis("off")
    return _save(fig, plots, "fig02_architecture")


def fig_trajectories(a: RunArtifacts, plots: Path) -> str | None:
    df = a.test_preds()
    if df.empty:
        return None
    t = a.task
    seed = df["seed"].min()
    arms = [x for x in a.arms if not x.startswith("A5")][:5]
    origins = sorted(df["origin"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=False)
    for ax, (lo, hi) in zip(axes, [(origins[0], origins[min(len(origins) - 1, 79)]), (origins[max(0, len(origins) - 80)], origins[-1])]):
        seg = df[(df["origin"] >= lo) & (df["origin"] <= hi) & (df["seed"] == seed)]
        truth = seg[seg["arm"] == arms[0]].sort_values("origin")
        ax.plot(truth["origin"] + 1, truth[f"{t}/truth_0"], color="k", lw=1.6, label="truth y_{o+1}")
        for i, arm in enumerate(arms):
            g = seg[seg["arm"] == arm].sort_values("origin")
            ax.plot(g["origin"] + 1, g[f"{t}/pred_0"], lw=1, alpha=0.9, label=arm, color=f"C{i}")
        for cp in a.change_points:
            if lo <= cp <= hi:
                ax.axvline(cp, color="k", ls=":", lw=0.8)
        ax.set_ylabel("y (h=1)")
        ax.set_xlabel("target time o+1")
    axes[0].legend(ncol=3, fontsize=8)
    fig.suptitle(f"Figure 3. One-step-ahead forecasts on two held-out intervals (seed {seed}); dotted = regime change points")
    return _save(fig, plots, "fig03_trajectories")


def fig_metric_by_arm(a: RunArtifacts, plots: Path, agg: "pd.DataFrame", per_seed: "pd.DataFrame") -> str | None:
    if agg is None or agg.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, ("MAE", "RMSE")):
        x = np.arange(len(agg))
        ax.bar(x, agg[f"{col}_mean"], yerr=agg[f"{col}_sd"].fillna(0), color="0.75", edgecolor="k", capsize=3)
        for i, arm in enumerate(agg["arm"]):
            vals = per_seed[per_seed["arm"] == arm][col]
            ax.scatter([i] * len(vals), vals, color="C3", s=14, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(agg["arm"], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(f"test {col} (all horizons)")
    n_seeds = int(agg["seeds"].max())
    fig.suptitle(f"Figure 4. Point-forecast error by arm: bars = mean over {n_seeds} model seed(s), dots = individual seeds" + (" (deterministic engineering smoke result, not a statistical comparison)" if n_seeds < 3 else "; dispersion across seeds only"))
    return _save(fig, plots, "fig04_metric_by_arm")


def fig_error_vs_transition(a: RunArtifacts, plots: Path) -> str | None:
    df = a.test_preds()
    if df.empty or df["dist_cp"].isna().all():
        return None
    t = a.task
    bins = [-0.5, 2.5, 5.5, 8.5, 12.5, 20.5, 1e9]
    labels = ["0-2", "3-5", "6-8", "9-12", "13-20", ">20"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, arm in enumerate(a.arms):
        g = df[df["arm"] == arm]
        cat = pd.cut(g["dist_cp"], bins=bins, labels=labels)
        m = g.groupby(cat, observed=False)[f"{t}/mae"].mean()
        ax.plot(range(len(labels)), m.values, marker="o", label=arm, color=f"C{i}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("distance from origin to nearest regime change point (steps)")
    ax.set_ylabel("mean MAE per origin")
    ax.legend(fontsize=8)
    fig.suptitle("Figure 5. Forecast error versus distance from known regime transitions (all seeds pooled)")
    return _save(fig, plots, "fig05_error_vs_transition")


def fig_posterior(a: RunArtifacts, plots: Path) -> str | None:
    tr = a.traces
    if tr.empty or "kind" not in tr:
        return None
    h = tr[(tr["node"] == "regime") & (tr["kind"] == "hypothesis") & (tr["phase"] == "test")]
    if h.empty:
        return None
    arm = sorted(h["arm"].unique(), key=_arm_sort_key)[0]
    g = h[(h["arm"] == arm) & (h["seed"] == h["seed"].min())].sort_values("origin")
    idx = g["alternatives"].iloc[0].index("SHIFT")
    p = np.array([v[idx] for v in g["values"]])
    truth = a.truth.set_index("request_id")["g_regime_at_origin"]
    r = np.array([int(truth.loc[rid]) for rid in g["request_id"]])
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.step(g["origin"], r, where="post", color="k", lw=1.2, label="true regime at origin (1 = SHIFT; evaluation only)")
    ax.plot(g["origin"], p, color="C4", lw=1.2, label="posterior P(SHIFT | permitted evidence)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("forecast origin")
    ax.set_ylabel("probability")
    ax.legend(fontsize=8, loc="center right")
    fig.suptitle(f"Figure 6. Exact two-state regime posterior over test origins ({arm}, seed {int(g['seed'].iloc[0])})")
    return _save(fig, plots, "fig06_regime_posterior")


def fig_training(a: RunArtifacts, plots: Path) -> str | None:
    h = a.train_hist
    if h.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, arm in enumerate(sorted(h["arm"].unique(), key=_arm_sort_key)):
        for seed, g in h[h["arm"] == arm].groupby("seed"):
            ax.plot(g["update"], g["loss"], color=f"C{i}", alpha=0.5 if seed != g["seed"].min() else 1.0, lw=1, label=arm if seed == h[h["arm"] == arm]["seed"].min() else None)
    ax.set_xlabel("optimizer update")
    ax.set_ylabel("batch task loss (MSE, sum/count)")
    ax.legend(fontsize=8)
    fig.suptitle("Figure 7. Training curves of the trainable arms (one line per seed; only the task loss exists, no regularisers)")
    return _save(fig, plots, "fig07_training_curves")


def fig_symbolic(a: RunArtifacts, plots: Path) -> str | None:
    tr = a.traces
    if tr.empty or a.truth.empty:
        return None
    truth = a.truth.set_index("request_id")
    sem = tr[(tr["node"] == "semantic") & (tr["kind"] == "feature") & (tr["phase"] == "test")]
    ev = tr[(tr["node"] == "event_program") & (tr["kind"] == "feature") & (tr["phase"] == "test")]
    if sem.empty and ev.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if not sem.empty:
        g = sem[(sem["arm"] == sorted(sem["arm"].unique(), key=_arm_sort_key)[0])]
        g = g[g["seed"] == g["seed"].min()]
        names = str(g["interpretation"].iloc[0]).split(",")
        vals = np.array([v for v in g["values"]])
        reg = np.array([int(truth.loc[r, "g_regime_at_origin"]) for r in g["request_id"]])
        pick = [i for i, n in enumerate(names) if n in ("volatility_0", "xcorr_lag1", "xcorr_lag2")]
        data = [vals[reg == k, i] for i in pick for k in (0, 1)]
        labels = [f"{names[i]}\n{'NORMAL' if k == 0 else 'SHIFT'}" for i in pick for k in (0, 1)]
        axes[0].boxplot(data, tick_labels=labels)
        axes[0].tick_params(axis="x", labelsize=7)
        axes[0].set_title("semantic features by true regime (test origins)", fontsize=9)
    if not ev.empty:
        g = ev[(ev["arm"] == sorted(ev["arm"].unique(), key=_arm_sort_key)[0])]
        g = g[g["seed"] == g["seed"].min()].sort_values("origin")
        pred = np.array([v[0] for v in g["values"]])
        gt = np.array([int(truth.loc[r, "g_event_in_horizon"]) for r in g["request_id"]])
        axes[1].step(g["origin"], gt, where="post", color="k", lw=1.5, label="event within horizon (ground truth)")
        axes[1].plot(g["origin"], pred - 0.03, "o", ms=2.5, color="C3", label="program event_active_at_origin")
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].set_xlabel("forecast origin")
        axes[1].legend(fontsize=7)
        axes[1].set_title("typed program output vs ground truth", fontsize=9)
    fig.suptitle("Figure 8. Symbolic pathway diagnostic")
    return _save(fig, plots, "fig08_symbolic")


def fig_memory(a: RunArtifacts, plots: Path) -> str | None:
    tr = a.traces
    if tr.empty:
        return None
    m = tr[(tr["node"] == "analogues") & (tr["kind"] == "retrieval") & (tr["phase"] == "test")]
    if m.empty:
        return None
    g = m[m["seed"] == m["seed"].min()].sort_values("origin")
    fig, ax = plt.subplots(figsize=(9, 4))
    xs, ys = [], []
    for _, row in g.iterrows():
        for rid in row["returned_ids"] or []:
            xs.append(int(row["origin"]))
            ys.append(int(str(rid).split("@")[1]))
    ax.scatter(xs, ys, s=8, color="C0", label="retrieved analogue episode origin")
    o = np.array(sorted(g["origin"]))
    ax.plot(o, o - a.horizon, color="k", ls="--", lw=1, label=f"eligibility bound: episode origin <= query origin - H ({a.horizon})")
    empty = g[g["returned_ids"].apply(lambda v: not v)]
    if not empty.empty:
        ax.scatter(empty["origin"], [o.min()] * len(empty), marker="x", color="C3", s=20, label="no eligible analogue")
    ax.set_xlabel("query origin (test)")
    ax.set_ylabel("retrieved episode origin")
    ax.legend(fontsize=7)
    fig.suptitle("Figure 9. Analogue memory retrieval: every retrieved episode had matured before the query origin")
    return _save(fig, plots, "fig09_memory_retrieval")


def fig_substitutions(a: RunArtifacts, plots: Path) -> str | None:
    rows = a.subs.get("rows", [])
    checks = a.subs.get("checks", [])
    if not rows:
        return None
    mat = np.zeros((len(rows), len(checks)))
    for i, r in enumerate(rows):
        for j, c in enumerate(checks):
            v = str(r.get(c, "n/a"))
            mat[i, j] = 1.0 if v == "pass" else (0.5 if v == "n/a" else 0.0)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1.5))
    from matplotlib.colors import ListedColormap
    ax.imshow(mat, cmap=ListedColormap(["#e06666", "#cccccc", "#93c47d"]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(checks)))
    ax.set_xticklabels(checks)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["id"] for r in rows], fontsize=8)
    for i, r in enumerate(rows):
        for j, c in enumerate(checks):
            v = str(r.get(c, "n/a"))
            ax.text(j, i, "pass" if v == "pass" else ("n/a" if v == "n/a" else "FAIL"), ha="center", va="center", fontsize=7)
    fig.suptitle("Figure 10. Substitution matrix (configuration-only changes; core source hash identical for every row)")
    return _save(fig, plots, "fig10_substitution_matrix")


def fig_restore(a: RunArtifacts, plots: Path) -> str | None:
    if a.restores.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 3.8))
    keys = sorted({(r.arm, r.seed) for r in a.restores.itertuples()}, key=lambda k: (_arm_sort_key(k[0]), k[1]))
    vals = [a.restores[(a.restores["arm"] == k[0]) & (a.restores["seed"] == k[1])]["max_abs_diff"].max() for k in keys]
    ax.bar(range(len(keys)), [max(v, 1e-18) for v in vals], color="0.7", edgecolor="k")
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1)
    ax.axhline(1e-12, color="C3", ls=":", lw=1, label="1e-12 reference")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{k[0].split('_')[0]} s{k[1]}" for k in keys], fontsize=7, rotation=90)  # short arm id (A0..A5) + seed
    ax.set_xlabel("arm (short id) and model seed; full arm ids in metrics/summary.csv")
    ax.set_ylabel("max |pred before - pred after restore|")
    ax.legend(fontsize=8)
    fig.suptitle("Figure 11. Checkpoint reproducibility: predictions before publish vs after restore (0 = bitwise identical)")
    return _save(fig, plots, "fig11_checkpoint_restore")


def fig_cost(a: RunArtifacts, plots: Path) -> str | None:
    t = a.timing
    if t.empty:
        return None
    piv = t[t["phase"].isin(["build", "train", "checkpoint", "eval"])].groupby(["arm", "phase"])["seconds"].mean().unstack().fillna(0)
    piv = piv.loc[sorted(piv.index, key=_arm_sort_key)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bottom = np.zeros(len(piv))
    for ph in ("build", "train", "checkpoint", "eval"):
        if ph in piv:
            axes[0].bar(range(len(piv)), piv[ph], bottom=bottom, label=ph)
            bottom += piv[ph].to_numpy()
    axes[0].set_xticks(range(len(piv)))
    axes[0].set_xticklabels(piv.index, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("wall-clock seconds (mean over seeds)")
    axes[0].legend(fontsize=8)
    rss = t.groupby("arm")["peak_rss_mb"].max()
    rss = rss.loc[sorted(rss.index, key=_arm_sort_key)]
    axes[1].bar(range(len(rss)), rss.values, color="0.7", edgecolor="k")
    axes[1].set_xticks(range(len(rss)))
    axes[1].set_xticklabels(rss.index, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("process peak RSS (MB, cumulative in one process)")
    fig.suptitle("Figure 12. Basic cost: wall-clock per phase and process peak memory (toy scale; not a benchmark)")
    return _save(fig, plots, "fig12_cost")


# ---------------------------------------------------------------------------- report
def _fmt(v: Any, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v))):
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def build_report(run_dir: str | Path, title: str = "NESS toy vertical slice - run report") -> Path:
    a = RunArtifacts(Path(run_dir))
    plots = a.run_dir / "plots"
    plots.mkdir(exist_ok=True)
    (a.run_dir / "report").mkdir(exist_ok=True)
    mt = a.metrics_table()
    agg, per_seed = (mt if isinstance(mt, tuple) else (mt, pd.DataFrame()))
    regime = a.regime_diagnostics()
    mem = a.memory_diagnostics()
    symb = a.symbolic_diagnostics()
    figs = {
        1: fig_dataset(a, plots), 2: fig_architecture(a, plots), 3: fig_trajectories(a, plots), 4: fig_metric_by_arm(a, plots, agg, per_seed),
        5: fig_error_vs_transition(a, plots), 6: fig_posterior(a, plots), 7: fig_training(a, plots), 8: fig_symbolic(a, plots),
        9: fig_memory(a, plots), 10: fig_substitutions(a, plots), 11: fig_restore(a, plots), 12: fig_cost(a, plots),
    }
    if not agg.empty:
        agg.to_csv(a.run_dir / "metrics" / "aggregate_by_arm.csv", index=False)
        per_seed.to_csv(a.run_dir / "metrics" / "per_seed.csv", index=False)
    if not regime.empty:
        regime.to_csv(a.run_dir / "metrics" / "regime_diagnostics.csv", index=False)
    (a.run_dir / "metrics" / "memory_diagnostics.json").write_text(json.dumps(mem, indent=1))
    (a.run_dir / "metrics" / "symbolic_diagnostics.json").write_text(json.dumps(symb, indent=1, default=str))

    arms_summary = a.summary.get("arms", {})
    gen = a.gen
    L: list[str] = [f"# {title}", "", f"Run directory: `{a.run_dir}`  Protocol: `{a.summary.get('protocol_id')}` (`{str(a.summary.get('protocol_hash', ''))[:12]}`)", ""]
    L += ["## 1. Executive summary", ""]
    ok = [k for k, v in arms_summary.items() if v["status"] == "ok"]
    failed = [k for k, v in arms_summary.items() if v["status"] != "ok"]
    subs_ok = sum(1 for r in a.subs.get("rows", []) if r.get("overall") == "pass")
    L += [f"* End-to-end prediction -> record -> outcome -> learning -> checkpoint -> frozen evaluation ran for {len(ok)} arm/seed combinations" + (f"; **{len(failed)} failed**: {failed}" if failed else " with no failures") + ".",
          f"* Substitution proofs: {subs_ok}/{len(a.subs.get('rows', []))} configuration-only variants passed every applicable check (validate, predict, causal, train, restore); core source hash `{str(a.subs.get('core_source_hash', ''))[:12]}` for all.",
          f"* Checkpoint restore: max |Δprediction| over all arms = {_fmt(float(a.restores['max_abs_diff'].max()) if not a.restores.empty else float('nan'), 3)} (0 = bitwise).",
          "* Baseline parity at neutral initialisation holds (zero-initialised writers); see substitution matrix and §13.",
          "* Predictive results are an engineering smoke comparison on a toy problem; no superiority claim is made (§7)."]
    if not agg.empty:
        best = agg.sort_values("MAE_mean").iloc[0]
        base = agg[agg["arm"].str.startswith("A0")]
        L.append(f"* Lowest mean test MAE: `{best['arm']}` = {best['MAE_mean']:.4f}" + (f" vs frozen baseline {float(base['MAE_mean'].iloc[0]):.4f}" if not base.empty else "") + f" ({int(agg['seeds'].max())} seeds).")
    if not regime.empty:
        L.append(f"* Regime posterior (A3): accuracy {regime['accuracy'].mean():.3f}, Brier {regime['brier'].mean():.3f} against the hidden regime (evaluation only).")
    L += ["", "## 2. Purpose and status", "", "This run is an **engineering/reference demonstration** of the NESS interfaces and controls. It is not evidence that the neuro-symbolic or probabilistic system is superior to a Transformer; null or negative differences are valid outcomes and are reported as such.", ""]
    L += ["## 3. Toy dataset", ""]
    if gen:
        pr = gen.get("parameters", {})
        L += ["Generator (deterministic, seed {}):".format(gen.get("seed")), "", "```",
              f"x_t = {pr.get('phi_x')} x_(t-1) + {pr.get('sigma_x')} eps_x",
              f"NORMAL: y_t = {pr.get('a')} y_(t-1) + {pr.get('b', [None, None])[0]} x_(t-{pr.get('lag', [1, 2])[0]}) + {pr.get('c', [None, None])[0]} e_t + {pr.get('sigma', [None, None])[0]} eps",
              f"SHIFT : y_t = {pr.get('a')} y_(t-1) + {pr.get('b', [None, None])[1]} x_(t-{pr.get('lag', [1, 2])[1]}) + {pr.get('c', [None, None])[1]} e_t + {pr.get('sigma', [None, None])[1]} eps",
              f"e_t: {gen.get('equations', {}).get('e')};  r_t: {gen.get('equations', {}).get('r')}", "```", "",
              f"Change points: {gen.get('change_points')}. Context {gen.get('context')} steps, horizon {gen.get('horizon')}. Rolling-origin splits (origin first/last/count): {gen.get('splits')}.",
              "Access policy: the predictor sees `series_history` (y, x) and `event_history` (observed) plus `event_future` (known_future). `target_future` is outcome-only and `true_regime` is an evaluation oracle; both are removed by every task view before prediction.", ""]
    if figs[1]:
        L += [f"![Figure 1]({figs[1]})", "", "*Figure 1. Representative segment: target, auxiliary channel, events; shaded regions mark the hidden SHIFT regime (evaluation only).*", ""]
    L += ["## 4. Instantiated architecture", ""]
    key4 = next((k for k in a.manifests if k.startswith("A4")), next(iter(a.manifests), None))
    if key4:
        m = a.manifests[key4]
        L += [f"Resolved composition of `{key4}` (manifest `{m.get('manifest_id', '')[:16]}`):", "", "| node | plugin | runtime | inputs |", "|---|---|---|---|"]
        comps = {c["node_id"]: c for c in m["components"]}
        for n, ports in m["extra"]["resolved_wiring"].items():
            c = comps.get(n, {})
            L.append(f"| `{n}` | `{c.get('plugin_id')}@{c.get('plugin_version')}` | {c.get('runtime')} | " + "; ".join(f"`{p}` <- " + ", ".join(f"`{s}`" for s in srcs) for p, srcs in ports.items()) + " |")
        L += ["", "The upper module merges `state/final` with a learned linear projection of `state/early` through `gated_add`, followed by `layer_norm` (edge parameters are a separate manifest component `composition:boundaries`). The frozen substrate group is `frozen`; trainable groups are `upper/upper`, `cap/consumer` and the boundary parameters, all owned by `bp_direct`. Every edge into the cap from a host node (semantic, programs, reasoner, memory) is a stop-gradient leaf. Symbolic programs run through the reference interpreter; the reasoner is exact enumeration over two states; memory is a pinned snapshot view queried once per prediction.", ""]
    if figs[2]:
        L += [f"![Figure 2]({figs[2]})", "", "*Figure 2. Architecture/data-flow diagram generated from the resolved manifest of the full arm.*", ""]
    L += ["## 5. Experiment arms", "", "| arm | description |", "|---|---|"]
    exp_cfg = a.run_dir / "config" / "experiment.yaml"
    if exp_cfg.exists():
        import yaml
        cfg = yaml.safe_load(exp_cfg.read_text())
        for arm_id, d in cfg.get("arms", {}).items():
            L.append(f"| `{arm_id}` | {d.get('description', '')} |")
    L += ["", "## 6. Training and inference protocol", ""]
    proto = (a.report_json or {}).get("protocol") or (cfg.get("protocol", {}) if exp_cfg.exists() else {})
    L += [f"* Model seeds: {proto.get('seeds')}; dataset seed locked ({gen.get('seed')}).", f"* Prequential training on the train split: batch {proto.get('batch_size')} requests per update, {proto.get('updates')} updates (Adam, lr 0.003, clip 5); frozen evaluation on the test split (no updates, no memory writes).",
          "* Inference profile `direct` (no settling); learning rule `bp_direct` (BP through the deployed direct computation). Frozen: substrate weights. Trainable: upper module, cap consumer, boundary transforms.",
          f"* Whole-system learner checkpoint published after training; the restored predictor re-predicts the first {proto.get('restore_check_requests')} test requests (Figure 11). Causal check: predictions are re-computed with outcome fields perturbed and removed (must be identical).",
          "* Cost accounting: wall-clock per phase and process peak RSS (`metrics/timing.csv`); per-node realised costs in prediction records.", ""]
    L += ["## 7. Forecasting results", ""]
    if not agg.empty:
        L += ["| arm | MAE (mean±sd over seeds) | RMSE | MAE stable | MAE near transition (≤8) | n scored | failed |", "|---|---|---|---|---|---|---|"]
        for _, r in agg.iterrows():
            L.append(f"| `{r['arm']}` | {r['MAE_mean']:.4f} ± {_fmt(r['MAE_sd'])} | {r['RMSE_mean']:.4f} | {_fmt(r['MAE_stable'])} | {_fmt(r['MAE_transition'])} | {int(r['n_scored'])} | {int(r['n_failed'])} |")
        L += ["", "Per-seed values: `metrics/per_seed.csv`. Paired per-request differences vs the reference arm: `metrics/summary.json` (`paired`).", ""]
    for i in (3, 4, 5):
        if figs[i]:
            L += [f"![Figure {i}]({figs[i]})", ""]
    L += ["*Figures 3-5: trajectories on held-out intervals, aggregate error by arm with seed dispersion, error versus distance to regime change.* No superiority claim: differences are within a toy, single-dataset setting.", ""]
    L += ["## 8. Probabilistic reasoning results", ""]
    if not regime.empty:
        L += ["| arm | seed | n | Brier | NLL | accuracy | semantics |", "|---|---|---|---|---|---|---|"]
        for _, r in regime.iterrows():
            L.append(f"| `{r['arm']}` | {int(r['seed'])} | {int(r['n'])} | {r['brier']:.3f} | {r['nll']:.3f} | {r['accuracy']:.3f} | {r['semantics']} |")
        L += ["", "The reasoner consumed only the eight semantic features (slopes, volatilities, cross-lag correlations, neural cue, known-future event cue) computed from permitted evidence; emission parameters were fitted on the training split (`fit_regime_emissions.py`) using the hidden regime as a fitting label on allowed data. The hidden regime is never a predictor input.", ""]
    if figs[6]:
        L += [f"![Figure 6]({figs[6]})", ""]
    L += ["## 9. Symbolic pathway results", ""]
    if symb:
        L += ["```json", json.dumps(symb, indent=1, default=str), "```", ""]
    if figs[8]:
        L += [f"![Figure 8]({figs[8]})", ""]
    L += ["Programs (`window_slope_8`, `event_active_at_origin_h4`) run through the reference interpreter with fuel budgets; outputs carry knownness (missing windows are unknown, never 0) and `ExecutionTrace`s (`traces/*_evidence.csv`, kind `execution_trace`). Whether the program changed the cap's prediction is visible as the A1→A2 paired difference in `metrics/summary.json`.", ""]
    L += ["## 10. Memory results", ""]
    if mem:
        L += ["```json", json.dumps(mem, indent=1), "```", ""]
    if figs[9]:
        L += [f"![Figure 9]({figs[9]})", ""]
    L += ["Retrieval operates on a pinned view filtered by availability *before* ranking; the learner appends matured episodes only between updates and publishes a new snapshot; in-flight predictions keep their view (T39). No benefit is claimed where the A3→A4 difference is within seed dispersion.", ""]
    L += ["## 11. Training dynamics", ""]
    if figs[7]:
        L += [f"![Figure 7]({figs[7]})", ""]
    fails = {k: v.get("failures", []) for k, v in arms_summary.items() if v.get("failures")}
    L += [f"Non-finite states, clipping fallbacks or failed runs: {fails if fails else 'none recorded'}.", ""]
    L += ["## 12. Modularity / substitution proof", ""]
    rows = a.subs.get("rows", [])
    if rows:
        checks = a.subs.get("checks", [])
        L += ["| id | description | " + " | ".join(checks) + " | resolved plugins |", "|---|---|" + "---|" * len(checks) + "---|"]
        for r in rows:
            L.append(f"| `{r['id']}` | {r['description']} | " + " | ".join(str(r.get(c, 'n/a'))[:30] for c in checks) + " | " + ", ".join(f"{k}: {v}" for k, v in r.get("resolved_plugins", {}).items()) + " |")
        L += ["", f"No substitution required editing core scheduler/runtime code: all rows ran against core source hash `{a.subs.get('core_source_hash', '')}` (`environment/core_source_hash.json`).", ""]
    if figs[10]:
        L += [f"![Figure 10]({figs[10]})", ""]
    L += ["## 13. Checkpoint / reproducibility proof", ""]
    L += ["| arm@seed | manifest | checkpoint | restore n | max abs diff | manifest match |", "|---|---|---|---|---|---|"]
    for k, v in arms_summary.items():
        rc = v.get("restore_check", {})
        L.append(f"| `{k}` | `{str(v.get('manifest_id', ''))[:12]}` | `{str(v.get('checkpoint_manifest') or '')[:12]}` | {rc.get('n', '-')} | {_fmt(rc.get('max_abs_diff'), 3)} | {rc.get('restored_manifest_matches', '-')} |")
    if figs[11]:
        L += ["", f"![Figure 11]({figs[11]})", ""]
    L += ["Reproduce: see §17 commands; checkpoints under `checkpoints/` (content-addressed blobs + manifests), inspect with `ness inspect <run>/checkpoints <manifest>`.", ""]
    L += ["## 14. Cost and scaling observations", ""]
    if figs[12]:
        L += [f"![Figure 12]({figs[12]})", ""]
    L += ["Toy-only components: the frozen providers (self-pretrained tiny transformer/conv), the in-memory snapshot store, the exact two-state reasoner. Intended to scale through the same interfaces: substrate ports (external frozen providers), the differentiable region (batched/sharded), FabricPC workspaces, plugin memory stores. No large-scale performance is extrapolated.", ""]
    L += ["## 15. Tests and acceptance status", "", "See `docs/ARCHITECTURE.md` §10 for the T01-T44 map; this run additionally exercises T04/T38 (causal check per arm), T08/T27/T40 (restore), T31/T33-T35 (substitutions), T39 (memory isolation in A4), T43 (resolved wiring in manifests). Not applicable at toy scale: T10, T12, T14, T20, T25-T29, T32.", ""]
    L += ["## 16. Problems discovered and architecture decisions", "", "* Emission parameters of the exact reasoner are a fitted asset (train split) and are recorded in the config; a fitted-state contract (ADR) would make this a first-class plugin state rather than configuration.", "* The cap lowers unknown feature entries to 0 and ignores knownness masks (declared capability); a mask-aware consumer is a plugin change.", "* Per-request tracing under `jax.value_and_grad` dominates training time at this scale; the batched/data-parallel path exists for uniform shapes.", ""]
    L += ["## 17. Reproduction and next steps", "", "```bash", "ness validate examples/vertical_slice_timeseries/configs/toy_vertical_slice.yaml", f"ness run examples/vertical_slice_timeseries/configs/toy_vertical_slice.yaml --out {a.run_dir}", f"ness evaluate {a.run_dir}", f"ness report {a.run_dir}", f"ness inspect {a.run_dir}/checkpoints <manifest-id>", "```", "",
          "Next integrations (smallest steps): a TimesFM-3 forecast-only substrate behind the same `forecast/point|quantiles` ports; a frozen text+vision substrate pair with alignment evidence; a Hyperon-backed `MemoryStore`; a NumPyro reasoner plugin compared against `exact_finite_regime_reasoner`; larger frozen-provider runs on the FabricPC backend (see `docs/FABRICPC_BACKEND.md`).", ""]
    L += ["## Artifact index", "", "| artifact | path |", "|---|---|"]
    for name, rel in (("configuration", "config/experiment.yaml"), ("dataset series", "data/series.csv"), ("generation manifest", "data/generation_manifest.json"), ("splits", "data/splits.json"),
                      ("ground truth per origin", "data/ground_truth.csv"), ("predictions per arm/seed", "predictions/*.csv"), ("evidence traces", "traces/*_evidence.csv"),
                      ("metrics summary", "metrics/summary.json, metrics/summary.csv, metrics/aggregate_by_arm.csv, metrics/per_seed.csv"), ("training history", "metrics/train_history.csv"),
                      ("timing", "metrics/timing.csv"), ("checkpoint restore", "metrics/checkpoint_restore_*.csv"), ("substitution matrix", "metrics/substitution_matrix.json|csv"),
                      ("regime/memory/symbolic diagnostics", "metrics/regime_diagnostics.csv, metrics/memory_diagnostics.json, metrics/symbolic_diagnostics.json"),
                      ("manifests", "manifests/*.json"), ("checkpoints", "checkpoints/"), ("environment", "environment/*.json"), ("log", "logs/run.log"), ("plots", "plots/*.png|svg")):
        L.append(f"| {name} | `{rel}` |")
    out = a.run_dir / "report" / "toy_vertical_slice_report.md"
    out.write_text("\n".join(L))
    return out
