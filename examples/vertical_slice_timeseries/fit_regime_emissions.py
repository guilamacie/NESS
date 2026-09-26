"""Fit-only step: estimate per-regime emission means/scales of the semantic feature vector on
the TRAINING split (the hidden regime is used here as a *fitting label on allowed data*, never
as a predictor input). Prints a YAML fragment to paste into the reasoner config, so the
choice is reproducible and auditable rather than hand-typed."""

from __future__ import annotations

import sys

import numpy as np
import yaml

from ness.plugin_api import default_registry
from ness.plugin_api.testkit import make_context, port_value


def main(config_path: str = "examples/vertical_slice_timeseries/configs/toy_vertical_slice.yaml") -> int:
    cfg = yaml.safe_load(open(config_path))
    reg = default_registry()
    sc = reg.create("toy_regime_dataset", cfg["scenario"]["config"])
    sem_cfg = next(n for n in cfg["x-semantic-a3"]["config"].items()) and cfg["x-semantic-a3"]["config"]
    sem = reg.create("simple_temporal_semantics", {k: v for k, v in sem_cfg.items() if k != "use_neural"} | {"use_neural": False})
    st = sem.initialize(np.random.default_rng(0))
    rows = {0: [], 1: []}
    gt = sc.ground_truth()
    for o in sc.origins("train"):
        b = sc.bundle(o)
        inputs = {"raw": port_value(b.field("series_history").array, "raw", "observation_series"),
                  "events": port_value(b.field("event_future").array, "events", "event_indicator")}
        out = sem.forward(inputs, st, make_context())
        rows[int(gt["regime"][o - 1])].append(out.ports["features"].value)
    names = sem.feature_names()
    means, scales = [], []
    for k in (0, 1):
        m = np.asarray(rows[k])
        means.append([round(float(v), 4) for v in m.mean(0)])
        scales.append([round(float(max(v, 0.02)), 4) for v in m.std(0)])
    print("# feature order:", list(names))
    print(yaml.safe_dump({"means": means, "scales": scales}, default_flow_style=None))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
