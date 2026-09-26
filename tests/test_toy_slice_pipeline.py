"""Tiny end-to-end run of the toy vertical slice pipeline (artifacts + report) on the new
regime dataset. Skips without jax; report generation skips without matplotlib/pandas."""

import pathlib

import pytest
import yaml

from ness.cli.main import main

CFG = pathlib.Path(__file__).resolve().parents[1] / "examples" / "vertical_slice_timeseries" / "configs" / "toy_vertical_slice.yaml"


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    pytest.importorskip("jax")
    cfg = yaml.safe_load(CFG.read_text())
    cfg["scenario"]["config"].update({"n_steps": 300, "segment_length": 60})
    cfg["protocol"].update({"updates": 2, "batch_size": 4, "seeds": [0], "max_train_requests": 12, "max_test_requests": 10, "restore_check_requests": 4})
    for a in list(cfg["arms"].values()) + cfg["substitutions"]:
        for n in a["composition"]["nodes"]:
            if n["plugin"].startswith("toy_frozen"):
                n["config"]["pretrain_steps"] = 60
    cfg["substitutions"] = cfg["substitutions"][:3]
    out = tmp_path_factory.mktemp("toy")
    p = out / "tiny.yaml"
    p.write_text(yaml.safe_dump(cfg))
    run_dir = out / "run"
    assert main(["run", str(p), "--out", str(run_dir)]) == 0
    return run_dir


def test_artifacts_written(tiny_run):
    for rel in ("config/experiment.yaml", "data/series.csv", "data/ground_truth.csv", "data/generation_manifest.json", "metrics/summary.json", "metrics/summary.csv",
                "metrics/substitution_matrix.json", "metrics/train_history.csv", "metrics/timing.csv", "environment/core_source_hash.json", "logs/run.log"):
        assert (tiny_run / rel).exists(), rel
    assert list((tiny_run / "predictions").glob("*.csv")) and list((tiny_run / "traces").glob("*_evidence.csv")) and list((tiny_run / "manifests").glob("*.json"))
    import json
    summary = json.loads((tiny_run / "metrics" / "summary.json").read_text())
    for k, v in summary["arms"].items():
        assert v["status"] == "ok", (k, v.get("error"))
        assert v["causal_check"] == "pass"
        if v["checkpoint_manifest"]:
            assert v["restore_check"]["max_abs_diff"] == 0.0 and v["restore_check"]["restored_manifest_matches"]
    subs = json.loads((tiny_run / "metrics" / "substitution_matrix.json").read_text())
    assert all(r["overall"] == "pass" for r in subs["rows"]), [(r["id"], {c: r[c] for c in subs["checks"]}) for r in subs["rows"] if r["overall"] != "pass"]


def test_report_regenerates_from_artifacts(tiny_run):
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    assert main(["report", str(tiny_run)]) == 0
    md = tiny_run / "report" / "toy_vertical_slice_report.md"
    assert md.exists() and "## 12. Modularity" in md.read_text()
    plots = {p.name for p in (tiny_run / "plots").glob("*.png")}
    for f in ("fig01_dataset.png", "fig02_architecture.png", "fig03_trajectories.png", "fig04_metric_by_arm.png", "fig06_regime_posterior.png", "fig07_training_curves.png",
              "fig08_symbolic.png", "fig09_memory_retrieval.png", "fig10_substitution_matrix.png", "fig11_checkpoint_restore.png", "fig12_cost.png"):
        assert f in plots, f
    assert main(["evaluate", str(tiny_run)]) == 0
