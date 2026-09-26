import pathlib

import pytest

from ness.cli.main import main

CFG = pathlib.Path(__file__).resolve().parents[1] / "examples" / "vertical_slice_timeseries" / "configs" / "vertical_slice.yaml"


def test_validate_schema_only_and_plugins_and_audit(capsys):
    assert main(["validate", str(CFG), "--schema-only"]) == 0
    assert main(["plugins"]) == 0
    assert main(["audit"]) == 0
    out = capsys.readouterr().out
    assert "toy_frozen_transformer" in out and "hyperon" in out and "fabricpc" in out  # lock lists every numerical dependency (absent/unresolved/version)


def test_validate_full_compiles_all_arms():
    pytest.importorskip("jax")
    assert main(["validate", str(CFG)]) == 0


def test_run_and_inspect_small_protocol(tmp_path):
    pytest.importorskip("jax")
    import yaml
    cfg = yaml.safe_load(CFG.read_text())
    cfg["protocol"].update({"updates": 1, "batch_size": 4, "max_train_requests": 8, "max_test_requests": 4})
    cfg["scenario"]["config"]["n_steps"] = 300
    for a in cfg["arms"].values():
        for n in a["composition"]["nodes"]:
            if n["plugin"].startswith("toy_frozen"):
                n["config"]["pretrain_steps"] = 80
    p = tmp_path / "small.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert main(["run", str(p), "--out", str(tmp_path / "out"), "--arms", "native_baseline", "neural_only_cap"]) == 0
    assert main(["evaluate", str(tmp_path / "out" / "report.json")]) == 0
    assert main(["inspect", str(tmp_path / "out" / "checkpoints")]) == 0
    from ness.checkpoint import CheckpointStore
    mid = CheckpointStore(tmp_path / "out" / "checkpoints").head("neural_only_cap")
    assert main(["inspect", str(tmp_path / "out" / "checkpoints"), mid]) == 0
    assert main(["validate", str(tmp_path / "does_not_exist.yaml")]) == 2
