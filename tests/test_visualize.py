"""`ness graph`: diagrams from configuration alone (unknown plugins allowed) and, when the
runtime extra is present, from the compiled graph. Skips without matplotlib."""

import importlib.util
import pathlib

import pytest

from ness.cli.main import main

pytestmark = pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="needs matplotlib (ness[report])")

QS = pathlib.Path(__file__).resolve().parents[1] / "examples" / "quickstart"
HAS_JAX = importlib.util.find_spec("jax") is not None


def test_spec_level_drawing_with_unknown_plugins(tmp_path):
    out = tmp_path / "C"
    assert main(["graph", str(QS / "C_bring_your_own_models.yaml"), "--out", str(out), "--spec-only"]) == 0
    names = sorted(p.name for p in out.glob("*.png"))
    assert "full_pc_workspace__composition.png" in names and "full_pc_workspace__stack.png" in names
    assert "experiment__arms.png" in names and "experiment__data_access.png" in names
    assert all(p.stat().st_size > 1000 for p in out.glob("*.png"))


def test_resolved_drawing_marks_gradient_boundaries(tmp_path):
    if not HAS_JAX:
        pytest.skip("needs jax")
    from ness.composition import compile_graph
    from ness.config import load_experiment
    from ness.plugin_api import default_registry
    from ness.visualize import draw_composition, model_from_compiled

    exp = load_experiment(QS / "02_neural_cap.yaml")
    reg = default_registry()
    scenario = reg.create(exp.scenario.plugin, exp.scenario.config)
    tasks = {t.task_id: reg.create(t.plugin, {**t.config, "task_id": t.task_id}) for t in exp.tasks}
    roles = frozenset.intersection(*(t.allowed_roles() for t in tasks.values()))
    arm = exp.arm("neural_cap")
    compiled = compile_graph(arm.composition, reg, scenario.observation_fields(), {k: t.prediction_space() for k, t in tasks.items()}, roles)
    m = model_from_compiled(arm, compiled)
    assert m.resolved and m.node("upper").trainable and m.node("base_ts").trainable is False
    stop = [e for e in m.edges if e.dst_node == "upper" and e.src_node == "base_ts"]
    assert stop and all(e.differentiable is False for e in stop)          # frozen provider: stop-gradient reads
    learned = [e for e in m.edges if e.learned]
    assert any("linear" in "".join(e.boundary) for e in learned)           # the projected early-state residual
    assert draw_composition(m, tmp_path / "x.png").stat().st_size > 1000


def test_cli_graph_resolved_or_fallback(tmp_path):
    out = tmp_path / "g"
    assert main(["graph", str(QS / "04_with_memory.yaml"), "--out", str(out), "--no-substitutions"]) == 0
    assert (out / "neural_cap_with_memory__composition.png").exists() and (out / "experiment__arms.png").exists()
