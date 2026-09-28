"""T41 third-party plugin isolation and T30 scoped dependency gates."""

import importlib.metadata as md
import pathlib

import pytest

from ness.contracts import PluginDependencyMissing
from ness.plugin_api import fresh_registry

from ness_test_helpers import BASE, UPPER, arm, build, cap, experiment, first_requests

EXAMPLE_IDS = {"toy_linear_ar_substrate", "deterministic_threshold_reasoner", "bounded_point_writer", "heavy_dependency_substrate"}
installed = pytest.mark.skipif(not any(ep.name in EXAMPLE_IDS for ep in md.entry_points(group="ness.plugins")), reason="ness-example-plugin not installed")


def test_core_source_never_mentions_example_plugins():
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "ness"
    text = "\n".join(p.read_text() for p in src.rglob("*.py"))
    for pid in EXAMPLE_IDS:
        assert pid not in text


@installed
def test_T41_example_plugins_discovered_by_entry_points_only(registry):
    reg = fresh_registry()
    for pid in EXAMPLE_IDS:
        assert reg.has(pid) and reg.describe(pid).provided_by.startswith("ness-example-plugin")


@installed
def test_T30_missing_dependency_fails_closed_but_does_not_block_unrelated_configs(registry, scenario):
    d = registry.describe("heavy_dependency_substrate")
    assert d.missing_dependencies() == ("torch_fake_dependency",)
    with pytest.raises(PluginDependencyMissing, match="torch_fake_dependency"):
        registry.create("heavy_dependency_substrate", {})
    sysm = build(registry, arm([BASE], output="substrate://base_ts/forecast/point", learning=None), scenario)  # unrelated config validates and runs
    assert sysm.predict(first_requests(scenario, sysm)[0])[0].outputs


@installed
def test_external_substrate_reasoner_and_writer_compose_with_core_modules(registry, scenario):
    pytest.importorskip("jax")
    ar = {"id": "base_ts", "plugin": "toy_linear_ar_substrate", "config": {"lags": 4, "width": 16, "channels": ["y0", "y1"], "horizons": 4}, "inputs": {"history": "observation://target_history"}}
    sem = {"id": "semantic", "plugin": "simple_temporal_semantics", "config": {"window": 8, "channels": 2, "use_neural": False}, "inputs": {"raw": "observation://target_history"}}
    reasoner = {"id": "regime", "plugin": "deterministic_threshold_reasoner", "config": {"feature_ports": {"features": 4}, "feature_index": 2, "threshold": 0.3}, "inputs": {"features": "semantic://semantic/features"}}
    c = cap({"neural": 16, "posterior": 2}, {"neural": "module://upper/hidden", "posterior": "reasoner://regime/posterior"}, writer={"plugin": "bounded_point_writer", "config": {"bound": 0.5}})
    upper = {"id": "upper", "plugin": "tiny_upper_mlp", "config": {"in_dim": 16, "model_dim": 16}, "inputs": {"main": "substrate://base_ts/state/final"}}
    sysm = build(registry, arm([ar, upper, sem, reasoner, c]), scenario)
    req = first_requests(scenario, sysm)[0]
    res, rec, _ = sysm.predict(req)
    post = rec.port_values[("regime", "posterior")].payload
    assert post.weight_semantics == "deterministic_assignment" and sysm.compiled.node("cap").module.writer.writer_id == "bounded_point_writer"
    assert res.diagnostics["node_versions"]["base_ts"] == ("toy_linear_ar_substrate", "0.1.0")


def test_third_party_program_primitive_via_entry_point():
    """T41 for symbolic operators: the example package contributes `window_max` through the
    `ness.primitives` entry point; the default registry sees it, the interpreter can call it, and
    core has no mention of it."""
    import importlib.util
    import numpy as np
    from ness.symbolic.operators import default_registry

    if importlib.util.find_spec("ness_example_plugin") is None:
        import pytest
        pytest.skip("example plugin not installed")
    reg = default_registry()
    assert "window_max" in reg.names()
    assert "window_max" not in default_registry(include_external=False).names()
    res = reg.get("window_max").fn(np.array([[1.0, 5.0], [3.0, 2.0], [2.0, 4.0]]), 2)
    from ness.contracts import Knownness
    assert res.knownness == Knownness.KNOWN and np.allclose(res.value, [3.0, 4.0])
    assert reg.get("window_max").fn(np.zeros((1, 2)), 3).knownness == Knownness.MISSING   # preconditions fail -> MISSING, never a fabricated value
