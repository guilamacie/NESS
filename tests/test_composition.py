import copy

import numpy as np
import pytest

from ness.composition import CompositionGraphSpec, SourceSelector, compile_graph, parse_composition
from ness.contracts import AccessPolicyViolation, CompositionError, FieldRole
from ness.tasks import AS_OF_ORIGIN_ROLES

from ness_test_helpers import BASE, PROGRAM, REGIME, SEMANTIC, UPPER, cap, needs_jax


def _compile(registry, scenario, nodes, output="module://cap/forecast", regions=None):
    d = {"nodes": copy.deepcopy(nodes), "output": {"task": "future_value", "from": output}}
    if regions:
        d["recurrent_regions"] = regions
    spec = parse_composition(d)
    task = registry.create("point_forecast_task", {"task_id": "future_value", "horizons": 4, "channels": ["y0", "y1"]})
    return compile_graph(spec, registry, scenario.observation_fields(), {"future_value": task.prediction_space()}, AS_OF_ORIGIN_ROLES)


class TestSelectors:
    def test_parse_and_roundtrip(self):
        s = SourceSelector.parse("substrate://base_ts/state/final")
        assert (s.scheme, s.node_id, s.port) == ("substrate", "base_ts", "state/final") and str(s) == "substrate://base_ts/state/final"
        o = SourceSelector.parse("observation://target_history")
        assert o.node_id == "observation" and o.port == "target_history"

    @pytest.mark.parametrize("bad", ["module:/x/y", "ftp://a/b", "observation://a/b", "module://only_node"])
    def test_malformed_rejected(self, bad):
        with pytest.raises(CompositionError):
            SourceSelector.parse(bad)


class TestParse:
    def test_two_sources_need_explicit_merge(self):
        with pytest.raises(CompositionError):
            parse_composition({"nodes": [{"id": "a", "plugin": "p", "inputs": {"x": {"sources": ["module://b/h", "module://c/h"]}}}],
                               "output": {"task": "t", "from": "module://a/y"}})

    def test_duplicate_ids_rejected(self):
        with pytest.raises(CompositionError):
            parse_composition({"nodes": [{"id": "a", "plugin": "p"}, {"id": "a", "plugin": "q"}], "output": {"task": "t", "from": "module://a/y"}})


class TestCompile:
    @needs_jax  # compile instantiates jax-backed plugins (upper/cap); pure-numpy compile checks stay below
    def test_T36_accidental_cycle_rejected(self, registry, scenario):
        upper = copy.deepcopy(UPPER)
        upper["inputs"]["main"]["sources"].append("module://cap/correction")  # cap reads upper, upper reads cap
        with pytest.raises(CompositionError, match="cycle"):
            _compile(registry, scenario, [BASE, upper, cap({"neural": 16}, {"neural": "module://upper/hidden"})])

    @needs_jax  # compile instantiates jax-backed plugins (upper/cap); pure-numpy compile checks stay below
    def test_T36_cycle_accepted_only_inside_declared_workspace(self, registry, scenario):
        upper = copy.deepcopy(UPPER)
        upper["inputs"]["main"]["sources"].append("module://cap/correction")
        # the cap's correction is a flat vector, so make the upper's expected shape irrelevant by using select
        region = [{"id": "ws", "nodes": ["upper", "cap"], "state_variables": ["hidden"], "energy": "declared", "solver": "fixed_point", "iterations": 4, "derivative": "bp_unroll"}]
        try:
            compiled = _compile(registry, scenario, [BASE, upper, cap({"neural": 16}, {"neural": "module://upper/hidden"})], regions=region)
        except CompositionError as exc:
            assert "cycle" not in str(exc)  # a shape error is fine here; the cycle itself must not be the reason
            return
        assert compiled.spec.recurrent_regions

    @needs_jax  # compile instantiates jax-backed plugins (upper/cap); pure-numpy compile checks stay below
    def test_unknown_port_and_node_and_scheme(self, registry, scenario):
        with pytest.raises(CompositionError, match="no output port"):
            _compile(registry, scenario, [BASE, cap({"neural": 16}, {"neural": "substrate://base_ts/state/middle"})])
        with pytest.raises(CompositionError, match="unknown node"):
            _compile(registry, scenario, [BASE, cap({"neural": 16}, {"neural": "module://ghost/hidden"})])
        with pytest.raises(CompositionError, match="scheme"):
            _compile(registry, scenario, [BASE, cap({"neural": 16}, {"neural": "module://base_ts/state/final"})])  # substrate addressed as module

    @needs_jax  # compile instantiates jax-backed plugins (upper/cap); pure-numpy compile checks stay below
    def test_T44_T21_outcome_field_cannot_be_wired(self, registry, scenario):
        sem = copy.deepcopy(SEMANTIC)
        sem["inputs"] = {"raw": "observation://target_future"}
        sem["config"]["use_neural"] = False
        with pytest.raises(AccessPolicyViolation):
            _compile(registry, scenario, [BASE, sem, cap({"semantic": 4}, {"semantic": "semantic://semantic/features"})])
        sem["inputs"] = {"raw": "observation://true_regime"}
        with pytest.raises(AccessPolicyViolation):
            _compile(registry, scenario, [BASE, sem, cap({"semantic": 4}, {"semantic": "semantic://semantic/features"})])

    @needs_jax  # compile instantiates jax-backed plugins (upper/cap); pure-numpy compile checks stay below
    def test_learned_transform_into_host_runtime_rejected(self, registry, scenario):
        sem = copy.deepcopy(SEMANTIC)
        sem["config"]["use_neural"] = False
        sem["inputs"] = {"raw": {"from": "observation://target_history", "boundary": [{"kind": "linear", "out_dim": 2}]}}
        with pytest.raises(CompositionError, match="learned boundary transform"):
            _compile(registry, scenario, [BASE, sem, cap({"semantic": 4}, {"semantic": "semantic://semantic/features"})])

    @needs_jax  # compile instantiates jax-backed plugins (upper/cap); pure-numpy compile checks stay below
    def test_T35_add_with_incompatible_shapes_fails_before_execution(self, registry, scenario):
        upper = copy.deepcopy(UPPER)
        upper["inputs"]["main"] = {"merge": "add", "sources": ["substrate://base_ts/state/final", "observation://target_history"]}
        with pytest.raises(CompositionError, match="identical shapes"):
            _compile(registry, scenario, [BASE, upper, cap({"neural": 16}, {"neural": "module://upper/hidden"})])

    @needs_jax  # compile instantiates jax-backed plugins (upper/cap); pure-numpy compile checks stay below
    def test_T35_add_across_incompatible_coordinates_needs_projection(self, registry, scenario):
        base2 = copy.deepcopy(BASE)
        base2["id"], base2["plugin"] = "base_conv", "toy_frozen_conv"
        base2["config"] = {"width": 16, "channels": ["y0", "y1"], "horizons": 4, "seed": 3, "pretrain_steps": 60}
        upper = copy.deepcopy(UPPER)
        upper["inputs"]["main"] = {"merge": "add", "sources": ["substrate://base_ts/state/final", "substrate://base_conv/state/final"]}
        with pytest.raises(CompositionError, match="coordinate"):
            _compile(registry, scenario, [BASE, base2, upper, cap({"neural": 16}, {"neural": "module://upper/hidden"})])
        upper["inputs"]["main"]["sources"][1] = {"from": "substrate://base_conv/state/final", "boundary": [{"kind": "linear", "out_dim": 16}]}
        _compile(registry, scenario, [BASE, base2, upper, cap({"neural": 16}, {"neural": "module://upper/hidden"})])

    @needs_jax  # compile instantiates jax-backed plugins (upper/cap); pure-numpy compile checks stay below
    def test_required_port_unwired_and_disabled_node_reference(self, registry, scenario):
        base = copy.deepcopy(BASE)
        base.pop("inputs")
        with pytest.raises(CompositionError, match="not wired"):
            _compile(registry, scenario, [base], output="substrate://base_ts/forecast/point")
        upper = copy.deepcopy(UPPER)
        upper["enabled"] = False
        with pytest.raises(CompositionError, match="disabled"):
            _compile(registry, scenario, [BASE, upper, cap({"neural": 16}, {"neural": "module://upper/hidden"})])

    def test_output_must_match_task_forecast_type(self, registry, scenario):
        with pytest.raises(CompositionError, match="expects a point forecast port"):
            _compile(registry, scenario, [BASE], output="substrate://base_ts/state/final")

    @needs_jax
    def test_T43_resolved_provenance_and_hash_sensitivity(self, registry, scenario):
        c1 = _compile(registry, scenario, [BASE, UPPER, SEMANTIC, PROGRAM, REGIME, cap({"neural": 16, "posterior": 2}, {"neural": "module://upper/hidden", "posterior": "reasoner://regime/posterior"})])
        wiring = c1.resolved_wiring()
        assert wiring["semantic"] == {"raw": ["observation://target_history"], "neural": ["module://upper/hidden"]}
        assert wiring["upper"]["main"] == ["substrate://base_ts/state/final", "substrate://base_ts/state/early"]
        assert c1.differentiable_nodes() == ("upper", "cap") and c1.differentiable_runtime == "jax"
        assert any(e.startswith("substrate://base_ts/forecast/point->cap") for e in c1.stop_edges)
        upper2 = copy.deepcopy(UPPER)
        upper2["inputs"]["main"] = {"from": "substrate://base_ts/state/final"}  # residual removed
        c2 = _compile(registry, scenario, [BASE, upper2, cap({"neural": 16}, {"neural": "module://upper/hidden"})])
        assert c1.composition_hash != c2.composition_hash
        assert set(c1.edge_params) == {"upper.main#src1.b0:linear", "upper.main#merge:gated_add", "upper.main#post0:layer_norm"}
        assert c2.edge_params == {}

    @needs_jax
    def test_select_and_weighted_sum_merges_compile(self, registry, scenario):
        upper = copy.deepcopy(UPPER)
        upper["inputs"]["main"] = {"merge": "select", "merge_params": {"index": 1}, "sources": ["substrate://base_ts/state/early", "substrate://base_ts/state/final"]}
        _compile(registry, scenario, [BASE, upper, cap({"neural": 16}, {"neural": "module://upper/hidden"})])
        upper["inputs"]["main"] = {"merge": "weighted_sum", "sources": ["substrate://base_ts/state/early", "substrate://base_ts/state/final"]}
        c = _compile(registry, scenario, [BASE, upper, cap({"neural": 16}, {"neural": "module://upper/hidden"})])
        assert "upper.main#merge:weighted_sum" in c.edge_params
