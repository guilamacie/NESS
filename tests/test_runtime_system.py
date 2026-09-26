import copy

import numpy as np
import pytest

from ness.contracts import AccessPolicyViolation, ContractViolation, FieldRole, ModuleState, PointForecast, UnsupportedCapability
from ness.contracts import Differentiability, ModuleDescriptor, ObservationField, PortSchema, PortSpec, ParameterGroupSpec, SubstrateCapabilities, CoordinateSchema, horizon_schema
from ness.plugin_api import BaseModule, ModuleOutputs, PluginDescriptor
from ness.plugin_api.testkit import assert_target_free, drop_outcome_fields, perturb_outcome_fields
from ness.memory import MemoryEvent, MemoryRecord

from ness_test_helpers import BASE, BASELINE_ARM, FULL_ARM, LEARNING, MEMORY, MEMORY_QUERY, PROGRAM, REGIME, SEMANTIC, UPPER, arm, build, cap, first_requests, needs_jax


def _dense_outputs(system):
    def predict(req):
        res, _, _ = system.predict(req)
        return {k: v.point() for k, v in res.outputs.items()}
    return predict


class TestBaselineArm:
    def test_predict_returns_typed_result_with_provenance(self, registry, scenario):
        sysm = build(registry, BASELINE_ARM, scenario)
        req = first_requests(scenario, sysm)[0]
        res, rec, ctx = sysm.predict(req)
        fc = res.outputs[req.queries[0].query_id]
        assert isinstance(fc, PointForecast) and fc.values.shape == (4, 2)
        assert res.diagnostics["resolved_sources"] == {"base_ts": {"history": ["observation://target_history"]}}
        assert res.manifest_id == sysm.manifest().manifest_id
        assert "observation:target_history@" + req.request_id in rec.evidence_graph.ids()
        assert not sysm.trainable and sysm.learning_profile == "none"

    def test_T04_prediction_is_target_free(self, registry, scenario):
        sysm = build(registry, BASELINE_ARM, scenario)
        assert_target_free(_dense_outputs(sysm), first_requests(scenario, sysm)[0])

    def test_permitted_view_removes_outcome_and_oracle_fields(self, registry, scenario):
        sysm = build(registry, BASELINE_ARM, scenario)
        req = first_requests(scenario, sysm)[0]
        assert "target_future" in req.bundle.field_names()  # raw bundle carries it for orchestration
        _, _, ctx = sysm.predict(req)
        assert set(ctx.permitted_bundle.field_names()) == {"target_history", "calendar_history", "calendar_future"}

    def test_learn_not_available_without_rule(self, registry, scenario):
        sysm = build(registry, BASELINE_ARM, scenario)
        with pytest.raises(UnsupportedCapability):
            sysm.learn([])

    def test_trainable_without_rule_is_a_validation_error(self, registry, scenario):
        pytest.importorskip("jax")
        from ness.contracts import ValidationError
        with pytest.raises(ValidationError, match="no learning.rule"):
            build(registry, arm([BASE, UPPER, cap({"neural": 16}, {"neural": "module://upper/hidden"})], learning=None), scenario)

    def test_unsupported_profiles_fail_explicitly(self, registry, scenario):
        pytest.importorskip("jax")
        a = arm([BASE, UPPER, cap({"neural": 16}, {"neural": "module://upper/hidden"})], learning={"rule": "workspace_ep_centered"})
        with pytest.raises(UnsupportedCapability, match="workspace_ep_centered"):
            build(registry, a, scenario)
        if __import__("importlib.util").util.find_spec("fabricpc") is not None:
            from ness.contracts import GradientBoundaryError
            a = arm([BASE, UPPER, cap({"neural": 16}, {"neural": "module://upper/hidden"})], learning={"rule": "workspace_pc_local"})
            with pytest.raises(GradientBoundaryError):  # a FabricPC rule cannot own jax-runtime parameters
                build(registry, a, scenario)
        a = copy.deepcopy(BASELINE_ARM)
        a["inference"] = {"profile": "recurrent_workspace"}
        with pytest.raises(UnsupportedCapability, match="recurrent_workspace"):
            build(registry, a, scenario)


@needs_jax
class TestFullArm:
    def test_zero_init_cap_reproduces_baseline_exactly(self, registry, scenario):  # T18 writer neutrality
        full = build(registry, FULL_ARM, scenario)
        base = build(registry, BASELINE_ARM, scenario)
        for req in first_requests(scenario, full, n=3):
            a = full.predict(req)[0].outputs[req.queries[0].query_id].values
            b = base.predict(req)[0].outputs[req.queries[0].query_id].values
            np.testing.assert_array_equal(a, b)

    def test_T04_T38_full_arm_is_target_free(self, registry, scenario):
        sysm = build(registry, FULL_ARM, scenario)
        assert_target_free(_dense_outputs(sysm), first_requests(scenario, sysm)[0])

    def test_T43_provenance_names_every_source_and_version(self, registry, scenario):
        sysm = build(registry, FULL_ARM, scenario)
        res, rec, _ = sysm.predict(first_requests(scenario, sysm)[0])
        ws = res.diagnostics["resolved_sources"]
        assert ws["regime"] == {"semantic": ["semantic://semantic/features"], "program": ["program://slope_program/value"]}
        assert set(res.diagnostics["node_versions"]) == {"base_ts", "upper", "semantic", "slope_program", "regime", "cap"}
        post = rec.port_values[("regime", "posterior")].payload
        assert post.weight_semantics == "exact_posterior" and set(post.provenance.dependencies) == {rec.port_values[("semantic", "features")].evidence_id, rec.port_values[("slope_program", "value")].evidence_id}
        assert rec.node_diagnostics["semantic"]["used_inputs"] == ["raw", "neural"]  # T34 source identity preserved

    def test_T34_semantic_routing_variants(self, registry, scenario):
        raw_only = copy.deepcopy(SEMANTIC); raw_only["config"]["use_neural"] = False; raw_only["inputs"] = {"raw": "observation://target_history"}
        base_state = copy.deepcopy(SEMANTIC); base_state["inputs"]["neural"] = "substrate://base_ts/state/final"
        both = copy.deepcopy(SEMANTIC)
        outs = {}
        for name, sem, dim in (("raw", raw_only, 4), ("base_final", base_state, 5), ("both", both, 5)):
            sysm = build(registry, arm([BASE, UPPER, sem, cap({"neural": 16, "semantic": dim}, {"neural": "module://upper/hidden", "semantic": "semantic://semantic/features"})]), scenario)
            req = first_requests(scenario, sysm)[0]
            _, rec, _ = sysm.predict(req)
            outs[name] = rec.port_values[("semantic", "features")].payload.value
            assert rec.resolved_sources["semantic"].get("neural", ["<none>"])[0] == {"raw": "<none>", "base_final": "substrate://base_ts/state/final", "both": "module://upper/hidden"}[name]
        assert outs["raw"].shape == (4,) and outs["both"].shape == (5,) and not np.allclose(outs["base_final"][-1], outs["both"][-1])

    def test_program_can_be_removed_by_config(self, registry, scenario):
        no_prog = arm([BASE, UPPER, SEMANTIC, cap({"neural": 16, "semantic": 5}, {"neural": "module://upper/hidden", "semantic": "semantic://semantic/features"})])
        sysm = build(registry, no_prog, scenario)
        assert "slope_program" not in sysm.compiled.node_ids()


@needs_jax
class TestMemoryArm:
    ARM = arm([BASE, UPPER, SEMANTIC, PROGRAM, REGIME, MEMORY_QUERY, cap({"neural": 16, "posterior": 2, "memory": 8}, {"neural": "module://upper/hidden", "posterior": "reasoner://regime/posterior", "memory": "memory://analogues/evidence"})], memory=MEMORY)

    def test_memory_query_runs_on_pinned_causal_view(self, registry, scenario):
        sysm = build(registry, self.ARM, scenario)
        first = first_requests(scenario, sysm, split="train", n=1)[0]
        _, rec0, _ = sysm.predict(first)
        assert rec0.port_values[("analogues", "retrieval")].payload.returned_ids == ()  # nothing matured yet: no match, not zeros-as-facts
        assert not rec0.port_values[("analogues", "evidence")].payload.knownness.any()
        req = first_requests(scenario, sysm, split="train", n=12)[-1]
        res, rec, ctx = sysm.predict(req)
        retr = rec.port_values[("analogues", "retrieval")].payload
        assert retr.returned_ids and all(int(r.split("@")[1]) + 4 <= req.origin for r in retr.returned_ids)  # matured before origin
        assert res.diagnostics["memory_views"]["episodes"] == ctx.memory_views["episodes"].view_id

    def test_T39_sibling_branch_mutation_cannot_change_in_flight_prediction(self, registry, scenario):
        sysm = build(registry, self.ARM, scenario)
        req = first_requests(scenario, sysm, split="train", n=1)[0]
        before = sysm.predict(req)[0].outputs[req.queries[0].query_id].values
        store = sysm.memory.store
        b = store.fork(sysm.memory.snapshot_id, "sibling")
        fake = MemoryRecord("episode@1", "episodes", "episode", 0, (0, 1), {"window": req.bundle.field("target_history").array.copy(), "continuation": np.full((4, 2), 99.0)})
        store.append(b, (MemoryEvent("x", "append", fake),), b.head)
        store.seal(b)
        after = sysm.predict(req)[0].outputs[req.queries[0].query_id].values
        np.testing.assert_array_equal(before, after)

    def test_memory_disabled_means_no_store(self, registry, scenario):
        sysm = build(registry, FULL_ARM, scenario)
        assert sysm.memory is None and sysm.manifest().memory_snapshot_id is None

    def test_memory_node_without_store_fails_explicitly(self, registry, scenario):
        a = copy.deepcopy(self.ARM); a.pop("memory")
        sysm = build(registry, a, scenario)
        with pytest.raises(ContractViolation, match="requires memory store"):
            sysm.predict(first_requests(scenario, sysm)[0])


class WeirdSubstrate(BaseModule):
    """Mock provider with an unseen modality tag (T17): the core must run it unchanged."""
    plugin_id = "weird_modality_substrate"; plugin_version = "0.0.1"; module_kind = "substrate"; runtime = "numpy"; state_schema_id = "test.weird/1"

    def describe(self):
        target = horizon_schema(("y0", "y1"), (1, 2, 3, 4))
        return self._descriptor(
            input_ports=(PortSpec("history", PortSchema("dense", (None, 2)), "observation_series", FieldRole.OBSERVED, Differentiability.STOP, accepts_semantic_types=("observation_series",)),),
            output_ports=(PortSpec("state/final", PortSchema("dense", (None, 3)), "hyperspectral_blob", FieldRole.DERIVED, Differentiability.STOP, "blob[3]"),
                          PortSpec("forecast/point", PortSchema("point_forecast", (4, 2)), "point_forecast", FieldRole.DERIVED, Differentiability.STOP, target.schema_id)),
            parameter_groups=(ParameterGroupSpec("frozen", "frozen", "numpy", False),),
            capabilities=SubstrateCapabilities("numpy", "frozen", ("state/final",), ("forecast/point",), "stop"))

    def forward(self, inputs, state, ctx):
        x = inputs["history"].dense()
        return ModuleOutputs(ports={"state/final": np.tile(x.mean(0)[:1], (x.shape[0], 3)), "forecast/point": PointForecast(np.tile(x[-1], (4, 1)), horizon_schema(("y0", "y1"), (1, 2, 3, 4)), "mean")})


def test_T17_unknown_modality_runs_through_generic_core(registry, scenario):
    registry.register(PluginDescriptor("weird_modality_substrate", "substrate", "0.0.1", WeirdSubstrate, (), "test mock"))
    weird = {"id": "base_ts", "plugin": "weird_modality_substrate", "inputs": {"history": "observation://target_history"}}
    sysm = build(registry, arm([weird], output="substrate://base_ts/forecast/point", learning=None), scenario)
    req = first_requests(scenario, sysm)[0]
    res, _, _ = sysm.predict(req)
    np.testing.assert_array_equal(res.outputs[req.queries[0].query_id].values, np.tile(req.bundle.field("target_history").array[-1], (4, 1)))
