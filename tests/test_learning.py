import copy

import numpy as np
import pytest

from ness.contracts import GradientBoundaryError, UnsupportedCapability
from ness.learning import LEARNING_PROFILES, build_credit_map, resolve_learning_profile
from ness.plugin_api.testkit import DifferentiableModuleMixin, ModuleContractMixin, port_value

from ness_test_helpers import BASE, FULL_ARM, UPPER, arm, build, cap, first_requests, needs_jax

pytestmark = needs_jax


def _batch(system, scenario, n=3):
    from ness.runtimes.jax_runtime import BatchItem
    items = []
    for req in first_requests(scenario, system, split="train", n=n):
        _, rec, _ = system.predict(req, mode="train")
        oc = {q.task_id: scenario.outcome(req.request_id, q.query_id) for q in req.queries}
        items.append(BatchItem(rec, oc))
    return items


class TestUpperTransformerContract(DifferentiableModuleMixin):
    def make_module(self):
        from ness.reference_plugins.upper_modules import TinyUpperTransformer
        return TinyUpperTransformer({"in_dim": 6, "model_dim": 8, "heads": 2})


class TestUpperMLPContract(DifferentiableModuleMixin):
    def make_module(self):
        from ness.reference_plugins.upper_modules import TinyUpperMLP
        return TinyUpperMLP({"in_dim": 6, "model_dim": 8, "hidden_dim": 12})


class TestResidualPointCapContract(DifferentiableModuleMixin):
    output_port = "forecast"

    def make_module(self):
        from ness.reference_plugins.caps import ResidualPointCap
        return ResidualPointCap({"channels": ["y0", "y1"], "horizons": 4, "features": {"neural": 5, "posterior": 2}, "consumer": {"plugin": "concat_linear_consumer", "config": {"init": "normal"}}})

    def make_inputs(self, module, rng):
        from ness.contracts import PointForecast, horizon_schema
        return {"baseline": port_value(PointForecast(rng.normal(size=(4, 2)), horizon_schema(("y0", "y1"), (1, 2, 3, 4))), "baseline", "point_forecast", "point_forecast"),
                "neural": port_value(rng.normal(size=5), "neural"), "posterior": port_value(np.array([0.3, 0.7]), "posterior")}


class TestProfiles:
    def test_vocabulary_declares_unverified_profiles_and_refuses_them(self):
        assert resolve_learning_profile("ff_bp") == "bp_direct"
        # PDF profiles implemented by the FabricPC backend resolve to their implementing plugin
        assert resolve_learning_profile("workspace_pc_local") == "fabricpc_pc_local"
        assert resolve_learning_profile("workspace_epc_local") == "fabricpc_pc_local"
        assert resolve_learning_profile("workspace_bp_unroll") == "fabricpc_bp_through_inference"
        for name in ("reliability_bp_unroll", "workspace_ep_centered"):
            assert LEARNING_PROFILES.get(name).status.value == "unsupported"
            with pytest.raises(UnsupportedCapability):
                resolve_learning_profile(name)


class TestGradientBoundary:
    def test_T22_T42_bp_cannot_own_host_runtime_parameters(self, registry, scenario):
        """A numpy-runtime module that declares a trainable group must be rejected by the BP credit map."""
        from ness.plugin_api import BaseModule, ModuleOutputs, PluginDescriptor
        from ness.contracts import Differentiability, FieldRole, ModuleState, ParameterGroupSpec, PortSchema, PortSpec

        class HostTrainable(BaseModule):
            plugin_id = "host_trainable_semantics"; plugin_version = "0.0.1"; module_kind = "semantic_adapter"; runtime = "numpy"; state_schema_id = "t/1"
            def describe(self):
                return self._descriptor(input_ports=(PortSpec("raw", PortSchema("dense", (None, 2)), "observation_series", FieldRole.OBSERVED, Differentiability.STOP),),
                                        output_ports=(PortSpec("features", PortSchema("dense", (3,)), "semantic_features", FieldRole.DERIVED, Differentiability.STOP),),
                                        parameter_groups=(ParameterGroupSpec("w", "trainable", "numpy", False),))
            def initialize(self, rng):
                return ModuleState(params={"w": {"m": np.zeros((2, 3))}})
            def forward(self, inputs, state, ctx):
                return ModuleOutputs(ports={"features": inputs["raw"].dense()[-1] @ state.params["w"]["m"]})

        registry.register(PluginDescriptor("host_trainable_semantics", "semantic_adapter", "0.0.1", HostTrainable, (), "test"))
        node = {"id": "semantic", "plugin": "host_trainable_semantics", "inputs": {"raw": "observation://target_history"}}
        a = arm([BASE, UPPER, node, cap({"neural": 16, "semantic": 3}, {"neural": "module://upper/hidden", "semantic": "semantic://semantic/features"})])
        with pytest.raises(GradientBoundaryError, match="not a derivative path"):
            build(registry, a, scenario)

    def test_stop_edges_recorded_and_frozen_groups_unowned(self, registry, scenario):
        sysm = build(registry, FULL_ARM, scenario)
        assert sysm.credit.owners["base_ts/frozen"] == "frozen"
        assert set(sysm.owned_groups()) == {"upper/upper", "cap/consumer", "upper.main#src1.b0:linear", "upper.main#merge:gated_add", "upper.main#post0:layer_norm"}
        assert any(e.startswith("semantic://semantic/features->cap") for e in sysm.credit.stop_boundaries)


class TestBPDirect:
    def test_T05_traced_region_matches_eager_forward(self, registry, scenario):
        sysm = build(registry, FULL_ARM, scenario)
        batch = _batch(sysm, scenario, 2)
        for it in batch:
            assert sysm.region().forward_parity(it.record, sysm.node_states, sysm.edge_params) < 1e-9

    def test_T06_gradients_match_finite_differences_on_cap_consumer(self, registry, scenario):
        a = copy.deepcopy(FULL_ARM)
        a["composition"]["nodes"][-1]["config"]["consumer"] = {"plugin": "concat_linear_consumer", "config": {"init": "normal"}}
        sysm = build(registry, a, scenario)
        batch = _batch(sysm, scenario, 2)
        owned = sysm.owned_groups()
        loss, grads, _ = sysm.region().loss_and_grads(batch, sysm.node_states, sysm.edge_params, owned, sysm.tasks)
        w = sysm.node_states["cap"].params["consumer"]["weight"]
        eps = 1e-6
        for (i, j) in [(0, 0), (3, 5), (16, 7)]:
            def f(delta):
                wp = w.copy(); wp[i, j] += delta
                sysm.node_states["cap"].params["consumer"]["weight"] = wp
                l, _, _ = sysm.region().loss_and_grads(batch, sysm.node_states, sysm.edge_params, owned, sysm.tasks)
                sysm.node_states["cap"].params["consumer"]["weight"] = w
                return l
            fd = (f(eps) - f(-eps)) / (2 * eps)
            assert abs(fd - grads["cap/consumer"]["weight"][i, j]) < 1e-5 * max(1.0, abs(fd))

    def test_update_reduces_batch_loss_and_leaves_frozen_untouched(self, registry, scenario):
        sysm = build(registry, FULL_ARM, scenario)
        batch = _batch(sysm, scenario, 4)
        frozen_before = sysm.node_states["base_ts"].params["frozen"]["embed_w"].copy()
        l0, _, _ = sysm.region().loss_and_grads(batch, sysm.node_states, sysm.edge_params, sysm.owned_groups(), sysm.tasks)
        for _ in range(5):
            diag = sysm.learn(batch)
        l1, _, _ = sysm.region().loss_and_grads(batch, sysm.node_states, sysm.edge_params, sysm.owned_groups(), sysm.tasks)
        assert l1 < l0 and diag["update"] == 5
        np.testing.assert_array_equal(sysm.node_states["base_ts"].params["frozen"]["embed_w"], frozen_before)  # T24
        assert sysm.manifest().extra["n_updates"] == 5

    def test_T07_sum_then_divide_equals_microbatch_accumulation(self, registry, scenario):
        sysm = build(registry, FULL_ARM, scenario)
        batch = _batch(sysm, scenario, 4)
        region, owned = sysm.region(), sysm.owned_groups()
        full, _, _ = region.loss_and_grads(batch, sysm.node_states, sysm.edge_params, owned, sysm.tasks)
        # unequal microbatches: a mean of microbatch means is NOT the declared reduction
        m1, _, _ = region.loss_and_grads(batch[:1], sysm.node_states, sysm.edge_params, owned, sysm.tasks)
        m2, _, _ = region.loss_and_grads(batch[1:], sysm.node_states, sysm.edge_params, owned, sysm.tasks)
        assert not np.isclose((m1 + m2) / 2, full)
        task = sysm.tasks["future_value"]
        nums = dens = 0.0
        for b in batch:
            pred = b.record.outputs["future_value"].payload.values
            oc = b.outcomes["future_value"]
            n, d = task.loss_terms(pred, oc.values, oc.mask.astype(float), np)
            nums, dens = nums + n, dens + d
        assert np.isclose(nums / dens, full)
        assert np.isclose((m1 * 8 + m2 * 24) / 32, full)  # numerators/counts combine exactly
