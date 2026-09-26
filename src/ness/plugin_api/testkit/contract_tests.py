"""Standard contract-test mixins. Subclass in a plugin's test module, override the
``make_*`` hooks, and pytest collects the checks. Each mixin documents which PDF /
addendum tests it partially covers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ...contracts import (
    ContractViolation,
    FieldRole,
    HypothesisEvidence,
    IncompatibleVersion,
    MODULE_KINDS,
    ModuleState,
    PREDICTION_TIME_ROLES,
    StateSnapshot,
    canonical_json,
)
from ..module import MacroModule, PortValue, RuntimeContext
from .harness import make_context, random_inputs_for


def _dense(o: Any) -> np.ndarray:
    if hasattr(o, "dense"):
        return np.asarray(o.dense(), dtype=np.float64)
    return np.asarray(o, dtype=np.float64)


class ModuleContractMixin:
    """Every MacroModule: valid descriptor, declared ports produced, determinism,
    restricted-data state round trip, fail-closed restore (T40)."""

    seed = 0

    def make_module(self) -> MacroModule:  # pragma: no cover - override
        raise NotImplementedError

    def make_inputs(self, module: MacroModule, rng: np.random.Generator) -> dict[str, PortValue]:
        return random_inputs_for(module.describe(), rng)

    def make_context(self, module: MacroModule) -> RuntimeContext:
        return make_context(node_id="node_under_test", seed=self.seed)

    def _run(self):
        m = self.make_module()
        rng = np.random.default_rng(self.seed)
        state = m.initialize(rng)
        inputs = self.make_inputs(m, np.random.default_rng(self.seed + 1))
        out = m.forward(inputs, state, self.make_context(m))
        return m, state, inputs, out

    def test_describe_is_valid(self):
        d = self.make_module().describe()
        assert d.module_kind in MODULE_KINDS
        assert d.plugin_id and d.plugin_version
        canonical_json(d.canonical())  # must be canonicalisable
        for p in d.output_ports:
            assert p.availability_role in PREDICTION_TIME_ROLES or p.availability_role == FieldRole.DERIVED

    def test_forward_produces_declared_ports(self):
        m, _, _, out = self._run()
        for p in m.describe().output_ports:
            assert p.name in out.ports, f"missing declared output port {p.name}"

    def test_forward_is_deterministic(self):
        m, state, inputs, out1 = self._run()
        out2 = m.forward(inputs, state, self.make_context(m))
        for k in out1.ports:
            a, b = out1.ports[k], out2.ports[k]
            if isinstance(a, (np.ndarray, float, int)) or hasattr(a, "dense"):
                np.testing.assert_allclose(_dense(a), _dense(b))

    def test_state_roundtrip_reproduces_forward(self):
        m, state, inputs, out1 = self._run()
        snap = m.snapshot_state(state)
        assert isinstance(snap, StateSnapshot)
        restored = m.restore_state(snap)
        assert m.snapshot_state(restored).content_hash() == snap.content_hash()
        out2 = m.forward(inputs, restored, self.make_context(m))
        for k in out1.ports:
            if isinstance(out1.ports[k], (np.ndarray,)) or hasattr(out1.ports[k], "dense"):
                np.testing.assert_allclose(_dense(out1.ports[k]), _dense(out2.ports[k]))

    def test_incompatible_state_fails_closed(self):
        m, state, _, _ = self._run()
        snap = m.snapshot_state(state)
        bad = StateSnapshot("someone_else", snap.plugin_version, snap.state_schema_id, snap.arrays, snap.data)
        with pytest.raises(IncompatibleVersion):
            m.restore_state(bad)
        bad_schema = StateSnapshot(snap.plugin_id, snap.plugin_version, snap.state_schema_id + ".other", snap.arrays, snap.data)
        with pytest.raises(IncompatibleVersion):
            m.restore_state(bad_schema)


class FrozenModuleMixin(ModuleContractMixin):
    """Frozen providers (T24): parameters untouched by forward; all groups declared frozen."""

    def test_all_parameter_groups_frozen(self):
        d = self.make_module().describe()
        assert d.parameter_groups, "a substrate must declare its parameter groups"
        assert all(g.mutability == "frozen" for g in d.parameter_groups)

    def test_forward_does_not_mutate_state(self):
        m, state, inputs, _ = self._run()
        before = m.snapshot_state(state).content_hash()
        m.forward(inputs, state, self.make_context(m))
        assert m.snapshot_state(state).content_hash() == before


class DifferentiableModuleMixin(ModuleContractMixin):
    """Trainable modules (T06/T42): apply()==forward(), gradients exist through the public
    path and match central finite differences on the trainable groups."""

    fd_eps = 1e-5
    fd_rtol = 1e-4
    output_port = "hidden"

    def test_apply_matches_forward(self):
        jnp = pytest.importorskip("jax.numpy")
        m, state, inputs, out = self._run()
        dense = {k: jnp.asarray(_dense(v.payload)) for k, v in inputs.items()}
        via_apply = m.apply(state.params, dense, state, jnp)  # type: ignore[attr-defined]
        np.testing.assert_allclose(np.asarray(via_apply[self.output_port]), _dense(out.ports[self.output_port]), rtol=1e-6, atol=1e-8)

    def test_gradient_matches_finite_differences(self):
        jax = pytest.importorskip("jax")
        jnp = jax.numpy
        m, state, inputs, _ = self._run()
        groups = [g.name for g in m.describe().trainable_groups()]
        assert groups, "differentiable module must declare a trainable group"
        dense = {k: jnp.asarray(_dense(v.payload)) for k, v in inputs.items()}
        probe = np.random.default_rng(3).normal(size=np.asarray(m.apply(state.params, dense, state, jnp)[self.output_port]).shape)  # type: ignore[attr-defined]

        def f(params):
            return jnp.sum(m.apply(params, dense, state, jnp)[self.output_port] * probe)  # type: ignore[attr-defined]

        trainable = {g: {n: jnp.asarray(a) for n, a in state.params[g].items()} for g in groups}
        frozen = {g: grp for g, grp in state.params.items() if g not in groups}
        grads = jax.grad(lambda tp: f({**frozen, **tp}))(trainable)
        checked = 0
        for g in groups:
            for n, arr in state.params[g].items():
                flat = np.array(arr, dtype=np.float64).reshape(-1)
                for idx in np.linspace(0, flat.size - 1, num=min(3, flat.size), dtype=int):
                    e = np.zeros_like(flat)
                    e[idx] = self.fd_eps
                    def fp(delta):
                        p = {gg: {nn: jnp.asarray(aa) for nn, aa in state.params[gg].items()} for gg in state.params}
                        p[g][n] = jnp.asarray((flat + delta).reshape(arr.shape))
                        return float(f(p))
                    fd = (fp(e) - fp(-e)) / (2 * self.fd_eps)
                    an = float(np.asarray(grads[g][n]).reshape(-1)[idx])
                    assert abs(fd - an) <= self.fd_rtol * max(1.0, abs(fd), abs(an)) + 1e-6, f"{g}/{n}[{idx}] fd={fd} ad={an}"
                    checked += 1
        assert checked > 0


class ReasonerContractMixin(ModuleContractMixin):
    """Probabilistic reasoners: posterior is a distribution with declared semantics and the
    approximation status is explicit (T37/T38 support)."""

    posterior_port = "posterior"

    def test_posterior_is_hypothesis_evidence(self):
        _, _, _, out = self._run()
        h = out.ports[self.posterior_port]
        assert isinstance(h, HypothesisEvidence)
        assert h.weight_semantics in ("exact_posterior", "approximate_posterior", "deterministic_assignment", "normalized_beam")
        assert np.isclose(h.weights.sum(), 1.0)

    def test_reasoner_declares_capabilities(self):
        d = self.make_module().describe()
        caps = d.capabilities
        for f in ("runtime", "inference_methods", "gradient_boundary", "approximation", "serialization_version"):
            assert hasattr(caps, f), f"reasoner capabilities missing {f}"

    def test_ignores_undeclared_ports(self):
        """Extra (e.g. outcome-like) values not declared as ports must not change the posterior."""
        m, state, inputs, out1 = self._run()
        from .harness import port_value
        extra = dict(inputs)
        extra["__leaked_target__"] = port_value(np.ones(3) * 1e6, "__leaked_target__")
        out2 = m.forward(extra, state, self.make_context(m))
        np.testing.assert_allclose(out1.ports[self.posterior_port].weights, out2.ports[self.posterior_port].weights)


class ScenarioContractMixin:
    """Dataset/scenario providers (T44): outcome-only fields never enter a permitted view;
    requests carry no targets; outcomes are released only after the origin."""

    def make_scenario(self):  # pragma: no cover - override
        raise NotImplementedError

    def make_tasks(self) -> tuple[Any, ...]:  # pragma: no cover - override
        raise NotImplementedError

    def test_permitted_view_excludes_outcome_fields(self):
        sc = self.make_scenario()
        tasks = self.make_tasks()
        fields = sc.observation_fields()
        outcome_fields = {n for n, f in fields.items() if f.role in (FieldRole.OUTCOME_ONLY, FieldRole.EVALUATION_ORACLE)}
        n = 0
        for req in sc.iter_requests(sc.splits()[0], tasks, {"max_requests": 5}):
            for q in req.queries:
                task = next(t for t in tasks if t.task_id == q.task_id)
                view = req.bundle.select(task.permitted_view(req.bundle, q))
                view.assert_prediction_safe()
                assert not (set(view.field_names()) & outcome_fields)
                for o in view.observations:
                    assert o.available_at <= req.origin
            n += 1
        assert n > 0

    def test_outcomes_release_after_origin(self):
        sc = self.make_scenario()
        tasks = self.make_tasks()
        for req in sc.iter_requests(sc.splits()[0], tasks, {"max_requests": 3}):
            for q in req.queries:
                oc = sc.outcome(req.request_id, q.query_id)
                assert oc is not None and oc.available_at > req.origin


class MemoryStoreContractMixin:
    """Memory stores (T09/T39): fork isolation, expected-head conflicts, idempotent events,
    content-addressed seals, and causal view cutoffs."""

    def make_store(self):  # pragma: no cover - override
        raise NotImplementedError

    def make_records(self, n: int = 4):
        from ...memory import MemoryRecord
        return tuple(MemoryRecord(f"r{i}", "ns", "episode", available_at=10 + i, event_span=(i, i + 1), arrays={"window": np.full((2, 2), float(i))}) for i in range(n))

    def test_fork_isolation_and_seal_identity(self):
        from ...contracts import AvailabilityCut, MemoryConflict, QueryBudget
        from ...memory import EMPTY_SNAPSHOT_ID, MemoryEvent, QueryPlan
        st = self.make_store()
        recs = self.make_records()
        a = st.fork(EMPTY_SNAPSHOT_ID, "a")
        st.append(a, tuple(MemoryEvent(f"e{r.record_id}", "append", r) for r in recs[:2]), a.head)
        base = st.seal(a)
        b1, b2 = st.fork(base.snapshot_id, "b1"), st.fork(base.snapshot_id, "b2")
        view_before = st.open_view(base.snapshot_id, AvailabilityCut(100))
        st.append(b1, (MemoryEvent("x", "append", recs[2]),), b1.head)
        assert len(st.open_view(base.snapshot_id, AvailabilityCut(100)).records) == len(view_before.records)  # sibling unaffected
        assert set(b2.record_ids()) == {"r0", "r1"}
        with pytest.raises(MemoryConflict):
            st.append(b1, (MemoryEvent("y", "append", recs[3]),), expected_head="stale")
        rc = st.append(b1, (MemoryEvent("x", "append", recs[2]),), b1.head)
        assert rc.skipped_duplicates == ("x",)
        s1 = st.seal(b1)
        b3 = st.fork(base.snapshot_id, "b3")
        st.append(b3, (MemoryEvent("z", "append", recs[2]),), b3.head)
        assert st.seal(b3).snapshot_id == s1.snapshot_id  # same content -> same id
        cut = st.open_view(s1.snapshot_id, AvailabilityCut(11))
        assert {r.record_id for r in cut.records} == {"r0", "r1"}
        res = st.query(cut, QueryPlan("list", "ns"), QueryBudget())
        assert len(res.returned) == 2


class WriterContractMixin:
    """Writers (T18): zero correction reproduces the baseline exactly; quantile writers keep order."""

    def make_writer(self):  # pragma: no cover - override
        raise NotImplementedError

    def make_baseline(self) -> np.ndarray:
        return np.sort(np.random.default_rng(0).normal(size=(4, 2, 3)), axis=-1)

    def test_zero_correction_is_identity(self):
        w = self.make_writer()
        b = self.make_baseline()
        if w.forecast_type == "point":
            b = b[..., 0]
        out = w.write(b, np.zeros(w.correction_dim(b.shape)), np)
        np.testing.assert_allclose(out, b)
        assert w.is_baseline_preserving_at_zero()

    def test_quantile_writer_preserves_monotonicity(self):
        w = self.make_writer()
        if w.forecast_type != "quantile":
            pytest.skip("point writer")
        b = self.make_baseline()
        out = w.write(b, np.random.default_rng(1).normal(size=w.correction_dim(b.shape)), np)
        assert np.all(np.diff(out, axis=-1) >= -1e-12)
