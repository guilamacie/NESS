import numpy as np
import pytest

from ness.contracts import AvailabilityCut, ContractViolation, QueryBudget
from ness.memory import EMPTY_SNAPSHOT_ID, InMemorySnapshotStore, MemoryEvent, MemoryRecord, QueryPlan
from ness.plugin_api.testkit import MemoryStoreContractMixin


class TestInMemoryStoreContract(MemoryStoreContractMixin):  # T09 fork isolation etc.
    def make_store(self):
        return InMemorySnapshotStore()


def _seed(store, records):
    b = store.fork(EMPTY_SNAPSHOT_ID, "seed")
    store.append(b, tuple(MemoryEvent(f"e:{r.record_id}", "append", r) for r in records), b.head)
    return store.seal(b)


def test_analogue_eligibility_filters_before_ranking():
    """A forbidden (future) record that would rank first must not even be a candidate."""
    st = InMemorySnapshotStore()
    q = np.sin(np.arange(16).reshape(8, 2))
    recs = (
        MemoryRecord("past_far", "episodes", "episode", 5, (0, 8), {"window": q + 3.0, "continuation": np.ones((4, 2))}),
        MemoryRecord("future_exact", "episodes", "episode", 50, (40, 48), {"window": q.copy(), "continuation": np.zeros((4, 2))}),
    )
    m = _seed(st, recs)
    view = st.open_view(m.snapshot_id, AvailabilityCut(10))
    res = st.query(view, QueryPlan("analogue", "episodes", {"query_window": q, "k": 1}), QueryBudget())
    assert [r.record_id for r in res.returned] == ["past_far"]
    assert res.candidate_count == 1
    later = st.open_view(m.snapshot_id, AvailabilityCut(60))
    res2 = st.query(later, QueryPlan("analogue", "episodes", {"query_window": q, "k": 1}), QueryBudget())
    assert [r.record_id for r in res2.returned] == ["future_exact"]


def test_export_import_roundtrip_is_content_addressed():
    st = InMemorySnapshotStore()
    m = _seed(st, (MemoryRecord("r", "ns", "episode", 1, (0, 1), {"window": np.arange(4.0).reshape(2, 2)}, {"k": "v"}),))
    arrays, data = st.export_snapshot(m.snapshot_id)
    st2 = InMemorySnapshotStore()
    m2 = st2.import_snapshot(arrays, data)
    assert m2.snapshot_id == m.snapshot_id
    data["records"][0]["data"]["k"] = "tampered"
    with pytest.raises(ContractViolation):
        InMemorySnapshotStore().import_snapshot(arrays, data)


def test_no_match_is_partial_not_zero():
    st = InMemorySnapshotStore()
    view = st.open_view(EMPTY_SNAPSHOT_ID, AvailabilityCut(10))
    res = st.query(view, QueryPlan("analogue", "episodes", {"query_window": np.zeros((8, 2)), "k": 2}), QueryBudget())
    assert res.returned == () and res.candidate_count == 0
