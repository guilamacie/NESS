"""The example plugin's own tests, written with the NESS contributor test kit only."""

import numpy as np
import pytest

from ness.plugin_api.testkit import FrozenModuleMixin, ReasonerContractMixin, WriterContractMixin, make_context, port_value
from ness_example_plugin.ar_substrate import ToyLinearARSubstrate
from ness_example_plugin.reasoner import DeterministicThresholdReasoner
from ness_example_plugin.writer import BoundedPointWriter


class TestARSubstrate(FrozenModuleMixin):
    def make_module(self):
        return ToyLinearARSubstrate({"lags": 3, "width": 8, "channels": ["y0", "y1"], "horizons": 4})

    def make_inputs(self, module, rng):
        return {"history": port_value(rng.normal(size=(20, 2)), "history", "observation_series")}

    def test_forecast_is_a_point_forecast_in_the_declared_space(self):
        m, st, inp, out = self._run()
        fc = out.ports["forecast/point"]
        assert fc.forecast_type == "point" and fc.values.shape == (4, 2)
        assert fc.target.schema_id == m.describe().output_port("forecast/point").coordinate_schema_id


class TestThresholdReasoner(ReasonerContractMixin):
    def make_module(self):
        return DeterministicThresholdReasoner({"feature_ports": {"features": 3}, "feature_index": 1, "threshold": 0.0})

    def test_semantics_are_declared_deterministic(self):
        _, _, _, out = self._run()
        assert out.ports["posterior"].weight_semantics == "deterministic_assignment"


class TestBoundedWriter(WriterContractMixin):
    def make_writer(self):
        return BoundedPointWriter({"bound": 0.5})

    def test_corrections_are_bounded(self):
        w = self.make_writer()
        b = np.zeros((4, 2))
        out = w.write(b, np.full(8, 1e6), np)
        assert np.all(np.abs(out) <= 0.5 + 1e-12)
