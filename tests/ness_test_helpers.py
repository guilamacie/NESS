"""Shared fixtures. Tests never edit core code: substitution proofs manipulate config only."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from ness.experiments import ExperimentSpec, parse_experiment
from ness.plugin_api import PluginRegistry, fresh_registry
from ness.runtime.system import NessSystem

try:
    import jax  # noqa: F401
    HAS_JAX = True
except Exception:  # pragma: no cover
    HAS_JAX = False

needs_jax = pytest.mark.skipif(not HAS_JAX, reason="jax not installed (pip install ness[jax])")

SCENARIO = {"plugin": "toy_temporal_dataset", "config": {"n_steps": 400, "context": 32, "horizon": 4, "seed": 1234, "train_fraction": 0.7}}
TASKS = [{"id": "future_value", "plugin": "point_forecast_task", "config": {"horizons": 4, "channels": ["y0", "y1"]}}]
PROTOCOL = {"train_split": "train", "test_split": "test", "batch_size": 4, "updates": 2, "max_train_requests": 12, "max_test_requests": 6}

BASE = {"id": "base_ts", "plugin": "toy_frozen_transformer", "config": {"width": 16, "heads": 2, "layers": 2, "channels": ["y0", "y1"], "horizons": 4, "seed": 7, "pretrain_steps": 120},
        "inputs": {"history": "observation://target_history"}}
UPPER = {"id": "upper", "plugin": "tiny_upper_transformer", "config": {"in_dim": 16, "model_dim": 16, "heads": 2},
         "inputs": {"main": {"merge": "gated_add", "sources": ["substrate://base_ts/state/final", {"from": "substrate://base_ts/state/early", "boundary": [{"kind": "linear", "out_dim": 16}]}],
                             "post": [{"kind": "layer_norm"}]}}}
SEMANTIC = {"id": "semantic", "plugin": "simple_temporal_semantics", "config": {"window": 8, "channels": 2, "use_neural": True},
            "inputs": {"raw": "observation://target_history", "neural": "module://upper/hidden"}}
PROGRAM = {"id": "slope_program", "plugin": "typed_program",
           "config": {"output_dim": 2, "port_bindings": {"series": {"port": "series", "dim": 2}}, "constants": {"window": 8},
                      "program": {"schema_version": "ness.program/1", "name": "window_slope_8", "semantics": "typed_numeric_v1", "inputs": {"series": "numeric_series", "window": "int"},
                                  "output": "numeric_vector", "body": {"op": "call", "primitive": "window_slope", "args": {"series": {"op": "input", "name": "series"}, "window": {"op": "input", "name": "window"}}}}},
           "inputs": {"series": "observation://target_history"}}
REGIME_MODEL = {"latent_name": "regime", "states": ["calm", "volatile"], "prior": [0.5, 0.5],
                "emission": {"family": "gaussian_diag", "means": [[0, 0, 0.15, 0.10, 0, 0, 0], [0, 0, 0.45, 0.25, 0, 0, 0]],
                             "scales": [[0.3, 0.3, 0.1, 0.08, 1, 0.3, 0.3], [0.3, 0.3, 0.2, 0.15, 1, 0.3, 0.3]]}}
REGIME = {"id": "regime", "plugin": "exact_finite_regime_reasoner", "config": {"feature_ports": {"semantic": 5, "program": 2}, "model": REGIME_MODEL},
          "inputs": {"semantic": "semantic://semantic/features", "program": "program://slope_program/value"}}
MEMORY_QUERY = {"id": "analogues", "plugin": "analogue_memory_query", "config": {"store": "episodes", "namespace": "episodes", "k": 3, "horizon": 4, "channels": 2},
                "inputs": {"query": "observation://target_history"}}
LEARNING = {"rule": "bp_direct", "optimizer": {"kind": "adam", "lr": 0.003}}
MEMORY = {"name": "episodes", "store": {"plugin": "in_memory_snapshot_store"}, "namespace": "episodes", "seed_from_scenario": True, "learner_appends": True}


def cap(features: dict[str, int], inputs: dict[str, str], writer: Any = None, consumer: Any = None) -> dict:
    cfg: dict[str, Any] = {"channels": ["y0", "y1"], "horizons": 4, "features": features}
    if writer:
        cfg["writer"] = writer
    if consumer:
        cfg["consumer"] = consumer
    return {"id": "cap", "plugin": "residual_point_cap", "config": cfg, "inputs": {"baseline": "substrate://base_ts/forecast/point", **inputs}}


def arm(nodes: list[dict], output: str = "module://cap/forecast", learning: dict | None = LEARNING, memory: dict | None = None, seed: int = 0) -> dict:
    d: dict[str, Any] = {"seed": seed, "composition": {"nodes": copy.deepcopy(nodes), "output": {"task": "future_value", "from": output}}}
    if learning:
        d["learning"] = learning
    if memory:
        d["memory"] = memory
    return d


def experiment(arms: dict[str, dict], protocol: dict | None = None, scenario: dict | None = None, tasks: list | None = None) -> ExperimentSpec:
    return parse_experiment({"schema_version": "ness.experiment/3", "protocol_id": "test", "scenario": scenario or SCENARIO, "tasks": tasks or TASKS,
                             "protocol": protocol or PROTOCOL, "arms": arms})


FULL_ARM = arm([BASE, UPPER, SEMANTIC, PROGRAM, REGIME, cap({"neural": 16, "semantic": 5, "program": 2, "posterior": 2},
                                                              {"neural": "module://upper/hidden", "semantic": "semantic://semantic/features",
                                                               "program": "program://slope_program/value", "posterior": "reasoner://regime/posterior"})])
BASELINE_ARM = arm([BASE], output="substrate://base_ts/forecast/point", learning=None)


def build(registry, arm_dict: dict, scenario=None, arm_id: str = "a", **kw) -> NessSystem:
    exp = experiment({arm_id: arm_dict}, **kw)
    return NessSystem.build(exp, exp.arm(arm_id), registry, scenario=scenario)


def first_requests(scenario, system: NessSystem, split: str = "test", n: int = 3):
    return list(scenario.iter_requests(split, tuple(system.tasks.values()), {"max_requests": n}))
