"""Experiment specifications and the paired-arm runner.

The runner depends on ``ness.runtime.system``; it is exposed lazily so that importing
``ness.runtime.system`` (which needs only the spec) never forms an import cycle."""

from .spec import EXPERIMENT_SCHEMA, ArmSpec, ExperimentSpec, PluginSpec, TaskSpec, parse_arm, parse_experiment

__all__ = ["EXPERIMENT_SCHEMA", "ArmSpec", "ExperimentSpec", "PluginSpec", "TaskSpec", "parse_arm", "parse_experiment",
           "ArmReport", "ExperimentReport", "run_arm", "run_experiment"]

_LAZY = {"ArmReport", "ExperimentReport", "run_arm", "run_experiment"}


def __getattr__(name: str):
    if name in _LAZY:
        from . import runner
        return getattr(runner, name)
    raise AttributeError(name)
