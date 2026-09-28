"""Diagrams of experiment configurations for humans (``ness graph``). Requires ``ness[report]``
(matplotlib); imports it lazily so ``import ness.visualize`` stays dependency-free."""

from .graph import GraphModel, build_models, draw_arms_matrix, draw_composition, draw_data_access, draw_stack, model_from_compiled, model_from_spec, render_experiment

__all__ = ["GraphModel", "build_models", "draw_arms_matrix", "draw_composition", "draw_data_access", "draw_stack",
           "model_from_compiled", "model_from_spec", "render_experiment"]
