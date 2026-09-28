"""Every quick-start example must stay valid: schema-level always, full compile when its
runtime extra is available (jax for 02-05/07, fabricpc for 06)."""

import importlib.util
import pathlib

import pytest

from ness.cli.main import main

QS = pathlib.Path(__file__).resolve().parents[1] / "examples" / "quickstart"
CONFIGS = sorted(QS.glob("0*.yaml")) + [QS / "B_two_providers_quantiles.yaml"]
TEMPLATE = QS / "C_bring_your_own_models.yaml"   # plugins are placeholders: schema-level only
HAS_JAX = importlib.util.find_spec("jax") is not None
HAS_FPC = importlib.util.find_spec("fabricpc") is not None
HAS_PLUGIN = importlib.util.find_spec("ness_example_plugin") is not None


@pytest.mark.parametrize("cfg", CONFIGS, ids=[c.stem for c in CONFIGS])
def test_quickstart_schema_valid(cfg):
    assert main(["validate", str(cfg), "--schema-only"]) == 0


@pytest.mark.parametrize("cfg", CONFIGS, ids=[c.stem for c in CONFIGS])
def test_quickstart_compiles(cfg):
    if cfg.stem.startswith("06") and not HAS_FPC:
        pytest.skip("needs fabricpc")
    if not cfg.stem.startswith("01") and not HAS_JAX:
        pytest.skip("needs jax")
    if cfg.stem.startswith("05") and not HAS_PLUGIN:
        pytest.skip("needs the external plugin example")
    assert main(["validate", str(cfg)]) == 0


def test_template_architecture_is_schema_valid_and_drawable(tmp_path):
    assert main(["validate", str(TEMPLATE), "--schema-only"]) == 0
    if importlib.util.find_spec("matplotlib") is not None:
        assert main(["graph", str(TEMPLATE), "--out", str(tmp_path), "--spec-only"]) == 0
