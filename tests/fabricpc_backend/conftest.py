"""FabricPC test layer: skipped entirely when the extra is not installed. Bootstraps the
FabricPC runtime on CPU before any JAX computation in this test process."""

import importlib.util

import pytest

if importlib.util.find_spec("fabricpc") is None:
    pytest.skip("fabricpc extra not installed", allow_module_level=True)

from ness.runtimes.bootstrap import RuntimeRequest, bootstrap  # noqa: E402

REPORT = bootstrap(RuntimeRequest(backend="fabricpc", platform="cpu"))


@pytest.fixture(scope="session")
def report():
    return REPORT
