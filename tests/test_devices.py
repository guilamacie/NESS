import pytest

from ness.contracts import UnsupportedCapability, ValidationError
from ness.runtimes.devices import DeviceSpec, resolve_devices


class _Dev:
    def __init__(self, id, platform, kind="k", process_index=0):
        self.id, self.platform, self.device_kind, self.process_index = id, platform, kind, process_index


class _FakeJax:
    def __init__(self, devs, default="gpu"):
        self._devs, self._default = devs, default

    def devices(self, platform=None):
        return [d for d in self._devs if platform is None or d.platform == platform]

    def default_backend(self):
        return self._default


def test_spec_parsing_and_validation():
    s = DeviceSpec.from_config({"platform": "gpu", "ids": [0, 1]})
    assert s.selection == "ids" and s.ids == (0, 1)
    with pytest.raises(ValidationError):
        DeviceSpec(platform="quantum")
    with pytest.raises(ValidationError):
        DeviceSpec(selection="ids")


def test_resolution_records_four_device_sets_and_fails_closed():
    jax = _FakeJax([_Dev(0, "cpu"), _Dev(1, "gpu", "A"), _Dev(2, "gpu", "A")])
    r = resolve_devices(DeviceSpec(platform="gpu", selection="all"), jax)
    assert [d.id for d in r.visible] == [0, 1, 2] and [d.id for d in r.selected] == [1, 2] and r.realized == ()
    r1 = resolve_devices(DeviceSpec(platform="gpu", selection="ids", ids=(2,)), jax)
    assert [d.id for d in r1.selected] == [2]
    with pytest.raises(UnsupportedCapability, match="ids"):
        resolve_devices(DeviceSpec(platform="gpu", selection="ids", ids=(7,)), jax)
    with pytest.raises(UnsupportedCapability, match="never silently shrinks"):
        resolve_devices(DeviceSpec(platform="gpu", count=8), jax)


def test_missing_platform_errors_unless_fallback_declared():
    jax = _FakeJax([_Dev(0, "cpu")], default="cpu")
    with pytest.raises(UnsupportedCapability, match="ness doctor"):
        resolve_devices(DeviceSpec(platform="gpu"), jax)
    r = resolve_devices(DeviceSpec(platform="gpu", fallback="cpu"), jax)
    assert r.fallback_applied and r.selected[0].platform == "cpu" and r.notes
