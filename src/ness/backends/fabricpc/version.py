"""FabricPC version detection and routing. Fail closed on unknown API families.

known tested version        -> verified
known family, untested patch -> qualified (allowed; recorded as such)
unknown minor / API family   -> unsupported (explicit override runs it as experimental,
                                never as verified)
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import importlib.util
import os
from dataclasses import dataclass
from typing import Any

from ...contracts import PluginDependencyMissing, UnsupportedCapability
from . import ADAPTER_VERSION

# One entry per supported API family. Widening this table is the *last* step of the upgrade
# protocol in docs/FABRICPC_UPGRADE.md, never the first.
SUPPORTED_FAMILIES: dict[str, dict[str, Any]] = {
    "0.6": {
        "adapter": "ness.backends.fabricpc.compat.v0_6",
        "tested_versions": ("0.6.0",),
        "tested_artifacts": {"0.6.0": "sha256:50766c7cda6dbe1325eb7e673410a88ad0bb0cd0e5020ab76f303575c006b46f"},
        "upstream_revision": {"0.6.0": "8406e6a838442391fd3089958e1a2c1b44c57eda"},
    },
}
OVERRIDE_ENV = "NESS_FABRICPC_ALLOW_UNVERIFIED"


@dataclass(frozen=True, slots=True)
class FabricPCVersionInfo:
    installed_version: str
    family: str
    adapter: str | None
    status: str  # verified | qualified | unsupported | experimental_override
    reason: str
    adapter_version: str = ADAPTER_VERSION

    def canonical(self) -> dict:
        return {"installed_version": self.installed_version, "family": self.family, "adapter": self.adapter, "status": self.status,
                "reason": self.reason, "adapter_version": self.adapter_version}


def installed_version() -> str | None:
    if importlib.util.find_spec("fabricpc") is None:
        return None
    try:
        return md.version("fabricpc")
    except md.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def classify(version: str) -> FabricPCVersionInfo:
    parts = version.split(".")
    family = ".".join(parts[:2]) if len(parts) >= 2 else version
    entry = SUPPORTED_FAMILIES.get(family)
    if entry is None:
        if os.environ.get(OVERRIDE_ENV) == "1":
            # An override selects the newest adapter and marks the run experimental. The
            # compatibility matrix is NOT updated by an override.
            newest = sorted(SUPPORTED_FAMILIES)[-1]
            return FabricPCVersionInfo(version, family, SUPPORTED_FAMILIES[newest]["adapter"], "experimental_override",
                                       f"family {family} is not in the compatibility table; {OVERRIDE_ENV}=1 forces the {newest} adapter (experimental, never verified)")
        return FabricPCVersionInfo(version, family, None, "unsupported",
                                   f"FabricPC {version} (API family {family}) is not in NESS's compatibility table {sorted(SUPPORTED_FAMILIES)}; "
                                   f"install a supported version (pip install 'fabricpc>=0.6.0,<0.7') or set {OVERRIDE_ENV}=1 to run experimentally")
    if version in entry["tested_versions"]:
        return FabricPCVersionInfo(version, family, entry["adapter"], "verified", f"tested version; artifact {entry['tested_artifacts'].get(version, '?')}")
    return FabricPCVersionInfo(version, family, entry["adapter"], "qualified", f"patch version in the supported {family}.x family but not itself in the tested list {entry['tested_versions']}")


def check_installed_version(profile: str = "development") -> FabricPCVersionInfo:
    v = installed_version()
    if v is None:
        raise PluginDependencyMissing("fabricpc is not installed; pip install 'ness[fabricpc]'")
    info = classify(v)
    if info.status == "unsupported":
        raise UnsupportedCapability(info.reason)
    if profile == "scientific" and info.status != "verified":
        raise UnsupportedCapability(f"scientific profile requires a verified FabricPC version; installed {v} is {info.status}: {info.reason}")
    return info


def load_adapter(profile: str = "development"):
    info = check_installed_version(profile)
    assert info.adapter is not None
    return importlib.import_module(info.adapter), info


def describe_installed() -> dict[str, Any]:
    v = installed_version()
    if v is None:
        return {"installed_version": None, "status": "absent", "reason": "fabricpc not installed", "adapter": None, "family": None, "adapter_version": ADAPTER_VERSION}
    return classify(v).canonical()
