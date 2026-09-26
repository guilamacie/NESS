"""FabricPC state in NESS checkpoints: restricted arrays only, never pickled FabricPC objects.

A ``fabricpc_residual_cap`` snapshot stores ``params/workspace/<node>|weights|<edge>`` arrays
plus JSON metadata (graph digest, FabricPC version, adapter version, algorithm spec). On
restore the plugin rebuilds the FabricPC structure from its own config, checks the digest,
and loads the arrays through the compat adapter. FabricPC's own checkpoint tooling (orbax)
is not used: the NESS manifest is authoritative."""

from __future__ import annotations

from typing import Any

from ...contracts import IncompatibleVersion

STATE_SCHEMA = "ness.cap.fabricpc_residual/1"


def check_restorable(meta: dict[str, Any], current_digest_hash: str, current_family: str) -> None:
    if meta.get("structure_digest_hash") != current_digest_hash:
        raise IncompatibleVersion("FabricPC workspace checkpoint was produced by a different graph structure/inference configuration; refusing to restore")
    fam = str(meta.get("fabricpc_family", ""))
    if fam and fam != current_family:
        raise IncompatibleVersion(f"checkpoint produced with FabricPC family {fam}, installed family {current_family}; no migration exists (fail closed)")
