"""Run the vertical slice programmatically and print the paired report."""

from __future__ import annotations

import pathlib
import sys

from ness.config import load_experiment
from ness.experiments import run_experiment
from ness.plugin_api import default_registry

HERE = pathlib.Path(__file__).resolve().parent


def main(out: str = "runs/vertical_slice") -> int:
    exp = load_experiment(HERE / "configs" / "vertical_slice.yaml")
    report = run_experiment(exp, default_registry(), out)
    print(report.summary_table())
    for arm_id, r in report.arms.items():
        print(f"{arm_id}: manifest {r.manifest_id[:12]} checkpoint {r.checkpoint_manifest[:12] if r.checkpoint_manifest else None} seconds {r.seconds:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
