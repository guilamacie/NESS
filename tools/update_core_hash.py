"""Regenerate the core-source hash baseline (`src/ness/_core_hash.py`).

Run this after any intentional change under `src/ness/` (except reference plugins) and at every
release. `tests/test_core_hash.py` fails until it is run; `ness audit` reports the mismatch.
Instance implementers never need this: their code lives outside `src/ness/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ness.integrity import write_baseline  # noqa: E402

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "src" / "ness"
    print(f"core source hash baseline written: {write_baseline(root)}")
