"""Load YAML/JSON configuration files into specs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contracts import ValidationError
from ..experiments.spec import ExperimentSpec, parse_experiment


def load_dict(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        import yaml  # small pure-python dependency
        data = yaml.safe_load(text)
    elif p.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValidationError(f"unsupported config extension {p.suffix!r}")
    if not isinstance(data, dict):
        raise ValidationError(f"{p}: top level must be a mapping")
    return data


def load_experiment(path: str | Path) -> ExperimentSpec:
    return parse_experiment(load_dict(path))
