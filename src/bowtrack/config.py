"""YAML config loading with a small dot-access wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve config paths relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
