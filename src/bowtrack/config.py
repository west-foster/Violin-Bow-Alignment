"""YAML config loading.

`configs/default.yaml` holds standard dimensions. If `configs/violin_fit.yaml`
exists (written by scripts/fit_violin_model.py), its fitted violin keypoints
and camera FOV override the defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
FIT_FILE = PROJECT_ROOT / "configs" / "violin_fit.yaml"


def load_config(path: str | Path | None = None, use_fit: bool = True) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["_fitted"] = False
    if use_fit and FIT_FILE.exists():
        fit = yaml.safe_load(FIT_FILE.read_text()) or {}
        cfg["violin_model"].update(fit.get("violin_model", {}))
        cfg["camera"].update(fit.get("camera", {}))
        cfg["_fitted"] = True
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve config paths relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
