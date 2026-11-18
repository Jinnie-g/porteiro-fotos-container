from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("data_root", "results_root"):
        p = Path(cfg["paths"][key])
        cfg["paths"][key] = p if p.is_absolute() else ROOT / p
    return cfg


def api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError("Defina ROBOFLOW_API_KEY no ambiente (nunca no código).")
    return key
