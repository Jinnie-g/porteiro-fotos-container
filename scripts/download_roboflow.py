"""Baixa o dataset do Roboflow (E0). Uso: ROBOFLOW_API_KEY=... python scripts/download_roboflow.py"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from porteiro.config import api_key, load_config  # noqa: E402
from porteiro.roboflow_io import download  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
cfg = load_config()
rf = cfg["roboflow"]
print(download(rf["workspace"], rf["project"], api_key(), cfg["paths"]["data_root"],
               workers=a.workers, limit=a.limit))
