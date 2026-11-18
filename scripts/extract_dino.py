"""Extrai DINOv2 (224 e 448) para os representantes do dataset único. Cache em data/features/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from porteiro.consolidate import build_unico  # noqa: E402
from porteiro.features_dino import load_model, extract  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(0)
run = sorted((ROOT / "results").glob("run_*"))[-1]
unico = build_unico(pd.read_parquet(run / "manifesto.parquet"))
ids = unico["image_id"].tolist()
paths = [ROOT / "data" / "roboflow" / "images" / f"{i}.jpg" for i in ids]
w = ROOT / "data" / "models" / "dinov2_vits14_pretrain.pth"
m = load_model(w)
out = ROOT / "data" / "features"
out.mkdir(parents=True, exist_ok=True)
for side in (224, 448):
    f = extract(m, paths, side)
    np.savez(out / f"dino_vits14_{side}.npz", image_id=np.array(ids), **f)
    print(side, f["cls"].shape, flush=True)
