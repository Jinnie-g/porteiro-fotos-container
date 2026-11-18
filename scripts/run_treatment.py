"""Executa F0/F1 do guia (E1 a E6) sobre o dataset baixado e grava results/run_<data>/.

Uso:
  ROBOFLOW_API_KEY=... python scripts/download_roboflow.py
  python scripts/run_treatment.py
"""
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from porteiro import dedup, env, geometry, ingest, quality, report, sampling, split  # noqa: E402
from porteiro.config import load_config  # noqa: E402

cfg = load_config()
data = Path(cfg["paths"]["data_root"])
out = Path(cfg["paths"]["results_root"]) / f"run_{date.today().isoformat()}"
out.mkdir(parents=True, exist_ok=True)
env.seed_all(cfg["seed"])
rng = np.random.default_rng(cfg["seed"])

print("E1/E2 manifesto e integridade")
df = ingest.build_manifest(data)

# caixas (xyxy em pixels da imagem) e embeddings do Roboflow, a partir dos metadados
boxes, embs = {}, {}
for iid in df["image_id"]:
    p = data / "meta" / f"{iid}.json"
    if not p.exists():
        continue
    m = json.loads(p.read_text(encoding="utf-8"))
    a = m.get("annotation") or {}
    boxes[iid] = [(b["x"] - b["width"] / 2, b["y"] - b["height"] / 2,
                   b["x"] + b["width"] / 2, b["y"] + b["height"] / 2) for b in a.get("boxes") or []]
    if m.get("embedding"):
        embs[iid] = np.asarray(m["embedding"], np.float32)

print("E4 caracterização")
q = quality.compute_quality_table(df, data, boxes, cfg["borders"])
df = df.merge(q, on="image_id", how="left")
bb = {i: tuple(map(int, s.split(","))) for i, s in zip(df["image_id"], df["bbox_conteudo"]) if isinstance(s, str)}
print("E4 descritores geométricos (OpenCV)")
g = geometry.compute_geometry_table(df, data, bb)
df = df.merge(g, on="image_id", how="left")

print("E3 duplicatas e grupos")
df = df.join(dedup.compute_phash(df, data))
df = df[df["image_id"].isin(embs)].reset_index(drop=True)
emb = np.stack([embs[i] for i in df["image_id"]])
df, dstats, _ = dedup.build_groups(df, emb, cfg["dedup"])

print("E5 auditoria do split original")
audit, astats = dedup.leakage_audit(df, emb, cfg["dedup"])
pairs = dedup.review_pairs(df, emb, rng)

print("E5 novo split agrupado")
df = split.assign_folds(df, cfg["split"]["n_splits"], cfg["seed"])
split.check_no_leak(df)

print("E6 amostra para rotulagem")
amostra = sampling.labeling_sample(df, cfg["labeling"]["n_sample"], cfg["labeling"]["random_frac"],
                                   seed=cfg["seed"])

info = env.run_info(ROOT)
df.to_parquet(out / "manifesto.parquet", index=False)
audit.to_csv(out / "auditoria_vazamento.csv", index=False)
pairs.to_csv(out / "pares_revisao_dedup.csv", index=False)
amostra.to_csv(out / "amostra_rotulagem.csv", index=False)
(out / "pip_freeze.txt").write_text(env.pip_freeze(), encoding="utf-8")
(out / "run_info.json").write_text(json.dumps({**info, "config": {k: v for k, v in cfg.items() if k != "paths"},
                                               "dedup": dstats, "vazamento": astats}, indent=1, default=str),
                                   encoding="utf-8")
report.write(df, audit, pairs, amostra, dstats, astats, info, cfg, data, out)
print("ok ->", out)
