"""ETL-4: materializa os datasets sem zoom, com zoom e revisão em pastas.

Uso: python scripts/export_datasets.py [results/zoom_<data>]

Gera data/datasets/<nome>/images/*.jpg (cópias dos ORIGINAIS, nenhum pixel
alterado), annotations_coco.json (caixas de avaria rust/dent/hole) e
manifest.csv. Só representantes: as cópias do mesmo cluster ficam fora.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
run = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted((ROOT / "results").glob("zoom_*"))[-1]
CATS = [{"id": 1, "name": "rust"}, {"id": 2, "name": "dent"}, {"id": 3, "name": "hole"}]
cat_id = {c["name"]: c["id"] for c in CATS}
for name in ("sem_zoom", "com_zoom", "revisao"):
    df = pd.read_parquet(run / f"dataset_{name}.parquet")
    out = ROOT / "data" / "datasets" / name
    (out / "images").mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    for k, r in enumerate(df.itertuples(), 1):
        src = ROOT / "data" / "roboflow" / "images" / f"{r.image_id}.jpg"
        shutil.copy2(src, out / "images" / src.name)
        meta = json.loads((ROOT / "data" / "roboflow" / "meta" / f"{r.image_id}.json").read_text())
        a = meta.get("annotation") or {}
        images.append({"id": k, "file_name": f"images/{src.name}", "width": a.get("width", 640),
                       "height": a.get("height", 640), "image_id_manifesto": r.image_id})
        for b in a.get("boxes") or []:
            x0, y0 = b["x"] - b["width"] / 2, b["y"] - b["height"] / 2
            anns.append({"id": len(anns) + 1, "image_id": k, "category_id": cat_id[b["label"]],
                         "bbox": [x0, y0, b["width"], b["height"]], "area": b["width"] * b["height"], "iscrowd": 0})
    coco = {"info": {"description": f"Porteiro de Fotos - dataset {name}", "origem": str(run.name)},
            "images": images, "annotations": anns, "categories": CATS}
    (out / "annotations_coco.json").write_text(json.dumps(coco), encoding="utf-8")
    df.to_csv(out / "manifest.csv", index=False)
    print(f"{name}: {len(images)} imagens, {len(anns)} caixas -> {out}")
