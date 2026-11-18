"""Gera pranchas numeradas para rotular o zoom (S4). Uso: python scripts/make_label_sheets.py <rodada>

Cada prancha tem 20 imagens com um número grande; o mapa número → image_id fica em
data/label_sheets/<rodada>_mapa.csv. O anotador não vê fonte, estrato nem split.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rodada = sys.argv[1]
amostra = pd.read_csv(ROOT / "labels" / "amostra_zoom.csv")
s = amostra[amostra["rodada"] == rodada].sort_values("ordem")
out = ROOT / "data" / "label_sheets"
out.mkdir(parents=True, exist_ok=True)
T, COLS, PER = 256, 5, 20
rows = []
for k in range(0, len(s), PER):
    part = s.iloc[k:k + PER]
    nrow = (len(part) + COLS - 1) // COLS
    canvas = np.full((nrow * (T + 8), COLS * (T + 8), 3), 255, np.uint8)
    for j, iid in enumerate(part["image_id"]):
        img = cv2.imread(str(ROOT / "data" / "roboflow" / "images" / f"{iid}.jpg"))
        img = cv2.resize(img, (T, T), interpolation=cv2.INTER_AREA)
        r, c = divmod(j, COLS)
        y, x = r * (T + 8), c * (T + 8)
        canvas[y:y + T, x:x + T] = img
        num = k + j
        cv2.rectangle(canvas, (x, y), (x + 54, y + 30), (255, 255, 255), -1)
        cv2.putText(canvas, str(num), (x + 4, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2, cv2.LINE_AA)
        rows.append({"num": num, "image_id": iid, "prancha": k // PER})
    cv2.imwrite(str(out / f"{rodada}_{k // PER:02d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
pd.DataFrame(rows).to_csv(out / f"{rodada}_mapa.csv", index=False)
print(len(rows), "imagens em", (len(rows) + PER - 1) // PER, "pranchas")
