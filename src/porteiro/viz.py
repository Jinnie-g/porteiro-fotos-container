"""Pranchas de contato para revisão visual (bordas, duplicatas, vazamento)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _thumb(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        return np.full((size, size, 3), 128, np.uint8)
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, np.uint8)
    y, x = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    return canvas


def grid(items: list[tuple[list[Path], str]], out: Path, size: int = 200, cols: int = 4) -> None:
    """items: lista de (caminhos lado a lado, legenda)."""
    if not items:
        return
    per = max(len(p) for p, _ in items)
    cell_w, cell_h = size * per + 6 * (per - 1), size + 24
    rows = (len(items) + cols - 1) // cols
    canvas = np.full((rows * (cell_h + 10), cols * (cell_w + 10), 3), 255, np.uint8)
    for k, (paths, cap) in enumerate(items):
        r, c = divmod(k, cols)
        y0, x0 = r * (cell_h + 10), c * (cell_w + 10)
        for m, p in enumerate(paths):
            x = x0 + m * (size + 6)
            canvas[y0:y0 + size, x:x + size] = _thumb(p, size)
        cv2.putText(canvas, cap[:60], (x0, y0 + size + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1,
                    cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
