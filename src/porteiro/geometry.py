"""E4/E7: descritores geométricos (OpenCV), calculados só na região de conteúdo.

Bordas Canny por célula 4x4, orientação e coerência do tensor de estrutura,
segmentos LSD, índice de paralelismo, cantos Shi-Tomasi e pico espectral da
periodicidade das corrugações. Tudo normalizado por área.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

WORK = 512  # lado maior de trabalho


def _resize(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    s = WORK / max(h, w)
    return cv2.resize(gray, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA)


def structure_tensor(gray: np.ndarray, sigma: float = 2.0) -> tuple[float, float, np.ndarray]:
    """Orientação dominante dos gradientes (rad), coerência global e histograma de 8 bins."""
    g = gray.astype(np.float32) / 255
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
    sxx, syy, sxy = Jxx.sum(), Jyy.sum(), Jxy.sum()
    theta = 0.5 * np.arctan2(2 * sxy, sxx - syy)  # direção do gradiente dominante
    coh = float(np.sqrt((sxx - syy) ** 2 + 4 * sxy ** 2) / (sxx + syy + 1e-9))
    ang = (0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)) % np.pi
    mag = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2)
    hist, _ = np.histogram(ang, bins=8, range=(0, np.pi), weights=mag)
    hist = hist / (hist.sum() + 1e-9)
    return float(theta), coh, hist


def corrugation_period(gray: np.ndarray, theta: float) -> tuple[float, float]:
    """Período (px na escala de trabalho) e força do pico do perfil ao longo do gradiente dominante."""
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), np.degrees(theta), 1.0)
    rot = cv2.warpAffine(gray.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    ch, cw = int(h * 0.2), int(w * 0.2)
    core = rot[ch:h - ch, cw:w - cw]
    if core.size == 0:
        return float("nan"), 0.0
    prof = core.mean(0)
    prof = prof - prof.mean()
    prof *= np.hanning(len(prof))
    P = np.abs(np.fft.rfft(prof)) ** 2
    if len(P) < 6:
        return float("nan"), 0.0
    P[:3] = 0  # ignora tendência
    k = int(np.argmax(P))
    strength = float(P[k] / (P.sum() + 1e-9))
    return float(len(prof) / k) if k > 0 else float("nan"), strength


def lsd_stats(gray: np.ndarray) -> dict:
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(gray)[0]
    area = gray.shape[0] * gray.shape[1]
    diag = float(np.hypot(*gray.shape))
    if lines is None or len(lines) == 0:
        return {"lsd_n_por_mpx": 0.0, "lsd_len_p50": 0.0, "lsd_len_p90": 0.0,
                "lsd_long_n": 0, "paralelismo": 0.0, "lsd_n_direcoes": 0}
    L = lines.reshape(-1, 4)
    dx, dy = L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]
    length = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx)) % 180
    hist, _ = np.histogram(ang, bins=18, range=(0, 180), weights=length)
    hist = hist / (hist.sum() + 1e-9)
    k = int(np.argmax(hist))
    paral = float(hist[k] + hist[(k - 1) % 18] + hist[(k + 1) % 18])
    # direções independentes entre segmentos longos (>= 15% da diagonal)
    long_ = length >= 0.15 * diag
    hl, _ = np.histogram(ang[long_], bins=18, range=(0, 180), weights=length[long_])
    peaks, used = 0, np.zeros(18, bool)
    for idx in np.argsort(hl)[::-1]:
        if hl[idx] <= 0 or used[idx]:
            continue
        peaks += 1
        for d in (-2, -1, 0, 1, 2):
            used[(idx + d) % 18] = True
    return {"lsd_n_por_mpx": float(len(L) / area * 1e6), "lsd_len_p50": float(np.median(length) / diag),
            "lsd_len_p90": float(np.percentile(length, 90) / diag), "lsd_long_n": int(long_.sum()),
            "paralelismo": paral, "lsd_n_direcoes": int(peaks)}


def describe(path: Path, bbox: tuple[int, int, int, int] | None) -> dict:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {}
    if bbox:
        x0, y0, x1, y1 = bbox
        if x1 - x0 > 16 and y1 - y0 > 16:
            gray = gray[y0:y1, x0:x1]
    g = _resize(gray)
    h, w = g.shape
    out = {}
    edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 1.2), 50, 150)
    for i in range(4):
        for j in range(4):
            cell = edges[i * h // 4:(i + 1) * h // 4, j * w // 4:(j + 1) * w // 4]
            out[f"canny_{i}{j}"] = float((cell > 0).mean())
    out["canny_total"] = float((edges > 0).mean())
    theta, coh, hist = structure_tensor(g)
    out["st_theta"], out["st_coerencia"] = theta, coh
    out.update({f"st_hist_{k}": float(v) for k, v in enumerate(hist)})
    out.update(lsd_stats(g))
    pts = cv2.goodFeaturesToTrack(g, maxCorners=500, qualityLevel=0.01, minDistance=5)
    out["shi_tomasi_por_mpx"] = float((0 if pts is None else len(pts)) / (h * w) * 1e6)
    per, strength = corrugation_period(g, theta)
    out["corr_periodo_rel"] = per / max(h, w) if per == per else float("nan")
    out["corr_pico"] = strength
    return out


def _star(a):
    return describe(*a)


def compute_geometry_table(df: pd.DataFrame, root: Path, bboxes: dict, workers: int = 4) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor
    ok = df[df["integridade"].isin(["ok", "convertido"])]
    args = [(Path(root) / p, bboxes.get(i)) for i, p in zip(ok["image_id"], ok["caminho"])]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_star, args, chunksize=32))
    g = pd.DataFrame(res)
    g.insert(0, "image_id", ok["image_id"].values)
    return g
