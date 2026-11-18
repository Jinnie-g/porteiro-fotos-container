"""E4: caracterização (medir antes de corrigir), tabela da seção 4.

Todas as medidas de nitidez, ruído e iluminação são feitas só na região de
conteúdo, para a borda preta não distorcer o resultado.
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

SIDES = ("top", "bottom", "left", "right")


def _dark_run(lines_mean: np.ndarray, lines_std: np.ndarray, lum_max: float, std_max: float) -> int:
    n = 0
    for m, s in zip(lines_mean, lines_std):
        if m < lum_max and s < std_max:
            n += 1
        else:
            break
    return n


def detect_borders(gray: np.ndarray, lum_max: float = 12, std_max: float = 6,
                   min_frac: float = 0.01) -> dict:
    """Bordas pretas por lado + colagem (faixa preta no meio).

    Uma borda só conta se tiver largura >= min_frac e degrau nítido para o
    conteúdo; se a passagem for gradual é conteúdo escuro real (noite, sombra)
    e vira a flag `borda_gradual`.
    """
    H, W = gray.shape
    g = gray.astype(np.float32)
    rows_m, rows_s = g.mean(1), g.std(1)
    cols_m, cols_s = g.mean(0), g.std(0)
    seq = {
        "top": (rows_m, rows_s), "bottom": (rows_m[::-1], rows_s[::-1]),
        "left": (cols_m, cols_s), "right": (cols_m[::-1], cols_s[::-1]),
    }
    out, gradual = {}, False
    for side, (m, s) in seq.items():
        dim = H if side in ("top", "bottom") else W
        n = _dark_run(m, s, lum_max, std_max)
        if n < max(1, int(round(min_frac * dim))) or n >= dim:
            n = 0 if n < dim else n
        elif n + 3 < dim:
            # degrau: as 3 linhas seguintes precisam ser claramente mais claras
            step = float(np.mean(m[n:n + 3]) - np.mean(m[max(0, n - 3):n]))
            if step < 2 * lum_max:
                gradual, n = True, 0
        out[f"border_{side}"] = n / dim
    y0, y1 = int(out["border_top"] * H), H - int(out["border_bottom"] * H)
    x0, x1 = int(out["border_left"] * W), W - int(out["border_right"] * W)
    out["bbox_conteudo"] = [x0, y0, x1, y1]
    out["border_frac"] = 1 - ((x1 - x0) * (y1 - y0)) / (W * H) if x1 > x0 and y1 > y0 else 1.0
    out["borda_gradual"] = gradual

    # colagem: faixa escura e uniforme longe das bordas, com conteúdo claro dos dois lados
    def _band(m, s, dim):
        dark = (m < lum_max) & (s < std_max)
        lo, hi = int(0.1 * dim), int(0.9 * dim)
        minw = max(2, int(min_frac * dim))
        k = lo
        while k < hi:
            if not dark[k]:
                k += 1
                continue
            e = k
            while e < hi and dark[e]:
                e += 1
            if e - k >= minw:
                left = float(np.mean(m[max(0, k - 4):k])) if k > 0 else 0.0
                right = float(np.mean(m[e:e + 4])) if e < dim else 0.0
                if left > 3 * lum_max and right > 3 * lum_max:
                    return True
            k = e
        return False
    out["colagem"] = bool(_band(rows_m, rows_s, H) or _band(cols_m, cols_s, W))
    return out


def detect_fill(gray: np.ndarray, sd_max: float = 0.8, tol: int = 2) -> dict:
    """Preenchimento artificial (rotação, cisalhamento, perspectiva de augmentation).

    Regiões de cor exatamente constante (o preenchimento medido tem desvio 0)
    ligadas a cantos ou meios das bordas. Mediana 5x5 antes, para tolerar o
    ruído sal-e-pimenta que a augmentation aplica por cima. Branco saturado
    (>= 250) é tratado como estouro de céu, não como preenchimento.
    Não depende de a região ser retangular, ao contrário de `detect_borders`.
    """
    H, W = gray.shape
    med = cv2.medianBlur(gray, 5)
    f = med.astype(np.float32)
    mu = cv2.blur(f, (7, 7))
    sd = np.sqrt(np.maximum(cv2.blur(f * f, (7, 7)) - mu * mu, 0))
    flat = sd < sd_max
    seeds = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1), (0, W // 2), (H - 1, W // 2), (H // 2, 0), (H // 2, W - 1)]
    mask = np.zeros((H, W), bool)
    corners = 0
    colors = []
    for k, (y, x) in enumerate(seeds):
        if mask[y, x] or not flat[y, x] or med[y, x] >= 250:
            continue
        c = int(med[y, x])
        cand = (flat & (np.abs(f - c) <= tol)).astype(np.uint8)
        n, lab = cv2.connectedComponents(cand, connectivity=4)
        comp = lab == lab[y, x]
        if comp.mean() < 0.002:
            continue
        mask |= comp
        colors.append(c)
        corners += k < 4
    if mask.any():  # compensa os 3 px que a janela 7x7 do desvio local erode na fronteira
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    frac = float(mask.mean())
    cor = "nenhum" if not colors else ("preto" if np.median(colors) < 20 else
                                        "branco" if np.median(colors) > 235 else "cinza")
    return {"fill_frac": frac, "fill_cor": cor, "fill_cantos": int(corners), "_fill_mask": mask}


def detect_cutout(gray: np.ndarray, min_area: float = 0.001) -> dict:
    """Cutout de augmentation: retângulos alinhados aos eixos, de cor exatamente constante."""
    H, W = gray.shape
    med = cv2.medianBlur(gray, 3)
    f = med.astype(np.float32)
    mu = cv2.blur(f, (5, 5))
    sd = np.sqrt(np.maximum(cv2.blur(f * f, (5, 5)) - mu * mu, 0))
    flat = (sd < 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(flat, connectivity=4)
    found, area = 0, 0
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if a < min_area * H * W or w < 6 or h < 6:
            continue
        if a / (w * h) < 0.9:
            continue
        vals = med[lab == k]
        if vals.std() > 1.5:
            continue
        touches = x == 0 or y == 0 or x + w == W or y + h == H
        if touches and (vals.mean() >= 250 or vals.mean() <= 5):
            continue  # céu estourado ou borda preta, não cutout
        found += 1
        # a janela 5x5 do desvio local erode 2 px de cada lado; devolve à área real
        area += min(W, w + 4) * min(H, h + 4)
    return {"cutout_n": found, "cutout_frac": area / (H * W)}


def safe_to_crop(borders: dict, boxes_xyxy: list[tuple[float, float, float, float]]) -> bool:
    """Recorte seguro: há borda nítida e nenhuma anotação cai dentro dela."""
    if borders["border_frac"] <= 0 or borders["borda_gradual"] or borders["colagem"]:
        return False
    x0, y0, x1, y1 = borders["bbox_conteudo"]
    tol = 2
    return all(bx0 >= x0 - tol and by0 >= y0 - tol and bx1 <= x1 + tol and by1 <= y1 + tol
               for bx0, by0, bx1, by1 in boxes_xyxy)


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkær (1996): estimador rápido do desvio do ruído."""
    H, W = gray.shape
    if H < 3 or W < 3:
        return float("nan")
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = cv2.filter2D(gray.astype(np.float32), -1, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (W - 2) * (H - 2)))


def sharpness(gray: np.ndarray, size: int = 512) -> tuple[float, float]:
    """Variância do Laplaciano e Tenengrad em tamanho fixo (lado maior = size)."""
    h, w = gray.shape
    s = size / max(h, w)
    g = cv2.resize(gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(g, cv2.CV_64F).var()
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1)
    ten = float(np.mean(gx ** 2 + gy ** 2))
    return float(lap), ten


def effective_bandwidth(gray: np.ndarray) -> float:
    """Fração da energia espectral acima de 1/4 da frequência de Nyquist.

    Imagem ampliada artificialmente tem pouca energia em alta frequência.
    """
    g = gray.astype(np.float32) - gray.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = g.shape
    yy, xx = np.ogrid[-(h // 2):h - h // 2, -(w // 2):w - w // 2]
    r = np.sqrt((yy / (h / 2)) ** 2 + (xx / (w / 2)) ** 2)
    tot = F.sum()
    return float(F[r > 0.25].sum() / tot) if tot > 0 else 0.0


def jpeg_quality_est(path: Path) -> float | None:
    """Estimativa da qualidade JPEG pela tabela de quantização de luminância."""
    try:
        with Image.open(path) as im:
            q = getattr(im, "quantization", None)
            if not q:
                return None
            t = np.array(q[0], np.float32)
    except Exception:
        return None
    std = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57,
                    69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64,
                    81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
                   np.float32)
    # PIL devolve em ordem zigue-zague; a razão média independe da ordem
    scale = float(np.mean(t) / np.mean(std) * 100)
    qual = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return float(np.clip(qual, 1, 100))


def photometry(bgr: np.ndarray) -> dict:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    p1, p99 = np.percentile(gray, [1, 99])
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    means = bgr.reshape(-1, 3).mean(0)  # B, G, R
    gw = means.mean()
    color_cast = float(np.abs(means - gw).max() / (gw + 1e-6))
    return {
        "lum_mean": float(gray.mean()), "p1": float(p1), "p99": float(p99),
        "faixa_dinamica": float(p99 - p1),
        "clip_frac": float(((gray <= 2) | (gray >= 253)).mean()),
        "contraste_rms": float(gray.std()),
        "entropia": ent, "color_cast": color_cast,
        "saturacao_media": float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].mean()),
    }


def measure_image(path: Path, boxes_xyxy: list, cfg_borders: dict) -> dict:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"q_erro": "leitura"}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    b = detect_borders(gray, cfg_borders["lum_max"], cfg_borders["std_max"], cfg_borders["min_frac"])
    fl = detect_fill(gray)
    valid = ~fl.pop("_fill_mask")
    x0, y0, x1, y1 = b["bbox_conteudo"]
    crop = bgr[y0:y1, x0:x1] if (x1 - x0) > 16 and (y1 - y0) > 16 else bgr
    cg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap, ten = sharpness(cg)
    out = {**b, **fl, "safe_to_crop": safe_to_crop(b, boxes_xyxy),
           # rotação/cisalhamento artificial: preenchimento que não é faixa retangular
           "aug_geometrica": bool(fl["fill_frac"] - b["border_frac"] > 0.02 and fl["fill_cantos"] >= 1
                                  and fl["fill_cor"] in ("preto", "cinza")),
           **detect_cutout(gray),
           "megapixels": W * H / 1e6, "aspect": W / H,
           "blur_score": lap, "tenengrad": ten, "noise_sigma": noise_sigma(cg),
           "bw_efetiva": effective_bandwidth(cv2.resize(cg, (256, 256), interpolation=cv2.INTER_AREA)),
           "jpeg_quality_est": jpeg_quality_est(path),
           **photometry(crop if valid.mean() > 0.98 else bgr[valid][:, None, :])}
    out["bbox_conteudo"] = ",".join(map(str, out["bbox_conteudo"]))
    flags = []
    if min(W, H) < 128:
        flags.append("miniatura")
    if max(W / H, H / W) > 3:
        flags.append("proporcao_extrema")
    out["flags"] = ";".join(flags)
    return out


def compute_quality_table(df: pd.DataFrame, root: Path, boxes: dict, cfg_borders: dict,
                          workers: int = 4) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor
    ok = df[df["integridade"].isin(["ok", "convertido"])]
    args = [(Path(root) / p, boxes.get(i, []), cfg_borders) for i, p in zip(ok["image_id"], ok["caminho"])]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_measure_star, args, chunksize=32))
    q = pd.DataFrame(res)
    q.insert(0, "image_id", ok["image_id"].values)
    # blur normalizado por percentil dentro de cada fonte (seção 4)
    q = q.merge(df[["image_id", "source_id"]], on="image_id")
    q["blur_pct_fonte"] = q.groupby("source_id")["blur_score"].rank(pct=True)
    return q.drop(columns=["source_id"])


def _measure_star(a):
    return measure_image(*a)
