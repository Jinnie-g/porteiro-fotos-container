"""S1: análise exploratória do zoom SEM rótulos.

Proxies de enquadramento vêm das caixas de avaria. Servem só para explorar e
estratificar: as caixas não existem na inferência e NÃO entram no modelo.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .dedup import center_normalize


def box_proxies(df: pd.DataFrame, meta_dir: Path) -> pd.DataFrame:
    rows = []
    for iid in df["image_id"]:
        m = json.loads((Path(meta_dir) / f"{iid}.json").read_text(encoding="utf-8"))
        a = m.get("annotation") or {}
        W, H = a.get("width") or 640, a.get("height") or 640
        bx = a.get("boxes") or []
        asp = [b["width"] / b["height"] for b in bx if b["height"] > 0]
        # extensão da união das caixas (fração da imagem coberta pelo retângulo envolvente)
        if bx:
            x0 = min(b["x"] - b["width"] / 2 for b in bx) / W
            x1 = max(b["x"] + b["width"] / 2 for b in bx) / W
            y0 = min(b["y"] - b["height"] / 2 for b in bx) / H
            y1 = max(b["y"] + b["height"] / 2 for b in bx) / H
            env = max(0, x1 - x0) * max(0, y1 - y0)
        else:
            env = 0.0
        rows.append({"image_id": iid, "caixa_aspect_med": float(np.median(asp)) if asp else np.nan,
                     "caixa_envoltoria": env})
    out = df[["image_id", "caixa_area_max", "caixa_area_soma", "n_caixas"]].merge(pd.DataFrame(rows), on="image_id")
    out["log_area_max"] = np.log10(out["caixa_area_max"].clip(lower=1e-4))
    return out


def classe_principal(df: pd.DataFrame) -> pd.Series:
    cls = df[["n_rust", "n_dent", "n_hole"]].fillna(0)
    return cls.idxmax(axis=1).str.replace("n_", "", regex=False)


def dip_tests(values: pd.Series, by: pd.Series, min_n: int = 30) -> pd.DataFrame:
    """Teste de dip de Hartigan: p pequeno sugere mais de um pico (níveis de zoom)."""
    try:
        import diptest
    except ImportError:  # pragma: no cover
        return pd.DataFrame()
    rows = []
    for g, v in values.groupby(by):
        v = v.dropna().values
        if len(v) < min_n:
            continue
        d, p = diptest.diptest(v)
        rows.append({"grupo": g, "n": len(v), "dip": float(d), "p": float(p)})
    allv = values.dropna().values
    d, p = diptest.diptest(allv)
    rows.append({"grupo": "TODOS", "n": len(allv), "dip": float(d), "p": float(p)})
    return pd.DataFrame(rows)


def embedding_clusters(emb: np.ndarray, ks=(3, 4, 5, 6), seed: int = 42) -> tuple[pd.DataFrame, dict]:
    e = center_normalize(emb)
    res, labels = [], {}
    for k in ks:
        km = KMeans(k, n_init=10, random_state=seed).fit(e)
        labels[k] = km.labels_
        res.append({"k": k, "silhueta": float(silhouette_score(e, km.labels_, sample_size=min(2000, len(e)),
                                                                random_state=seed)),
                    "inercia": float(km.inertia_)})
    return pd.DataFrame(res), labels


def medoids(emb: np.ndarray, labels: np.ndarray, per: int = 8) -> dict[int, np.ndarray]:
    e = center_normalize(emb)
    out = {}
    for c in np.unique(labels):
        idx = np.nonzero(labels == c)[0]
        cen = e[idx].mean(0)
        out[int(c)] = idx[np.argsort(-(e[idx] @ cen))[:per]]
    return out
