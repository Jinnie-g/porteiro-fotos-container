"""S6-S8: modelo de triagem do zoom (Z0 = sem zoom contra o resto).

Entradas que existem na inferência: DINOv2 congelado e descritores OpenCV.
As caixas de avaria NÃO entram (não existem quando uma imagem nova chega).

Protocolo sem vazamento:
- validação cruzada agrupada (group_id) e estratificada, repetida;
- dentro de cada fold externo: C escolhido e Platt ajustado em predições
  fora-da-amostra de um CV interno agrupado; o fold externo só é previsto;
- o teste fixo só é usado uma vez, na confirmação final.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

QUALITY_COLS = ["blur_score", "tenengrad", "noise_sigma", "bw_efetiva", "lum_mean", "faixa_dinamica", "clip_frac",
                "contraste_rms", "entropia", "color_cast", "saturacao_media", "fill_frac", "cutout_n", "border_frac"]
OPENCV_COLS = ([f"canny_{i}{j}" for i in range(4) for j in range(4)] + ["canny_total", "st_coerencia"]
               + [f"st_hist_{k}" for k in range(8)]
               + ["lsd_n_por_mpx", "lsd_len_p50", "lsd_len_p90", "lsd_long_n", "paralelismo", "lsd_n_direcoes",
                  "shi_tomasi_por_mpx", "corr_pico"])
CS = (0.001, 0.01, 0.1, 1.0)


def load_dino(path) -> dict[str, np.ndarray]:
    z = np.load(path)
    ids = z["image_id"]
    return {"ids": ids, "cls": z["cls"].astype(np.float32), "mean": z["mean"].astype(np.float32)}


def feature_sets(df: pd.DataFrame, dino224: dict, dino448: dict) -> dict[str, np.ndarray]:
    """Matrizes alinhadas às linhas de df (pela coluna image_id)."""
    def rows(d):
        pos = {k: i for i, k in enumerate(d["ids"])}
        return np.array([pos[i] for i in df["image_id"]])
    r224, r448 = rows(dino224), rows(dino448)
    src = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(df[["source_id"]])
    qual = df[QUALITY_COLS].fillna(0).to_numpy(float)
    ocv = df[OPENCV_COLS].fillna(0).to_numpy(float)
    cls448, mean448 = dino448["cls"][r448], dino448["mean"][r448]
    return {
        "Z2a_so_fonte": src,
        "Z2b_trivial_metadados_fonte": np.hstack([qual, src]),
        "Z5_opencv": ocv,
        "Z3a_dino_cls_224": dino224["cls"][r224],
        "Z3b_dino_cls_448": cls448,
        "Z4a_dino_media_448": mean448,
        "Z4b_dino_concat_448": np.hstack([cls448, mean448]),
        "Z6_dino_concat_opencv": np.hstack([cls448, mean448, ocv]),
    }


def _model(kind: str, C: float):
    if kind == "svm":
        return make_pipeline(StandardScaler(), LinearSVC(C=C, class_weight="balanced", max_iter=20000))
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def _score(m, X):
    return m.decision_function(X)


def _inner_oof(X, y, g, C, kind, seed, k=4):
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(X, y, g):
        oof[te] = _score(_model(kind, C).fit(X[tr], y[tr]), X[te])
    return oof


class PlattModel:
    """Modelo linear + C escolhido + Platt, tudo ajustado só com o treino recebido."""

    def __init__(self, kind: str = "lr", seed: int = 0):
        self.kind, self.seed = kind, seed

    def fit(self, X, y, g):
        best = None
        for C in CS:
            oof = _inner_oof(X, y, g, C, self.kind, self.seed)
            ap = average_precision_score(y, oof)
            if best is None or ap > best[0]:
                best = (ap, C, oof)
        _, self.C, oof = best
        self.platt = LogisticRegression(C=1e6).fit(oof.reshape(-1, 1), y)
        self.m = _model(self.kind, self.C).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.platt.predict_proba(_score(self.m, X).reshape(-1, 1))[:, 1]


class Majority:
    def fit(self, X, y, g):
        self.p = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.p)


def repeated_oof(X, y, g, make, n_splits=5, n_repeats=5, seed=42) -> np.ndarray:
    """Probabilidades fora-da-amostra [n_repeats, n]."""
    out = np.zeros((n_repeats, len(y)))
    for r in range(n_repeats):
        for tr, te in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed + r).split(X, y, g):
            out[r, te] = make(r).fit(X[tr], y[tr], g[tr]).predict_proba(X[te])
    return out


def ece(y, p, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def metrics(y, p, thr: float = 0.5) -> dict:
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    return {
        "pr_auc": float(average_precision_score(y, p)) if y.any() else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan"),
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "precisao": tp / pred.sum() if pred.sum() else float("nan"),
        "revocacao": tp / y.sum() if y.sum() else float("nan"),
        "ece": ece(y, p), "brier": float(np.mean((p - y) ** 2)),
    }


def choose_thresholds(y, p, target: float = 0.95, min_n: int = 5) -> dict:
    """τ_alto: menor limiar com precisão de A >= alvo (máxima cobertura de A).
    τ_baixo: maior limiar com precisão de B (não-Z0) >= alvo. Entre os dois: abstenção."""
    order = np.unique(p)
    t_hi = None
    for t in order:
        sel = p >= t
        if sel.sum() >= min_n and y[sel].mean() >= target:
            t_hi = float(t)
            break
    t_lo = None
    for t in order[::-1]:
        sel = p < t
        if sel.sum() >= min_n and (1 - y[sel]).mean() >= target:
            t_lo = float(t)
            break
    t_hi = t_hi if t_hi is not None else 1.01
    t_lo = t_lo if t_lo is not None else 0.0
    if t_lo > t_hi:
        t_lo = t_hi
    return {"tau_alto": t_hi, "tau_baixo": t_lo}


def route(p: np.ndarray, t: dict) -> np.ndarray:
    return np.where(p >= t["tau_alto"], "sem_zoom", np.where(p < t["tau_baixo"], "com_zoom", "revisao"))


def coverage_risk(y, p) -> pd.DataFrame:
    """Curva cobertura × risco para A: ordena por p e mede o erro entre os aceitos."""
    o = np.argsort(-p)
    ys = y[o]
    k = np.arange(1, len(y) + 1)
    return pd.DataFrame({"cobertura_A": k / y.sum(), "n_aceitos": k, "risco": 1 - np.cumsum(ys) / k,
                         "limiar": p[o]})
