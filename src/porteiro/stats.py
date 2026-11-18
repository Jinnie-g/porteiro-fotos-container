"""Estatística usada em S1, S4, S5, S7 e S9 (sem dependências além de numpy/scipy)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def cohen_kappa(a, b, labels=None, weights: str | None = None) -> float:
    a, b = np.asarray(a), np.asarray(b)
    labels = list(labels) if labels is not None else sorted(set(a) | set(b))
    k = len(labels)
    idx = {v: i for i, v in enumerate(labels)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    n = O.sum()
    E = np.outer(O.sum(1), O.sum(0)) / n
    if weights is None:
        W = 1 - np.eye(k)
    else:
        i, j = np.meshgrid(range(k), range(k), indexing="ij")
        W = np.abs(i - j) / (k - 1) if weights == "linear" else ((i - j) / (k - 1)) ** 2
    return float(1 - (W * O).sum() / (W * E).sum()) if (W * E).sum() > 0 else float("nan")


def cramers_v(x, y) -> tuple[float, float]:
    t = pd.crosstab(pd.Series(np.asarray(x)), pd.Series(np.asarray(y)))
    if t.shape[0] < 2 or t.shape[1] < 2:
        return float("nan"), float("nan")
    chi2, p, _, _ = st.chi2_contingency(t, correction=False)
    n = t.values.sum()
    r, c = t.shape
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1)))), float(p)


def mutual_info(x, y) -> float:
    t = pd.crosstab(pd.Series(np.asarray(x)), pd.Series(np.asarray(y))).values.astype(float)
    p = t / t.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    u = st.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def group_bootstrap(groups: np.ndarray, fn, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """IC 95% por bootstrap de GRUPOS (reamostra grupos inteiros, não imagens)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    pos = {g: np.nonzero(groups == g)[0] for g in uniq}
    point = fn(np.arange(len(groups)))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([pos[g] for g in pick])
        v = fn(idx)
        if v == v:
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"), float("nan"))
    return float(point), float(lo), float(hi)
